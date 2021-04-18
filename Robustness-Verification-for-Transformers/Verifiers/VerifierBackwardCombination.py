# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import copy
import gc
from typing import Tuple

from Verifiers import Verifier, Bounds
from Verifiers.ConvexCombination import Container, ConvexCombinationData
from Verifiers.Edge import *
from Verifiers.memory import get_gpu_memory_usage
from Verifiers.utils import check
from Verifiers.autograd import LayerGradientManager, LambdaBoundsGradientTracker, get_val_and_jacobians

epsilon = 1e-12


def detach_jacobians(jacobians_list):
    return tuple([
        tuple([x.detach() for x in y]) for y in jacobians_list
    ])


class VerifierBackwardConvexCombination(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierBackwardConvexCombination, self).__init__(args, target, logger)
        self.controller = None
        self.use_forward = ("baf" in args.method)
        self.empty_cache = args.empty_cache
        self.should_compute_jacobians = True
        self.gradient_tracker = None
        self.convex_combination_elements = None
        self.handle_one_lambda_at_the_time = args.handle_one_lambda_at_the_time
        self.reset()

    ##################################################################
    # Lifecycle management - setup and delete memory + key variables #
    ##################################################################

    def start_computing_jacobians(self):
        self.should_compute_jacobians = True

    def stop_computing_jacobians(self):
        self.should_compute_jacobians = False

    def start_verification_new_input(self):
        self.reset()

    def reset(self):
        self.delete_convex_combination_elements()
        self.setup_new_verification_round()
        self.convex_combination_elements = Container()
        self.is_initialized = False

    def setup_new_verification_round(self):
        with torch.no_grad():
            self.delete_gradient_tracker()
            self.gradient_tracker = LambdaBoundsGradientTracker(
                use_cuda=not self.args.cpu,
                handle_one_lambda_at_the_time=self.handle_one_lambda_at_the_time
            )
            self.number_relevant_lambdas = 0

    def delete_convex_combination_elements(self):
        if self.convex_combination_elements is not None and not self.convex_combination_elements.is_empty():
            for element in self.convex_combination_elements.get_all_elements():
                element: ConvexCombinationData
                del element.lambdas_verification
                del element.lambdas_optimization
                del element.hyperplanes_coeffs

            del self.convex_combination_elements
            self.convex_combination_elements = None
            torch.cuda.empty_cache()

    def delete_gradient_tracker(self):
        if self.gradient_tracker is not None:
            self.gradient_tracker.delete_all()
            del self.gradient_tracker
            self.gradient_tracker = None
            torch.cuda.empty_cache()

    def cleanup_after_iteration(self):
        # We don't delete the convex combination elements since we will need them
        self.controller.cleanup()
        self.delete_gradient_tracker()
        torch.cuda.empty_cache()

    def cleanup_after_verification(self):
        self.controller.cleanup()
        self.delete_gradient_tracker()
        self.delete_convex_combination_elements()
        torch.cuda.empty_cache()

    #####################
    # Deal with lambdas #
    #####################

    def setup_lambdas(self, length: int):
        for encoding_layer in self.encoding_layers:
            num_attention_heads = encoding_layer.attention.self.num_attention_heads
            num_elements = (length, num_attention_heads * length)
            lambdas_optimization = get_initial_lambdas(num_elements, device='cpu')
            lambdas_verification = to_dev(lambdas_optimization, self.args)  # Either GPU or CPU
            self.convex_combination_elements.add_element(ConvexCombinationData(
                lambdas_optimization=lambdas_optimization,
                lambdas_verification=lambdas_verification,
                hyperplanes_coeffs=None
            ))

        self.convex_combination_elements.mark_initialization_over()
        self.convex_combination_elements.reset_cursor()
        self.is_initialized = True

    def get_lambdas_for_verification(self) -> List[torch.Tensor]:
        return [convex_combination_data.lambdas_verification for convex_combination_data in self.convex_combination_elements.get_all_elements()]

    def get_lambdas_for_optimizer(self) -> List[torch.Tensor]:
        return [convex_combination_data.lambdas_optimization for convex_combination_data in self.convex_combination_elements.get_all_elements()]

    def update_verification_lambdas(self):
        with torch.no_grad():
            for convex_combination_data in self.convex_combination_elements.get_all_elements():
                convex_combination_data: ConvexCombinationData
                del convex_combination_data.lambdas_verification
                convex_combination_data.lambdas_verification = to_dev(convex_combination_data.lambdas_optimization, self.args)

    ###################
    # Do verification #
    ###################

    def verify_safety(self, example, embeddings: torch.Tensor, index: int, eps: float):
        errorType = OSError if self.debug else AssertionError
        #torch.autograd.set_detect_anomaly(True)

        # cannot accept a batch
        embeddings = embeddings[0]

        length, dim = embeddings.shape[0], embeddings.shape[1]
        self.reset()
        self.setup_lambdas(length)
        try:
            with torch.no_grad():
                label = example["label"]

                self.stop_computing_jacobians()
                gc.collect(); torch.cuda.empty_cache()
                concretized_bounds, safety = self.run_verifier(embeddings, eps, index, label)

                if safety:
                    self.cleanup_after_verification()
                    return True

                lr_fn = lambda e: 100 * 0.99 ** e

                lambdas_for_optimizer_list = self.get_lambdas_for_optimizer()
                optimizer = torch.optim.Adam(lambdas_for_optimizer_list)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

                print("Starting optimization!")
                max_iter = self.args.max_optim_iters
                for epoch in range(max_iter):
                    gc.collect(); torch.cuda.empty_cache()
                    self.start_computing_jacobians()
                    concretized_bounds, safety = self.run_verifier(embeddings, eps, index, label)

                    if safety:
                        print("\nDid %d round of optimization: success" % epoch)
                        return True

                    # Compute loss and manually set the gradient
                    for lambdas_index, lambdas in enumerate(lambdas_for_optimizer_list):
                        layer_pos = concretized_bounds.get_pos()
                        if label == 0:
                            loss = -concretized_bounds.l[0][0]  # We want to increase the value of this, so the loss which we minimize is the negation
                            grad = -self.gradient_tracker.get_lambda_jacobians(layer_pos, lambdas_index)[0][0][0].float()
                        else:
                            loss = concretized_bounds.u[0][0]  # We want to increase the value of this, so the loss which we minimize is the negation
                            grad = self.gradient_tracker.get_lambda_jacobians(layer_pos, lambdas_index)[1][0][0].float()

                        assert not torch.isnan(grad).any(), "Gradient has NaNs"

                        # Without the clone, I get "RuntimeError: Can't detach views in-place. Use detach() instead"
                        # when I do optimizer.zero_grad() later. The clone solves this, as decribed in
                        # https://discuss.pytorch.org/t/manually-applied-gradient-to-neural-network-raises-error-cant-detach-views-in-place/69031
                        lambdas.grad = grad.clone().detach()

                    print("\rEpoch %d: Loss = %s    (for eps: %s)" % (epoch, loss, eps), end="")

                    # Don't zero out the gradients, since we manually set them above!
                    optimizer.step()
                    scheduler.step()

                    self.update_verification_lambdas()
                    self.cleanup_after_iteration()
                    for lambdas in lambdas_for_optimizer_list:
                        del lambdas.grad
                    del concretized_bounds
                    del safety
                    del loss
                    del grad

                print("\nAfter %d round of optimization: no success" % (max_iter - 1))
                self.cleanup_after_verification()

                return False
        except errorType as err:  # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            print(err)
            return False

    def run_verifier(self, embeddings: torch.Tensor, eps: float, index: int, label: int) -> Tuple[Layer, bool]:
        self.setup_new_verification_round()

        # def check(*args, **kwargs): pass

        bounds = self._bound_input(embeddings, index=index, eps=eps)  # hard-coded yet
        check("embedding", l=bounds.l, u=bounds.u, std=self.std["embedding_output"][0], verbose=self.debug)
        if self.verbose:
            bounds.print("embedding")

        for i, layer in enumerate(self.encoding_layers):
            attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer)

            std = self.std["attention_scores"][i][0]
            std = std.transpose(0, 1).reshape(1, std.shape[1], -1)
            check("layer {} attention_scores".format(i), l=attention_scores.l, u=attention_scores.u, std=std[0], verbose=self.debug)
            std = self.std["attention_probs"][i][0]
            std = std.transpose(0, 1).reshape(1, std.shape[1], -1)
            check("layer {} attention_probs".format(i), l=attention_probs.l, u=attention_probs.u, std=std[0], verbose=self.debug)
            check("layer {}".format(i), l=bounds.l, u=bounds.u, std=self.std["encoded_layers"][i][0], verbose=self.debug)

        bounds = self._bound_pooling(bounds, self.pooler)
        check("pooled output", l=bounds.l[:1], u=bounds.u[:1], std=self.std["pooled_output"][0], verbose=self.debug)
        concretized_bounds, safety = self._bound_classifier(bounds, self.classifier, label)

        self.convex_combination_elements.reset_cursor()

        return concretized_bounds, safety

    def sum_nested_tuples(self, val1, val2):
        """
        sum_nested_tuples(None, ((1, 2, 3), (4, 5, 6))) == ((1, 2, 3), (4, 5, 6))
        sum_nested_tuples(((2, 3 , 4), (4, 5, 6)),
                          ((2, -2, 3), (0, 1, 0))) == ((4, 1, 7), (4, 6, 6))
        """
        with torch.no_grad():
            if val1 is None:
                return val2
            else:
                return tuple([
                    tuple([x + y for (x, y) in zip(x_list, y_list)])
                    for (x_list, y_list) in zip(val1, val2)
                ])

    def get_bounds(self, input_bounds: Layer, edge: Edge, length=None, dim=None, at_least_one_next_layer_params_depend_on_bounds=True):
        if not self.should_compute_jacobians:
            return input_bounds.next(edge, length, dim)

        # Insight: the only case where the lambdas dB(i)/dλ is useful is if later we need it
        # Later, we need it only if
        # 1) we're in the last layer
        # 2) the next layer's parameters depend on the B(i), and we need to compute dp(i + 1) / dλ
        #    it's useful because dp(i + 1) / dλ = dp(i + 1) / dB(i)  *   dB(i) / dλ
        #    however, if we know in advance that the parameters of the layer afterwards don't depend on
        #    the bounds of the previous layer, it may not be required to compute them :)
        #    this will save time, memory and may reduce the number of computations

        if self.number_relevant_lambdas == 0 or not at_least_one_next_layer_params_depend_on_bounds:  # Not in a layer that requires lambdas
            # print("Getting bounds for edge %s (not computing jacobian)" % edge)
            return input_bounds.next(edge, length, dim)

        torch.cuda.empty_cache()

        # print("Getting bounds for edge %s (computing jacobian)" % edge)

        output_bounds: Optional[Layer] = None
        layer_gradient_manager: Optional[LayerGradientManager] = None

        def magic_function(*lambdas_list):
            nonlocal output_bounds
            nonlocal layer_gradient_manager
            # noinspection PyTypeChecker
            layer_gradient_manager = LayerGradientManager(lambdas_list, self.gradient_tracker)
            output_bounds = input_bounds.next(edge, length, dim, layer_gradient_manager=layer_gradient_manager)
            return output_bounds.l, output_bounds.u

        print("Memory used before doing backsubstitution for layer %i: %s MB" %
              (input_bounds.backsubstitution_computer.get_num_layers(), get_gpu_memory_usage()))
        jacobians_list = None
        relevant_lambdas = self.get_lambdas_for_verification()[:self.number_relevant_lambdas]

        # Prepare for the loop
        self.gradient_tracker.start_new_iteration()
        finished = self.gradient_tracker.have_injected_all_gradients()

        # Accumulate the gradients (we may inject the gradients one by one instead of all at once)
        # Currently it's all at once but that's easily changeable
        while not finished:
            self.gradient_tracker.prepare_next_gradients_to_be_injected()

            # Get the value and the jacobians
            _, new_jacobians_list = get_val_and_jacobians(magic_function, *relevant_lambdas)

            # Undo effect of adding gradient
            layer_gradient_manager.restore_original_bounds()
            layer_gradient_manager = None

            # Detach things
            output_bounds.l, output_bounds.u = output_bounds.l.detach(), output_bounds.u.detach()
            new_jacobians_list = detach_jacobians(new_jacobians_list)

            # Accumulate jacobians
            jacobians_list = self.sum_nested_tuples(jacobians_list, new_jacobians_list)

            # Prepare for new iteration
            self.gradient_tracker.mark_end_iteration()
            finished = self.gradient_tracker.have_injected_all_gradients()

            # Remove the last layer to avoid having it multiple times
            if not finished:
                input_bounds.backsubstitution_computer.layers.pop()

        print("Memory used after doing backsubstitution for layer %i: %s MB" %
              (input_bounds.backsubstitution_computer.get_num_layers(), get_gpu_memory_usage()))
        for lambda_index in range(len(relevant_lambdas)):
            #                        dl / dλ                          du / dλ
            jacobians = (jacobians_list[0][lambda_index], jacobians_list[1][lambda_index])
            self.gradient_tracker.add_bounds_lambda_grad(
                lambda_index=lambda_index,
                layer_pos=output_bounds.get_pos(),
                lambda_layer_bounds_jacobians=jacobians
            )

        return output_bounds

    def get_bounds_direct_sum(self, output: Layer,  bounds_needed_for_next_layer_params=True):
        assert all([isinstance(parent, EdgeDirect) for parent in output.parents]), "Not all parents edges are EdgeDirect"

        # print("Getting bounds for Layer with Direct Edges")
        output.compute()

        if not self.should_compute_jacobians:
            return output

        # No need to compute the jacobians here
        if self.number_relevant_lambdas == 0:  # Not yet in a layer that requires lambdas
            return output

        if not bounds_needed_for_next_layer_params:
            return output

        # The thing that's being computed is B = A1 + A2 + ... + An
        # The gradient dB/dλ = dA1/dλ + ... + dAn/dλ
        # So we can simply compute it directly very simply.
        relevant_lambdas = self.get_lambdas_for_verification()[:self.number_relevant_lambdas]

        for lambda_index in range(len(relevant_lambdas)):
            # dl / dλ     du / dλ
            jacobian_l, jacobian_u = None, None

            for edge_direct in output.parents:
                edge_direct: EdgeDirect
                par_layer = edge_direct.par

                try:
                    jacobian_l_par, jacobian_u_par = self.gradient_tracker.get_lambda_jacobians(par_layer.get_pos(), lambda_index)
                    if jacobian_l is None:
                        jacobian_l, jacobian_u = jacobian_l_par, jacobian_u_par
                    else:
                        jacobian_l += jacobian_l_par
                        jacobian_u += jacobian_u_par
                except KeyError:
                    # On the first encoding layer, one of the parents doesn't depend on the lambdas, and therefore
                    # there isn't any d parent layer / dλ
                    pass

            self.gradient_tracker.add_bounds_lambda_grad(
                lambda_index=lambda_index,
                layer_pos=output.get_pos(),
                lambda_layer_bounds_jacobians=(jacobian_l, jacobian_u)
            )

        return output

    def _bound_input(self, embeddings: torch.Tensor, index: int, eps: float) -> Layer:
        # No need to compute jacobians here
        length, dim = embeddings.shape[0], embeddings.shape[1]

        self.controller = BacksubstitutionComputer(self.args, eps)

        layer = Layer(self.args, self.controller, length, dim)
        layer.add_edge(EdgeInput(self.args, self.controller, embeddings, index))
        layer.compute()

        layer = self._bound_layer_normalization(
            layer, self.embeddings.LayerNorm,
            final_layer_bounds_needed_for_next_layer_params=False  # Will go into encoding layer, which does Dense Layers on this
        )

        return layer

    def _bound_layer_normalization(self, layer: Layer, bert_normalizer, debug=False, final_layer_bounds_needed_for_next_layer_params=True):
        if self.layer_norm == "no":
            return layer

        length, dim = layer.length, layer.dim

        if self.args.cpu:
            eye = torch.eye(dim)
            # zeros = torch.zeros(dim, dim)
            ones = torch.ones((dim, dim))
        else:
            eye = torch.eye(dim).cuda()
            # zeros = torch.zeros(dim, dim).cuda()
            ones = torch.ones((dim, dim)).cuda()

        w_avg = ones / layer.dim  # w_avg = 1/n x1 - ... - 1/n xn = avg(x's)

        # Computes x - avg(x's)
        minus_mu = self.get_bounds(layer, EdgeDense(self.args, self.controller, layer, w=eye - w_avg, b=0.),
                                   at_least_one_next_layer_params_depend_on_bounds=(self.layer_norm == "standard"))

        if self.layer_norm == "standard":
            # Compute the variance:
            # 1: (x - avg(x's))²
            # 2: sum 1/n over of term in (1)
            # 3: sqrt of (2)
            # 4: divide by (3) (e.g. divide by the standard deviation)
            minus_mu_sqr = self.get_bounds(minus_mu, EdgeSqr(self.args, self.controller, minus_mu),
                                           at_least_one_next_layer_params_depend_on_bounds=False)
            variance = self.get_bounds(minus_mu_sqr, EdgeDense(
                self.args, self.controller, minus_mu_sqr,
                w=w_avg, b=epsilon
            ), at_least_one_next_layer_params_depend_on_bounds=True)

            if self.verbose:
                variance.print("variance")

            std = self.get_bounds(variance, EdgeSqrt(self.args, self.controller, variance),
                                  at_least_one_next_layer_params_depend_on_bounds=True)

            normalized = self.get_bounds(minus_mu, EdgeDivide(self.args, self.controller, minus_mu, std),
                                         at_least_one_next_layer_params_depend_on_bounds=False)
        else:
            assert (self.layer_norm == "no_var"), "if layer_norm is not 'standard', then it has to be 'no_var'"
            normalized = minus_mu

        normalized = self.get_bounds(
            normalized,
            EdgeLinear(self.args, self.controller, normalized, bert_normalizer.weight, bert_normalizer.bias),
            at_least_one_next_layer_params_depend_on_bounds=final_layer_bounds_needed_for_next_layer_params
        )

        return normalized

    def _bound_layer(self, bounds_input: Layer, bert_layer) -> Tuple[Layer, Layer, Layer]:
        # Here's what they do:
        # 1) Compute edges of self-attention
        # 2) Multiply by the attention's output matrix
        # 3) Sum the original x and the output of (2) and then normalize
        # 4) Multiply by an intermediate matrix
        # 5) Apply an activation (ReLU)
        # 6) Multiply by an output matrix
        # 7) Add the attention values computed in 2
        # 8) Normalize
        # 9) Return the output of 8, the attention scores and the attention probs

        attention_scores, attention_probs, attention = self._bound_attention(bounds_input, bert_layer.attention)

        attention = self.get_bounds(attention, EdgeDense(self.args, self.controller, attention, dense=bert_layer.attention.output.dense),
                                    at_least_one_next_layer_params_depend_on_bounds=False)  # bounds_input will go only go into Dense layers

        attention_residual = Layer(self.args, self.controller, attention.length, attention.dim)
        attention_residual.add_edge(EdgeDirect(self.args, self.controller, attention))
        attention_residual.add_edge(EdgeDirect(self.args, self.controller, bounds_input))
        attention_residual = self.get_bounds_direct_sum(
            attention_residual, bounds_needed_for_next_layer_params=False  # Because normalization starts with EdgeDense
        )

        attention = self._bound_layer_normalization(
            attention_residual, bert_layer.attention.output.LayerNorm,
            debug=True, final_layer_bounds_needed_for_next_layer_params=False  # Used in EdgeDense and normalization (starts with Dense)
        )

        intermediate = self.get_bounds(attention, EdgeDense(
            self.args, self.controller, attention, dense=bert_layer.intermediate.dense
        ), dim=bert_layer.intermediate.dense.weight.shape[0], at_least_one_next_layer_params_depend_on_bounds=True)
        assert (self.hidden_act == "relu"), "Only ReLU activation is supported"
        intermediate = self.get_bounds(intermediate, EdgeRelu(self.args, self.controller, intermediate),
                                       at_least_one_next_layer_params_depend_on_bounds=False)

        dense = self.get_bounds(
            intermediate, EdgeDense(self.args, self.controller, intermediate, dense=bert_layer.output.dense),
            dim=bert_layer.output.dense.weight.shape[0],
            at_least_one_next_layer_params_depend_on_bounds=False
        )

        dense_residual = Layer(self.args, self.controller, dense.length, dense.dim)
        dense_residual.add_edge(EdgeDirect(self.args, self.controller, dense))
        dense_residual.add_edge(EdgeDirect(self.args, self.controller, attention))
        dense_residual = self.get_bounds_direct_sum(dense_residual, bounds_needed_for_next_layer_params=False)

        output = self._bound_layer_normalization(
            dense_residual, bert_layer.output.LayerNorm,
            final_layer_bounds_needed_for_next_layer_params=False
            # Will go into another encoding layer or pooling, both of which apply only EdgeDense on the input
        )

        return attention_scores, attention_probs, output

    def _bound_attention(self, bounds_input: Layer, bert_attention) -> Tuple[Layer, Layer, Layer]:
        # This in practice is not the case, but the explanation for the forwards case is already present in the
        # VerifierForward code, so I don't repeat it here). Here are the steps:
        # 1) Bound the queries, keys and values for each attention head
        # 2) Do the dot product between the keys and the queries, and ther normalize by 1 / sqrt(attention head size)
        #    obtaining the attention scores
        # 3) Compute the softmax of the attention scores, obtaining the attention probs
        # 4) They transpose the values
        # 5) They do a dot-product like thing between the attention scores and the values
        # 6) They return the attention scores, attention probs and the result of (7)

        num_attention_heads = bert_attention.self.num_attention_heads
        attention_head_size = bert_attention.self.attention_head_size

        query = self.get_bounds(bounds_input, EdgeDense(self.args, self.controller, bounds_input, dense=bert_attention.self.query),
                                at_least_one_next_layer_params_depend_on_bounds=True)
        key = self.get_bounds(bounds_input, EdgeDense(self.args, self.controller, bounds_input, dense=bert_attention.self.key),
                              at_least_one_next_layer_params_depend_on_bounds=True)
        value = self.get_bounds(bounds_input, EdgeDense(self.args, self.controller, bounds_input, dense=bert_attention.self.value),
                                at_least_one_next_layer_params_depend_on_bounds=False  # Because it's a transpose afterwards
                                )

        # IMPORTANT: we can process the self-attention using either a forward pass or a backwards pass
        if self.use_forward:  # This is for the "Backwards and Forwards" method
            query = Bounds(
                self.args, self.controller.p, self.controller.eps,
                lw=query.final_lw.unsqueeze(0).transpose(-1, -2), lb=query.final_lb.unsqueeze(0),
                uw=query.final_uw.unsqueeze(0).transpose(-1, -2), ub=query.final_ub.unsqueeze(0)
            )

            key = Bounds(
                self.args, self.controller.p, self.controller.eps,
                lw=key.final_lw.unsqueeze(0).transpose(-1, -2), lb=key.final_lb.unsqueeze(0),
                uw=key.final_uw.unsqueeze(0).transpose(-1, -2), ub=key.final_ub.unsqueeze(0)
            )

            value = Bounds(
                self.args, self.controller.p, self.controller.eps,
                lw=value.final_lw.unsqueeze(0).transpose(-1, -2), lb=value.final_lb.unsqueeze(0),
                uw=value.final_uw.unsqueeze(0).transpose(-1, -2), ub=value.final_ub.unsqueeze(0)
            )

            # copied from the forward framework
            def transpose_for_scores(x):
                def transpose_w(x):
                    return x.reshape(x.shape[0], x.shape[1], x.shape[2], num_attention_heads, attention_head_size) \
                            .permute(0, 3, 1, 2, 4) \
                            .reshape(-1, x.shape[1], x.shape[2], attention_head_size)

                def transpose_b(x):
                    return x.reshape(x.shape[0], x.shape[1], num_attention_heads, attention_head_size) \
                            .permute(0, 2, 1, 3) \
                            .reshape(-1, x.shape[1], attention_head_size)

                x.lw = transpose_w(x.lw)
                x.uw = transpose_w(x.uw)
                x.lb = transpose_b(x.lb)
                x.ub = transpose_b(x.ub)
                x.update_shape()

            transpose_for_scores(query)
            transpose_for_scores(key)

            # ignoring the attention mask
            attention_scores = query.dot_product(key, verbose=self.verbose).multiply(1. / math.sqrt(attention_head_size))

            del query
            del key
            attention_probs = attention_scores.softmax(verbose=self.verbose)

            transpose_for_scores(value)

            context = attention_probs.context(value)

            def transpose_back(x):
                def transpose_w(x):
                    return x.permute(1, 2, 0, 3).reshape(1, x.shape[1], x.shape[2], -1)

                def transpose_b(x):
                    return x.permute(1, 0, 2).reshape(1, x.shape[1], -1)

                x.lw = transpose_w(x.lw)
                x.uw = transpose_w(x.uw)
                x.lb = transpose_b(x.lb)
                x.ub = transpose_b(x.ub)
                x.update_shape()

            transpose_back(context)

            context = Layer(
                self.args, self.controller, bounds_input.length, bounds_input.dim,
                bounds=context
            )

            attention_scores.l, attention_scores.u = attention_scores.concretize()
            attention_probs.l, attention_probs.u = attention_probs.concretize()
            attention_scores.l = attention_scores.l.transpose(0, 1).reshape(bounds_input.length, -1)
            attention_scores.u = attention_scores.u.transpose(0, 1).reshape(bounds_input.length, -1)
            attention_probs.l = attention_probs.l.transpose(0, 1).reshape(bounds_input.length, -1)
            attention_probs.u = attention_probs.u.transpose(0, 1).reshape(bounds_input.length, -1)
        else:
            attention_scores = self.get_bounds(
                input_bounds=query,
                edge=EdgeDotProduct(self.args, self.controller, query, key, num_attention_heads),
                dim=num_attention_heads * query.length,
                at_least_one_next_layer_params_depend_on_bounds=False
            )

            attention_scores = self.get_bounds(
                attention_scores,
                EdgeLinear(self.args, self.controller, attention_scores, w=1. / math.sqrt(attention_head_size), b=0.),
                at_least_one_next_layer_params_depend_on_bounds=True)

            attention_probs = self._bound_softmax(attention_scores, num_attention_heads, bounds_needed_for_next_layer_params=True)

            dim_out = value.dim
            # For this transpose, the dB/dλ won't be used in the first layer, since the value doesn't depend on any lambdas
            # This explain why I had to disable the strict=True thing in the autograd.py file
            value = self.get_bounds(value, EdgeTranspose(
                self.args, self.controller, value, num_attention_heads
            ), length=attention_head_size, dim=num_attention_heads * value.length, at_least_one_next_layer_params_depend_on_bounds=True)

            context = self.get_bounds(
                attention_probs, EdgeDotProduct(self.args, self.controller, attention_probs, value, num_attention_heads),
                dim=dim_out,
                at_least_one_next_layer_params_depend_on_bounds=False  # Afterwards it's a Dense Layer
            )

        return attention_scores, attention_probs, context

    def _bound_softmax(self, bounds: Layer, num_attention_heads: int, bounds_needed_for_next_layer_params=True):
        softmax_data: ConvexCombinationData = self.convex_combination_elements.get_next_element()

        bounds_exp = self.get_bounds(bounds, EdgeExp(self.args, self.controller, bounds),
                                     at_least_one_next_layer_params_depend_on_bounds=True)
        bounds_exp_sum = self.get_bounds(bounds_exp, EdgeSum.build_sum_edge(self.args, self.controller, bounds_exp, num_attention_heads),
                                         at_least_one_next_layer_params_depend_on_bounds=True)
        divide_edge = EdgeDivideHyperplanes(
            self.args, self.controller, bounds_exp, bounds_exp_sum, softmax_data.lambdas_verification, softmax_data.hyperplanes_coeffs,
            lambdas_index=self.number_relevant_lambdas,
        )

        self.number_relevant_lambdas += 1
        
        softmax_bounds = self.get_bounds(bounds_exp, divide_edge,
                                         at_least_one_next_layer_params_depend_on_bounds=bounds_needed_for_next_layer_params)

        if softmax_data.hyperplanes_coeffs is None:
            softmax_data.hyperplanes_coeffs = divide_edge.coeffs

        return softmax_bounds

    def _bound_pooling(self, bounds: Layer, bert_pooler) -> Layer:
        # 1) They multiply the bounds by the pooler's matrix
        # 2) They apply a tanh activation on the result
        bounds = self.get_bounds(bounds, EdgeDense(self.args, self.controller, bounds, dense=bert_pooler.dense),
                                 at_least_one_next_layer_params_depend_on_bounds=True)

        bounds = self.get_bounds(bounds, EdgeTanh(self.args, self.controller, bounds),
                                 at_least_one_next_layer_params_depend_on_bounds=False  # It's a Dense (the classifier) afterwards
                                 )

        return bounds

    def _bound_classifier(self, bounds, classifier, label) -> Tuple[Layer, bool]:
        # 1) They compute linear layer that computes the how higher class 0 is over class 1
        # 2) They multiply the bounds by that linear layer's matrix
        # 3) They check if things are safe or not (e.g. if the lower bound of c0 - c1 > 0, then we're good)

        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        bounds = self.get_bounds(bounds, EdgeDense(self.args, self.controller, bounds, dense=classifier),
                                 dim=classifier.weight.shape[0],
                                 at_least_one_next_layer_params_depend_on_bounds=True,  # To ensure we have the jacobian
                                 )

        if label == 0:
            safe = bounds.l[0][0] > 0
        else:
            safe = bounds.u[0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return bounds, safe



