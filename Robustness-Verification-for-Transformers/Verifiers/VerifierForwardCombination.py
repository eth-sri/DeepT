# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import copy
import math

import torch

from Verifiers import Verifier
from Verifiers.Bounds import Bounds, SoftmaxOp, DotProductOp, MultiplyOp, DenseOp, ContextOp, TanhOp, AddOp, LayerNormOp, ActOp
from Verifiers.ConvexCombination import Container
from Verifiers.utils import check


# Changes:
# - For some layers, we keep track of the best concrete bounds and always use that.
# - In other words, when computing the lower and uppper bounds in a new iteration, we do:
#   new_bounds.lb = torch.max(old_bounds.lb, compute_lb_bounds())
#   new_bounds.ub = torch.min(new_bounds.ub, compute_ub_bounds())
# At which layers do we need to do this? Is it at each layer or only at the layer with the lambdas?
# There are 2 types of layers, those which adapt to the bounds of the previous layer (almost all layers) and those who don't (the
# layer obtained using the sampling / Gubori, which is done only once). Therefore, if we update the Gurobi layer to use the
# concrete bounds update rules to ensure they don't get worse, then there we never get worse than before; and can only get better.
# Since the other layers adapt to the bounds, then they will also stay as good or get better.
# Conclusion: we only need to adapt the Gurobi / sampling layers. This is quite nice because it makes it easier to implemenent
# since they are already a special class (currently only SoftmaxOp) which can keep track of state.


# What are the linearizations that we need to improve?
# The ReLU activations in the intermediate layer  NOT DONE
# The softmax activation in the self-attention. DONE

# can only accept one example in each batch
class VerifierForwardConvexCombination(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierForwardConvexCombination, self).__init__(args, target, logger)
        self.ibp = args.method == "ibp"
        self.previous_eps = None
        self.reset()

    def reset(self):
        self.softmaxes = Container()
        self.layers = Container()
        self.is_initialized = False

    def get_lambdas(self):
        return [softmax.lambdas for softmax in self.softmaxes.get_all_elements()]

    def start_verification_new_input(self):
        self.reset()

    def get_next_layer(self, default_cls, *args):
        if self.is_initialized:
            return self.layers.get_next_element()
        else:
            layer = default_cls(*args)
            self.layers.add_element(layer)
            return layer

    def get_and_apply_layer(self, default_cls, bounds: Bounds, *other_args):
        layer = self.get_next_layer(default_cls)
        return layer.forward(bounds, *other_args)

    def verify_safety(self, example, embeddings, index, eps):
        errorType = OSError if self.debug else AssertionError
        label = example["label"]

        # Ensure that we don't use min/max bounds from a smaller eps
        # We are fine if the previous eps was bigger (we are not fine if it was smaller)
        if self.is_initialized:  # and not (self.previous_eps > eps):
            # print("Resetting best_lb / best_ub because previous_eps (%f) < eps (%f)" % (self.previous_eps, eps))
            print("Resetting best_lb / best_ub for softmaxes")
            for softmax in self.softmaxes.get_all_elements():
                softmax.reset_best_lb_and_best_ub()

            print("Resetting best_lb / best_ub for the other layers")
            for layer in self.layers.get_all_elements():
                layer.reset_best_lb_and_best_ub()

        # We don't reset the lambdas because we saw that it led to either the same results (if we reset best_lb/best_ub when eps increases) 
        # or worst results (if we reset best_lb/best_ub every time eps changes)
        self.previous_eps = eps

        try:
            bounds, concretized_bounds, safety = self.run_verifier(embeddings, eps, index, label)
            if safety:
                return True

            print()
            lr_fn = lambda e: 100 * 0.99 ** e
            optimizer = torch.optim.Adam(self.get_lambdas())
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
            max_iter = self.args.max_optim_iters
            for epoch in range(max_iter):
                # Compute everything using the lambdas!
                # The thing that I need to add are:
                # 1) Put all the code to do a full pass in a function
                # 2) Find a way to re-use the lambdas during that full pass
                # 3) Create the different planes for the ReLU (intermediate) and Softmax (self-attention)
                #    using the Gurobi technique described by Mislav
                if epoch > 0:  # In the first round, we can use the results we got before this loop
                    bounds, concretized_bounds, safety = self.run_verifier(embeddings, eps, index, label)

                if safety:
                    print("\nDid %d round of optimization: success" % epoch)
                    return True

                l, u = concretized_bounds
                if label == 0:
                    loss = -l[0][0][0]   # We want to increase the value of this, so the loss which we minimize is the negation
                else:
                    loss = u[0][0][0]  # We want to increase the value of this, so the loss which we minimize is the negation
                print("\rEpoch %d: Loss = %s    (for eps: %s)" % (epoch, loss, eps), end="")

                # old_lambdas = [lambdas.clone() for lambdas in self.get_lambdas()]

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

                # new_lambdas = self.get_lambdas()
                # for i, (old_lambda, new_lambda) in enumerate(zip(old_lambdas, new_lambdas)):
                #    print("The maximum change of lambdas at layer %d is: %f" % (i, (old_lambda - new_lambda).abs().max()))

            print("\nAfter %d round of optimization: no success" % (max_iter - 1))

            return False

        except errorType as err:  # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            return False

    def run_verifier(self, embeddings, eps, index, label):
        bounds = self._bound_input(embeddings, index=index, eps=eps)  # hard-coded yet
        check("embedding", bounds=bounds, std=self.std["embedding_output"], verbose=self.verbose)

        if self.verbose:
            bounds.print("embedding")

        for i, layer in enumerate(self.encoding_layers):
            # print("Layer %d" % i)
            attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer, i)
            check("layer %d attention_scores" % i, bounds=attention_scores, std=self.std["attention_scores"][i][0], verbose=self.verbose)
            check("layer %d attention_probs" % i, bounds=attention_probs, std=self.std["attention_probs"][i][0], verbose=self.verbose)
            check("layer %d" % i, bounds=bounds, std=self.std["encoded_layers"][i], verbose=self.verbose)

        bounds = self._bound_pooling(bounds, self.pooler)
        check("pooled output", bounds=bounds, std=self.std["pooled_output"], verbose=self.verbose)

        bounds, concretized_bounds, safety = self._bound_classifier(bounds, self.classifier, label)

        # TODO: create util to visualise range of tensor
        # The brightness would allow us to see the distribution very easily
        # Example output:    ░░░░░░░▒▒▒▓▒▒▒▓▓▓▓▓▒▒░░░░▒▒▓▒▒░░░░░░░
        #                   Min                                 Max
        #                   93                                  134

        if not self.is_initialized:
            self.is_initialized = True
            self.softmaxes.mark_initialization_over()
            self.layers.mark_initialization_over()

        self.softmaxes.reset_cursor()
        self.layers.reset_cursor()

        return bounds, concretized_bounds, safety

    def _bound_input(self, embeddings, index, eps):
        length, dim = embeddings.shape[1], embeddings.shape[2]

        # 1) If IBP, they decrease/increase the lower/upper bounds of the perturbed words by eps
        # 2) Otherwise, they say that each element of the perturbed words may be perturbed by eps
        #    They say this using error terms which are stored in a weight matrix

        # Weights: for every element of the sequence (dim), wheck the link between the perturbed embedding (dim 1)
        # and the embedding of that element (dim 2)
        w = torch.zeros((length, dim * self.perturbed_words, dim)).to(self.device)
        # Bias = Initial value of the embeddings for the whole sequence
        b = embeddings[0]
        lb, ub = b, b.clone()

        if self.perturbed_words == 1:
            if self.ibp:
                lb[index], ub[index] = lb[index] - eps, ub[index] + eps
            else:
                w[index] = torch.eye(dim).to(self.device)  # The perturbed embedding depends on itself
        else:
            if self.ibp:
                for i in range(self.perturbed_words):
                    lb[index[i]], ub[index[i]] = lb[index[i]] - eps, ub[index[i]] + eps
            else:
                for i in range(self.perturbed_words):
                    w[index[i], (dim * i):(dim * (i + 1)), :] = torch.eye(dim).to(self.device)

        lw = w.unsqueeze(0)
        uw = lw.clone()
        lb = lb.unsqueeze(0)
        ub = ub.unsqueeze(0)

        bounds = Bounds(self.args, self.p, eps, lw=lw, lb=lb, uw=uw, ub=ub)
        bounds = self.get_and_apply_layer(LayerNormOp, bounds, self.embeddings.LayerNorm, self.layer_norm)

        return bounds

    def _bound_layer(self, bounds_input, layer, layer_num):
        # Here's what they do:
        # 1) Compute bounds of self-attention
        # 2) Multiply by the attention's output matrix
        # 3) Add the original x (residual connection) and then normalize
        # 4) Multiply by an intermediate matrix
        # 5) Apply an activation (ReLU)
        # 6) Multiply by an output matrix
        # 7) Add the attention values computed in 2
        # 8) Normalize
        # 9) Return the output of 8, the attention scores and the attention probs
        attention_scores, attention_probs, attention = self._bound_attention(bounds_input, layer.attention, layer_num)

        attention = self.get_and_apply_layer(DenseOp, attention, layer.attention.output.dense)
        attention = self.get_and_apply_layer(AddOp, attention, bounds_input)
        attention = self.get_and_apply_layer(LayerNormOp, attention, layer.attention.output.LayerNorm, self.layer_norm)

        if self.verbose:
            attention.print("after attention layernorm")
            attention.dense(layer.intermediate.dense).print("intermediate pre-activation")
            print("dense norm", torch.norm(layer.intermediate.dense.weight, p=self.p))

        intermediate = self.get_and_apply_layer(DenseOp, attention, layer.intermediate.dense)
        # TODO: linearize ReLu
        intermediate = self.get_and_apply_layer(ActOp, intermediate)  # intermediate.act(self.hidden_act)

        if self.verbose:
            intermediate.print("intermediate")

        dense = self.get_and_apply_layer(DenseOp, intermediate, layer.output.dense)
        dense = self.get_and_apply_layer(AddOp, dense, attention)

        if self.verbose:
            print("dense norm", torch.norm(layer.output.dense.weight, p=self.p))
            dense.print("output pre layer norm")

        output = self.get_and_apply_layer(LayerNormOp, dense, layer.output.LayerNorm, self.layer_norm)

        if self.verbose:
            output.print("output")

        return attention_scores, attention_probs, output

    def _bound_attention(self, bounds_input, attention, layer_num):
        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        # 1) Bound the queries, keys and values for each attention head
        # 2) Transpose and reshape the queries and the keys (TODO: don't know why this is needed)
        # 3) Do the dot product between the keys and the queries, and ther normalize by 1 / sqrt(attention head size)
        #    obtaining the attention scores
        # 4) Compute the softmax of the attention scores, obtaining the attention probs
        # 5) They transpose the values (so that the multiplication works fine)
        # 6) They do a dot-product like thing between the attention scores and the values
        # 7) They transpose the result of (6) back
        # 8) They return the attention scores, attention probs and the result of (7)

        query = self.get_and_apply_layer(DenseOp, bounds_input, attention.self.query)
        key = self.get_and_apply_layer(DenseOp, bounds_input, attention.self.key)
        value = self.get_and_apply_layer(DenseOp, bounds_input, attention.self.value)

        def transpose_for_scores(x):
            def transpose_w(x):
                # Original shape: batch_size x length x embedding dim x embedding_dim_out
                # Shape 1: batch_size x length x dim_in x num_attention_heads (4) x attention_head_size (64)
                # Shape 2: batch_size x num_attention_heads (4) x length x dim_in x attention_head_size (64)
                # Output shape 3: (batch_size * num_attention_heads) x length x dim_in x attention_head_size (64)
                return x.reshape(x.shape[0], x.shape[1], x.shape[2], num_attention_heads, attention_head_size) \
                    .permute(0, 3, 1, 2, 4) \
                    .reshape(-1, x.shape[1], x.shape[2], attention_head_size)

            def transpose_b(x):
                # Original shape: batch_size x length x embedding dim
                # Shape 1: batch_size x length x num_attention_heads (4) x attention_head_size (64)
                # Shape 2: batch_size x num_attention_heads (4) x length x attention_head_size (64)
                # Output shape 3: (batch_size * num_attention_heads) x length x attention_head_size (64)
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

        attention_scores = self.get_and_apply_layer(DotProductOp, query, key)
        attention_scores = self.get_and_apply_layer(MultiplyOp, attention_scores, 1. / math.sqrt(attention_head_size))

        if self.verbose:
            attention_scores.print("attention score")

        if not self.is_initialized:
            softmax_op = SoftmaxOp(attention_scores, self.args)
            self.softmaxes.add_element(softmax_op)
        else:
            softmax_op = self.softmaxes.get_next_element()
            # if layer_num > 0:
            #     softmax_op.needs_to_adjust_hyperplanes()

        attention_probs = softmax_op.forward(attention_scores)

        if self.verbose:
            attention_probs.print("attention probs")

        transpose_for_scores(value)

        context = self.get_and_apply_layer(ContextOp, attention_probs, value)

        if self.verbose:
            value.print("value")
            context.print("context")

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
        
        return attention_scores, attention_probs, context

    def _bound_pooling(self, bounds, pooler):
        # 1) They update the bounds and keep only some elements
        # 2) They multiply the bounds by the pooler's matrix
        # 3) They apply a tanh activation on the result

        bounds = Bounds(
            self.args, bounds.p, bounds.eps,
            lw=bounds.lw[:, :1, :, :], lb=bounds.lb[:, :1, :],
            uw=bounds.uw[:, :1, :, :], ub=bounds.ub[:, :1, :]
        )
        if self.verbose:
            bounds.print("pooling before dense")

        bounds = self.get_and_apply_layer(DenseOp, bounds, pooler.dense)

        if self.verbose:
            bounds.print("pooling pre-activation")

        bounds = self.get_and_apply_layer(TanhOp, bounds)

        if self.verbose:
            bounds.print("pooling after activation")
        return bounds

    def _bound_classifier(self, bounds, classifier, label):
        # 1) They compute linear layer that computes the how higher class 0 is over class 1
        # 2) They multiply the bounds by that linear layer's matrix
        # 3) They concretize the bounds (e.g. they compute the actual values, instead of having error terms)
        # 4) They check if things are safe or not (e.g. if the lower bound of c0 - c1 > 0, then we're good)

        # We have to carefully update this classifier, because otherwise we get
        # error during the optimization "RuntimeError: leaf variable has been moved into the graph interior"
        # To avoid this, I can't just directly update the weights in place, I have to
        classifier = copy.deepcopy(classifier)
        with torch.no_grad():
            classifier.weight[0, :].sub_(classifier.weight[1, :])
            classifier.bias[0].sub_(classifier.bias[1])

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.lw, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(classifier).lw, dim=-2)))

        bounds = self.get_and_apply_layer(DenseOp, bounds, classifier)

        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()

        if self.verbose:
            print(l[0][0][0])
            print(u[0][0][0])

        if label == 0:
            safe = l[0][0][0] > 0
        else:
            safe = u[0][0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return bounds, (l, u), safe
