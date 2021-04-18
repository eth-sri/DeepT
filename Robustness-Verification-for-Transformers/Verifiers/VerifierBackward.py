# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import copy
import math
import time

from Verifiers import Verifier, Bounds
from Verifiers.Edge import *
from Verifiers.memory import get_gpu_memory_usage
from Verifiers.utils import check

epsilon = 1e-12


# For synonym verification, I can set perturbed words = #seq length
# And then I only need to adapt the concretize for both Bounds (done) and Layer

# can only accept one example in each batch
class VerifierBackward(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierBackward, self).__init__(args, target, logger)
        self.controller = None
        self.use_forward = ("baf" in args.method)
        self.empty_cache = args.empty_cache

    def verify_safety(self, example, embeddings, index, eps, **kwargs):
        errorType = OSError if self.debug else AssertionError


        # cannot accept a batch
        embeddings = embeddings[0]

        try:
            with torch.no_grad():
                start = time.time()
                bounds = self._bound_input(embeddings, index=index, eps=eps)  # hard-coded yet
                # check("embedding", l=bounds.l, u=bounds.u, std=self.std["embedding_output"][0], verbose=self.debug)

                if self.verbose:
                    bounds.print("embedding")


                for i, layer in enumerate(self.encoding_layers):
                    if self.args.method == 'backward' and self.args.cpu:
                        print("Processing encoding layer %d" % i)
                    start_layer = time.time()
                    #print("Memory used before doing layer %i: %d MB" % (i, get_gpu_memory_usage()))
                    attention_scores, attention_probs, bounds = self._bound_layer(bounds, layer)
                    #print("Memory used after doing layer %i: %d MB" % (i, get_gpu_memory_usage()))
                    end_layer = time.time()
                    if self.args.method == 'backward' and self.args.cpu:
                        print("Finishing processing encoding layer %d, it took %f s" % (i, end_layer - start_layer))

                    # std = self.std["attention_scores"][i][0]
                    # std = std.transpose(0, 1).reshape(1, std.shape[1], -1)
                    # check("layer {} attention_scores".format(i),
                    #       l=attention_scores.l, u=attention_scores.u, std=std[0], verbose=self.debug)
                    # std = self.std["attention_probs"][i][0]
                    # std = std.transpose(0, 1).reshape(1, std.shape[1], -1)
                    # check("layer {} attention_probs".format(i),
                    #       l=attention_probs.l, u=attention_probs.u, std=std[0], verbose=self.debug)
                    # check("layer {}".format(i), l=bounds.l, u=bounds.u, std=self.std["encoded_layers"][i][0],
                    #       verbose=self.debug)

                bounds = self._bound_pooling(bounds, self.pooler)
                # check("pooled output", l=bounds.l[:1], u=bounds.u[:1], std=self.std["pooled_output"][0],
                #       verbose=self.debug)

                safety = self._bound_classifier(bounds, self.classifier, example["label"])
                
                end = time.time()
                # if self.args.method == 'backward':
                #     print("Iteration took: %s seconds" % (end - start))

                for layer in self.controller.layers:
                    del layer.lw
                    del layer.uw
                    del layer.final_lw
                    del layer.final_uw
                    del layer.final_lb
                    del layer.final_ub
                if self.empty_cache:
                    torch.cuda.empty_cache()

                return safety
        except errorType as err:  # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            print("Warning: failed assertion", eps)
            print(err)
            return False

    def _bound_input(self, embeddings, index, eps) -> Layer:
        length, dim = embeddings.shape[0], embeddings.shape[1]

        self.controller = BacksubstitutionComputer(self.args, eps)

        layer = Layer(self.args, self.controller, length, dim)
        layer.add_edge(EdgeInput(self.args, self.controller, embeddings, index))
        layer.compute()

        if self.args.with_lirpa_transformer:
            layer = layer.next(EdgeDense(self.args, self.controller, layer,
                                         w=self.target.model_from_embeddings.linear_in.weight,
                                         b=self.target.model_from_embeddings.linear_in.bias))
            layer = self._bound_layer_normalization(layer, self.target.model_from_embeddings.LayerNorm)
        else:
            layer = self._bound_layer_normalization(layer, self.embeddings.LayerNorm)

        return layer

    def _bound_layer_normalization(self, layer, normalizer, debug=False):
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

        minus_mu = layer.next(EdgeDense(
            self.args, self.controller, layer, w=eye - w_avg, b=0.))  # Computes x - avg(x's)

        if self.layer_norm == "standard":
            # Compute the variance:
            # 1: (x - avg(x's))²
            # 2: sum 1/n over of term in (1)
            # 3: sqrt of (2)
            # 4: divide by (3) (e.g. divide by the standard deviation)
            minus_mu_sqr = minus_mu.next(EdgeSqr(self.args, self.controller, minus_mu))
            variance = minus_mu_sqr.next(EdgeDense(
                self.args, self.controller, minus_mu_sqr,
                w=w_avg, b=epsilon
            ))

            if self.verbose:
                variance.print("variance")

            std = variance.next(EdgeSqrt(self.args, self.controller, variance))

            normalized = minus_mu.next(EdgeDivide(self.args, self.controller, minus_mu, std))
        else:
            assert (self.layer_norm == "no_var"), "if layer_norm is not 'standard', then it has to be 'no_var'"
            normalized = minus_mu

        # TODO: don't know why they're doing this; it seems like they're normalizing twice.
        # It look like normalizer is a Bert Normalizer layer, so they
        normalized = normalized.next(
            EdgeLinear(self.args, self.controller, normalized, normalizer.weight, normalizer.bias))

        return normalized

    def _bound_layer(self, bounds_input: Layer, layer):
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

        attention_scores, attention_probs, attention = self._bound_attention(bounds_input, layer.attention)
        attention = attention.next(EdgeDense(self.args, self.controller, attention, dense=layer.attention.output.dense))

        attention_residual = Layer(self.args, self.controller, attention.length, attention.dim)
        attention_residual.add_edge(EdgeDirect(self.args, self.controller, attention))
        attention_residual.add_edge(EdgeDirect(self.args, self.controller, bounds_input))
        attention_residual.compute()

        attention = self._bound_layer_normalization(attention_residual, layer.attention.output.LayerNorm, debug=True)

        intermediate = attention.next(EdgeDense(
            self.args, self.controller, attention, dense=layer.intermediate.dense
        ), dim=layer.intermediate.dense.weight.shape[0])
        assert (self.hidden_act == "relu"), "Only ReLU activation is supported"
        intermediate = intermediate.next(EdgeRelu(self.args, self.controller, intermediate))

        dense = intermediate.next(EdgeDense(
            self.args, self.controller, intermediate, dense=layer.output.dense
        ), dim=layer.output.dense.weight.shape[0])

        dense_residual = Layer(self.args, self.controller, dense.length, dense.dim)
        dense_residual.add_edge(EdgeDirect(self.args, self.controller, dense))
        dense_residual.add_edge(EdgeDirect(self.args, self.controller, attention))
        dense_residual.compute()

        output = self._bound_layer_normalization(dense_residual, layer.output.LayerNorm)

        return attention_scores, attention_probs, output

    def _bound_attention(self, bounds_input: Layer, attention):

        # This in practice is not the case, but the explanation for the forwards case is already present in the
        # VerifierForward code, so I don't repeat it here). Here are the steps:
        # 1) Bound the queries, keys and values for each attention head
        # 2) Do the dot product between the keys and the queries, and ther normalize by 1 / sqrt(attention head size)
        #    obtaining the attention scores
        # 3) Compute the softmax of the attention scores, obtaining the attention probs
        # 4) They transpose the values (TODO: don't know why this is needed)
        # 5) They do a dot-product like thing between the attention scores and the values
        # 6) They return the attention scores, attention probs and the result of (7)

        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        query = bounds_input.next(EdgeDense(self.args, self.controller, bounds_input, dense=attention.self.query))
        key = bounds_input.next(EdgeDense(self.args, self.controller, bounds_input, dense=attention.self.key))
        value = bounds_input.next(EdgeDense(self.args, self.controller, bounds_input, dense=attention.self.value))

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
                    return x \
                        .reshape(
                        x.shape[0], x.shape[1], x.shape[2],
                        num_attention_heads, attention_head_size) \
                        .permute(0, 3, 1, 2, 4) \
                        .reshape(-1, x.shape[1], x.shape[2], attention_head_size)

                def transpose_b(x):
                    return x \
                        .reshape(
                        x.shape[0], x.shape[1], num_attention_heads, attention_head_size) \
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
            attention_scores = query.dot_product(key, verbose=self.verbose) \
                .multiply(1. / math.sqrt(attention_head_size))

            del query  # self.controller.delete_tensor(query)
            del key  # self.controller.delete_tensor(key)
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
            attention_scores = query.next(EdgeDotProduct(
                self.args, self.controller, query, key, num_attention_heads),
                dim=num_attention_heads * query.length)

            attention_scores = attention_scores.next(EdgeLinear(
                self.args, self.controller, attention_scores, w=1. / math.sqrt(attention_head_size), b=0.))

            if self.args.use_new_softmax:
                attention_probs = attention_scores.next(EdgeSoftmaxNew(
                    self.args, self.controller, attention_scores, num_attention_heads
                ))
            else:
                attention_probs = attention_scores.next(EdgeSoftmax(
                    self.args, self.controller, attention_scores, num_attention_heads
                ))

            dim_out = value.dim
            value = value.next(EdgeTranspose(
                self.args, self.controller, value, num_attention_heads
            ), length=attention_head_size, dim=num_attention_heads * value.length)
            context = attention_probs.next(EdgeDotProduct(
                self.args, self.controller, attention_probs, value, num_attention_heads
            ), dim=dim_out)

        return attention_scores, attention_probs, context

    def _bound_pooling(self, bounds, pooler):
        # 1) They multiply the bounds by the pooler's matrix
        # 2) They apply a tanh activation on the result
        bounds = bounds.next(EdgeDense(
            self.args, self.controller, bounds, dense=pooler.dense
        ))

        bounds = bounds.next(EdgeTanh(
            self.args, self.controller, bounds
        ))

        return bounds

    def _bound_classifier(self, bounds, classifier, label):
        # 1) They compute linear layer that computes the how higher class 0 is over class 1
        # 2) They multiply the bounds by that linear layer's matrix
        # 3) They check if things are safe or not (e.g. if the lower bound of c0 - c1 > 0, then we're good)

        classifier = copy.deepcopy(classifier)
        classifier.weight[0, :] -= classifier.weight[1, :]
        classifier.bias[0] -= classifier.bias[1]

        bounds = bounds.next(EdgeDense(
            self.args, self.controller, bounds, dense=classifier
        ), dim=classifier.weight.shape[0])

        if label == 0:
            safe = bounds.l[0][0] > 0
        else:
            safe = bounds.u[0][0] < 0

        if self.verbose:
            print("Safe" if safe else "Unsafe")

        return safe
