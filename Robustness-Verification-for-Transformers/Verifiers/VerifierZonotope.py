import copy
import time
import math
from typing import Tuple, List, Optional

import torch

from Verifiers import Verifier
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args, cleanup_memory
from Verifiers.utils import check_zonotope


# can only accept one example in each batch
class VerifierZonotope(Verifier):
    def __init__(self, args, target, logger):
        super(VerifierZonotope, self).__init__(args, target, logger)
        self.ibp = args.method == "ibp"
        self.should_collect_zonotopes = False

        self.showed_warning = False

    def collect_zonotopes_bounds(self, examples, sentence_num, word_num, eps):
        example = examples[sentence_num]

        embeddings, tokens = self.target.get_embeddings([example])
        embeddings = embeddings if self.args.cpu else embeddings.cuda()
        tokens = tokens[0]

        self.start_verification_new_input()
        assert not (tokens[word_num][0] == "#" or tokens[word_num + 1][0] == "#"), "Chosen word is OOV"

        self.should_collect_zonotopes = True
        self.zonotopes_bounds = []
        safe, zonotopes_bounds = self.verify_safety(example, embeddings, word_num, eps)
        self.should_collect_zonotopes = False
        return zonotopes_bounds, example

    def check_samples(self, example, zonotopes_bounds, sentence_num, word_num, eps, num_samples=1000):
        embeddings, tokens = self.target.get_embeddings([example])
        embeddings = embeddings if self.args.cpu else embeddings.cuda()

        tokens = tokens[0]
        assert not (tokens[word_num][0] == "#" or tokens[word_num + 1][0] == "#"), "Chosen word is OOV"
        print()
        violations_per_layers = torch.zeros(len(zonotopes_bounds))
        for i in range(num_samples):
            # Perturb the embeddings
            perturbation = -eps + (2 * eps) * torch.rand(embeddings[0, word_num].shape, device=embeddings.device)
            perturbed_embeddings = embeddings.clone()
            perturbed_embeddings[0, word_num] += perturbation

            new_violations_per_layer = self.verify_samples_fits_in_zonotope(example, perturbed_embeddings, zonotopes_bounds)
            violations_per_layers += new_violations_per_layer
            print("\r%s" % violations_per_layers, end="")
        print()

        return violations_per_layers

    def verify_samples_fits_in_zonotope(self, example, perturbed_embeddings: torch.Tensor, zonotopes_bounds: List[Zonotope]):
        eps = 1.0e-6

        violations_per_layer = torch.zeros(len(zonotopes_bounds))
        ret, values = self.target.step([example], embeddings=perturbed_embeddings, capture_values=True)
        assert len(values) == len(zonotopes_bounds), "Mismatch: there are %d values but %d zonotopes" % (len(values), len(zonotopes_bounds))
        for i, ((l, u, description), value) in enumerate(zip(zonotopes_bounds, values)):
            value, l, u = value.squeeze(), l.squeeze(), u.squeeze()

            if (value > u + eps).sum() > 0:
                violations_per_layer[i] += 1
            elif (value < l - eps).sum() > 0:
                violations_per_layer[i] += 1

        return violations_per_layer

    def verify_safety(self, example, embeddings, index, eps, initial_zonotope_weights: torch.Tensor = None):
        bounds_difference_scores = self.get_bounds_difference_in_scores(
            embeddings, index, eps, initial_zonotope_weights=initial_zonotope_weights
        )
        if bounds_difference_scores is None:
            model_is_robust = False  # There was an exception or some pre-conditions were not met
        else:
            l, u = bounds_difference_scores
            model_is_robust = self.get_safety(label=example["label"], lower_bound=l, upper_bound=u)

        if self.should_collect_zonotopes:
            return model_is_robust, self.zonotopes_bounds[:]
        else:
            return model_is_robust

    def get_bounds_difference_in_scores(
        self, embeddings: torch.Tensor, index: int, eps: float, gradient_enabled=False, initial_zonotope_weights: torch.Tensor = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.args.error_reduction_method == "None" and not self.showed_warning:
            START_WARNING = '\033[93m'
            END_WARNING = '\033[0m'
            print(START_WARNING + "Warning: No error reduction method for Zonotope verifier!" + END_WARNING)
            self.showed_warning = True

        cleanup_memory()
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.set_grad_enabled(gradient_enabled):

                bounds = self._bound_input(embeddings, index=index, eps=eps, initial_zonotope_weights=initial_zonotope_weights)  # hard-coded yet

                if self.args.check_zonotopes:
                    check_zonotope("embedding", zonotope=bounds, actual_values=self.std["embedding_output"], verbose=self.verbose)

                if self.verbose:
                    bounds.print("embedding")

                if self.args.log_error_terms_and_time: print("Input to attention layer 0 has %d error terms" % bounds.num_error_terms)

                for i, layer in enumerate(self.encoding_layers):
                    if self.args.log_error_terms_and_time:
                        print()
                        print("Layer %d" % i)

                    start = time.time()
                    attention_scores, attention_probs, self_attention_output, bounds = self._bound_layer(bounds, layer, layer_num=i)
                    end = time.time()

                    if self.args.log_error_terms_and_time:
                        print("Output of attention layer %d has %d error terms" % (i, bounds.num_error_terms))
                        print("Time to do attention layer %d is %.3f seconds" % (i, end - start))

                    if self.args.check_zonotopes:
                        check_zonotope("layer %d attention_scores" % i, zonotope=attention_scores, actual_values=self.std["attention_scores"][i][0], verbose=self.verbose)
                        check_zonotope("layer %d attention_probs" % i, zonotope=attention_probs, actual_values=self.std["attention_probs"][i][0], verbose=self.verbose)
                        check_zonotope("layer %d self_attention_output" % i, zonotope=self_attention_output, actual_values=self.std["self_output"][i][0], verbose=self.verbose)
                        check_zonotope("layer %d" % i, zonotope=bounds, actual_values=self.std["encoded_layers"][i], verbose=self.verbose)

                    if not self.args.keep_intermediate_zonotopes:
                        del attention_scores, attention_probs, self_attention_output
                        cleanup_memory()

                bounds = self._bound_pooling(bounds, self.pooler)
                if self.args.check_zonotopes:
                    check_zonotope("pooled output", zonotope=bounds, actual_values=self.std["pooled_output"], verbose=self.verbose)

                l, u = self._bound_classifier(bounds, self.classifier)
                return l, u
        except errorType as err:  # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            # raise yerr
            return None

    def store_zonotope_bounds(self, zonotope: Zonotope, zonotope_description: str):
        if self.should_collect_zonotopes:
            l, u = zonotope.concretize()
            self.zonotopes_bounds.append((l, u, zonotope_description))

    def _bound_input(self, embeddings: torch.Tensor, index: int, eps: float, initial_zonotope_weights: torch.Tensor = None) -> Zonotope:
        if initial_zonotope_weights is None:
            assert embeddings[0].shape[1] == self.args.num_input_error_terms, "Invalid num error terms"
            bounds = Zonotope(self.args, p=self.p, eps=eps, perturbed_word_index=index, value=embeddings[0])
        else:
            # center, errors = initial_zonotope_weights[0], initial_zonotope_weights[1:]
            # error_values = errors[errors != 0]

            # print(f"Number of words: {center.size(0)}")
            # print(f"Center:  mean {center.mean():f}  ")
            # if error_values.nelement() > 0:
            #     print(f"Errors:  {error_values.nelement()} errors have a non-zero value, sum error: {error_values.abs().sum():f}, max error: {error_values.abs().max():f}, avg error: {error_values.abs().mean():f}")
            # else:
            #     print("No error terms")
            bounds = Zonotope(self.args, p=self.p, eps=-1, perturbed_word_index=-1, zonotope_w=initial_zonotope_weights)

        self.store_zonotope_bounds(bounds, "Input Bounds")
        if self.args.with_lirpa_transformer:
            bounds = bounds.dense(self.target.model_from_embeddings.linear_in)
            bounds = bounds.layer_norm(self.target.model_from_embeddings.LayerNorm, self.layer_norm)
        else:
            bounds = bounds.layer_norm(self.embeddings.LayerNorm, self.layer_norm)
        self.store_zonotope_bounds(bounds, "Input Bounds after Layer Norm")

        return bounds

    def is_in_fast_layer(self, layer_num: int):
        assert self.args.variant2plus1 or self.args.variant1plus2, "Variant12 or Variant21 must be active"
        assert self.args.num_fast_dot_product_layers_due_to_switch != -1, "Must select switching layer"

        if layer_num == 0:
            return False

        # Note: num_fast_dot_product_layers_due_to_switch is 1-indexed but layer_num is 0-indexes
        if self.args.variant2plus1:
            # Example: last 5 layers out of 12 are fast (
            first_fast_layer = self.args.num_layers - self.args.num_fast_dot_product_layers_due_to_switch + 1
            return layer_num + 1 >= first_fast_layer
        else:
            # Example: first 4 layers are fast (layer_num + 1 <= 4)
            return layer_num + 1 <= self.args.num_fast_dot_product_layers_due_to_switch

    def _bound_layer(self, bounds_input: Zonotope, layer, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        # start_time = time.time()

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

        if self.args.log_error_terms_and_time:
            print("Bound_input before PCA reduction has %d error terms" % bounds_input.num_error_terms)

        # We recenter the zonotope and eliminate the error_term_ranges (the new zonotope is equivalent).
        # If we didn't do it, the Box or PCA error term reduction functions would have to take the errror ranges
        # into account and that would make things more complicated. By simply re-writing the zonotope in an equaivalent
        # fashion such that the error_ranges are eliminated, life is simpler and we don't have to worry about the error ranges anymore
        if bounds_input.error_term_range_low is not None:
            bounds_input = bounds_input.recenter_zonotope_and_eliminate_error_term_ranges()

        if self.args.error_reduction_method == 'box':
            if self.args.num_fast_dot_product_layers_due_to_switch != -1 and self.is_in_fast_layer(layer_num):
                bounds_input_reduced_box = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms_fast_layers)
            else:
                bounds_input_reduced_box = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms)
            bounds_input = bounds_input_reduced_box
            # print(f"Layer {layer_num} - after box reduction there are {bounds_input.num_error_terms} noise symbols")

        if self.args.log_error_terms_and_time:
            print("Bound_input after PCA reduction has %d error terms" % bounds_input.num_error_terms)

        # main self-attention
        attention_scores, attention_probs, attention = self._bound_attention(bounds_input, layer.attention, layer_num=layer_num)

        context = attention

        attention = attention.dense(layer.attention.output.dense)
        self.store_zonotope_bounds(attention, "Attention after Dense")

        bounds_input = bounds_input.expand_error_terms_to_match_zonotope(attention)

        attention = attention.add(bounds_input)
        self.store_zonotope_bounds(attention, "Attention after Dense + Input")

        attention = attention.layer_norm(layer.attention.output.LayerNorm, self.layer_norm)
        self.store_zonotope_bounds(attention, "Attention after Dense + Input -> Layer Norm")

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        if self.verbose:
            attention.print("after attention layernorm")
            attention.dense(layer.intermediate.dense).print("intermediate pre-activation")
            print("dense norm", torch.norm(layer.intermediate.dense.weight, p=self.p))
            # start_time = time.time()

        intermediate = attention.dense(layer.intermediate.dense).relu()
        self.store_zonotope_bounds(intermediate, "Intermediate -> ReLU")

        if self.verbose:
            intermediate.print("intermediate")

        attention = attention.expand_error_terms_to_match_zonotope(intermediate)
        dense = intermediate.dense(layer.output.dense).add(attention)
        self.store_zonotope_bounds(dense, "Dense")

        if not self.args.keep_intermediate_zonotopes:
            del intermediate
            del attention

        if self.verbose:
            print("dense norm", torch.norm(layer.output.dense.weight, p=self.p))
            dense.print("output pre layer norm")

        output = dense.layer_norm(layer.output.LayerNorm, self.layer_norm)
        self.store_zonotope_bounds(output, "Output")

        if self.verbose:
            output.print("output")

        return attention_scores, attention_probs, context, output

    def do_dot_product(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int):
        if self.args.num_fast_dot_product_layers_due_to_switch == -1:
            return left_z.dot_product(right_z, verbose=self.verbose)

        if self.is_in_fast_layer(layer_num=current_layer_num):
            return left_z.dot_product_fast(right_z, verbose=self.verbose)
        else:
            return left_z.dot_product_precise(right_z, verbose=self.verbose)

    def do_context(self, left_z: Zonotope, right_z: Zonotope, current_layer_num: int):
        return self.do_dot_product(left_z, right_z.t(), current_layer_num)

    def _bound_attention(self, bounds_input: Zonotope, attention, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope]:
        num_attention_heads = attention.self.num_attention_heads
        attention_head_size = attention.self.attention_head_size

        # 1) Bound the queries, keys and values for each attention head
        # 2) Transpose and reshape the queries and the keys
        # 3) Do the dot product between the keys and the queries, and ther normalize by 1 / sqrt(attention head size)
        #    obtaining the attention scores
        # 4) Compute the softmax of the attention scores, obtaining the attention probs
        # 5) They transpose the values
        # 6) They do a dot-product like thing between the attention scores and the values
        # 7) They transpose the result of (6) back
        # 8) They return the attention scores, attention probs and the result of (7)

        # No new error terms created here, this is only a linear combination :)
        query = bounds_input.dense(attention.self.query)
        key = bounds_input.dense(attention.self.key)

        query = query.add_attention_heads_dim(num_attention_heads)  # (A, 1 + #error, length, E)
        key = key.add_attention_heads_dim(num_attention_heads)  # (A, 1 + #error, length, E)

        self.store_zonotope_bounds(query, "Queries")
        self.store_zonotope_bounds(key, "Keys")

        attention_scores = self.do_dot_product(query, key, layer_num)
        attention_scores = attention_scores.multiply(1. / math.sqrt(attention_head_size))
        self.store_zonotope_bounds(attention_scores, "Attention scores")

        if self.verbose:
            attention_scores.print("attention score")

        if not self.args.keep_intermediate_zonotopes:
            del query
            del key

        attention_probs = attention_scores.softmax(verbose=self.verbose, no_constraints=not self.args.add_softmax_sum_constraint)
        self.store_zonotope_bounds(attention_probs, "Attention Probs")

        if self.args.log_softmax:
            num_softmaxes = attention_probs.zonotope_w.size(3)
            other_w = attention_probs.zonotope_w.transpose(0, 1).reshape(1 + attention_probs.num_error_terms, 4 * num_softmaxes, num_softmaxes)
            other = make_zonotope_new_weights_same_args(other_w, attention_probs)
            l, u = other.concretize()
            print("Layer %d: l < 0: %d instances       u > 1: %d instances" % (layer_num, (l < 0).sum().item(), (u > 1).sum().item()))
            print("l.min(): %f       u.max(): %f" % (l.min().item(), u.max().item()))
            # softmax_sum_w = attention_probs.zonotope_w.sum(dim=-1).unsqueeze(3)  # Shape: num attention heads x 1 + n_error_terms x words x 1
            # softmax_sum = make_zonotope_new_weights_same_args(softmax_sum_w, source_zonotope=attention_probs).remove_attention_heads_dim()
            # l, u = softmax_sum.concretize()  # Shape: num attention heads x words
            # l, u = l.squeeze(), u.squeeze()
            #
            # for head in range(4):
            #     print()
            #     print("Attention head %d" % head)
            #     print("Lower bound of softmax sum:")
            #     print(l[:, head])
            #     print("Upper bound of softmax sum:")
            #     print(u[:, head])

        if self.verbose:
            attention_probs.print("attention probs")

        value = bounds_input.dense(attention.self.value)

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        value = value.add_attention_heads_dim(num_attention_heads)
        self.store_zonotope_bounds(value, "Values")
        context = self.do_context(attention_probs, value, layer_num)
        self.store_zonotope_bounds(context, "Context")

        if self.verbose:
            value.print("value")
            context.print("context")

        context = context.remove_attention_heads_dim()
        self.store_zonotope_bounds(context, "Context Without Attention Head Dim")

        return attention_scores, attention_probs, context

    def _bound_pooling(self, bounds: Zonotope, pooler) -> Zonotope:
        # 1) They update the bounds and keep only some elements
        # 2) They multiply the bounds by the pooler's matrix
        # 3) They apply a tanh activation on the result
        bounds = make_zonotope_new_weights_same_args(new_weights=bounds.zonotope_w[:, :1, :], source_zonotope=bounds, clone=False)
        if self.verbose:
            bounds.print("pooling before dense")

        self.store_zonotope_bounds(bounds, "Pooling bounds")

        bounds = bounds.dense(pooler.dense)
        self.store_zonotope_bounds(bounds, "Pooling bounds -> Dense")

        if self.verbose:
            bounds.print("pooling pre-activation")

        if self.args.with_relu_in_pooling:
            bounds = bounds.relu()
        else:
            bounds = bounds.tanh()
        self.store_zonotope_bounds(bounds, "Pooling bounds -> Dense -> Tanh")

        if self.verbose:
            bounds.print("pooling after activation")
        return bounds

    def _bound_classifier(self, bounds: Zonotope, classifier) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) They compute linear layer that computes the how higher class 0 is over class 1
        # 2) They multiply the bounds by that linear layer's matrix
        # 3) They concretize the bounds (e.g. they compute the actual values, instead of having error terms)
        # 4) They check if things are safe or not (e.g. if the lower bound of c0 - c1 > 0, then we're good)
        if self.should_collect_zonotopes:
            bounds_normal = bounds.dense(classifier)
            self.store_zonotope_bounds(bounds_normal, "Classifier output")

        # TODO: the classifier weights below will be adapted by the optimizer, I need to fix this
        new_classifier = copy.deepcopy(classifier)
        new_classifier.weight = torch.nn.Parameter((classifier.weight[0:1, :] - classifier.weight[1:2, :]).clone().detach())
        new_classifier.bias = torch.nn.Parameter((classifier.bias[0] - classifier.bias[1]).clone().detach())

        if self.verbose:
            bounds.print("before dense")
            print(torch.norm(new_classifier.weight[0, :]))
            print(torch.mean(torch.norm(bounds.zonotope_w, dim=-2)))
            print(torch.mean(torch.norm(bounds.dense(new_classifier).zonotope_w, dim=-2)))

        bounds = bounds.dense(new_classifier)

        if self.verbose:
            bounds.print("after dense")

        l, u = bounds.concretize()
        # print("l and u shapes", l.shape, u.shape)
        return l[0][0], u[0][0]

    def get_safety(self, label: int, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> bool:
        if label == 0:
            model_is_robust = lower_bound > 0
        else:
            model_is_robust = upper_bound < 0

        return model_is_robust.item()
