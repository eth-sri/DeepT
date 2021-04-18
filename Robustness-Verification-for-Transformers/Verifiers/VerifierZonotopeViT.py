import os
import random
import time
from pathlib import Path
from typing import Tuple, Optional

from datetime import datetime
import torch
from einops.einops import repeat
from einops.layers.torch import Rearrange

from Verifiers.Verifier import Verifier
from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args, cleanup_memory
from vit import ViT


def get_layernorm(x):
    return x.fn


def get_inner(x):
    return x.fn.fn


def sample_correct_samples(args, data, target):
    examples = []
    for i in range(args.samples):
        while True:
            example = data[random.randint(0, len(data) - 1)]
            logits = target(example["image"])
            prediction = torch.argmax(logits, dim=-1)

            if prediction != example["label"]:
                continue  # incorrectly classified

            examples.append(example)
            break

    return examples


class VerifierZonotopeViT(Verifier):
    def __init__(self, args, target: ViT, logger, num_classes: int, normalizer):
        self.args = args
        self.device = args.device
        self.target = target
        self.logger = logger
        self.res = args.res
        self.results_directory = args.results_directory

        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps
        self.debug = args.debug
        self.verbose = args.debug or args.verbose
        self.method = args.method
        self.num_verify_iters = args.num_verify_iters
        self.max_eps = args.max_eps
        self.debug_pos = args.debug_pos
        self.perturbed_words = args.perturbed_words
        self.warmed = False

        self.hidden_act = args.hidden_act
        self.layer_norm = target.layer_norm_type
        self.normalizer = normalizer

        time_tag = datetime.now().strftime('%b%d_%H-%M-%S')
        self.res_filename = f"resultsVit_p_{args.p}_{time_tag}.csv"

        self.ibp = args.method == "ibp"

        self.showed_warning = False
        self.target: ViT
        self.num_classes = num_classes

    def run(self, data):
        examples = sample_correct_samples(self.args, data, self.target)

        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        self.results_file = open(file_path, "w")  # Erase stuff
        self.results_file.write("index,eps,timing,memory\n")

        print("{} valid examples".format(len(examples)))
        sum_avg, minimum = 0, 100000000000000000
        for i, example in enumerate(examples):
            self.logger.write("Sample", i)

            res = self.verify(example, example_num=i)
            sum_avg += res[0]
            minimum = min(float(res[1]), minimum)

        self.logger.write("{} valid examples".format(len(examples)))
        self.logger.write("Minimum: {:.5f}".format(minimum))
        self.logger.write("Average: {:.5f}".format(float(sum_avg) / len(examples)))

        self.results_file.close()

    def verify(self, example, example_num: int):
        """ Verify the given example sentence """
        embeddings = example["image"]

        embeddings = embeddings if self.args.cpu else embeddings.cuda()

        num_iters = self.num_verify_iters

        cnt = 0
        sum_eps, min_eps = 0, 1e30

        max_refinement_iterations = 12
        iteration_num = 0

        if self.perturbed_words == 1:
            # warm up
            # import pdb; pdb.set_trace()
            if not self.warmed:
                if self.args.hardcoded_max_eps:
                    print(f"Using hardcoded max eps: {self.max_eps}")
                else:
                    self.start_verification_new_input()
                    print("Warming up...")
                    went_up, went_down = False, False

                    while iteration_num < max_refinement_iterations and not self.verify_safety(example, embeddings, 1, self.max_eps):
                        print(f"eps {self.max_eps} - wasn't safe - dividing by 2")
                        self.max_eps /= 2
                        went_down = True
                        iteration_num += 1
                    while not went_down and iteration_num < max_refinement_iterations and self.verify_safety(example, embeddings, 1,
                                                                                                             self.max_eps):
                        print(f"eps {self.max_eps} - was safe - multiplying by 2")
                        self.max_eps *= 2
                        iteration_num += 1
                self.warmed = True
                print("Approximate maximum eps:", self.max_eps)

            torch.cuda.reset_peak_memory_stats()
            torch.cuda.max_memory_allocated()

            iteration_num = 0
            start = time.time()
            self.start_verification_new_input()

            cnt += 1

            l, r = 0, self.max_eps
            i = 0
            print("{} {:.5f} {:.5f}".format(i, l, r), end="")

            torch.cuda.reset_peak_memory_stats()
            safe = self.verify_safety(example, embeddings, i, r)
            max_memory_used = torch.cuda.max_memory_allocated()

            while safe and iteration_num < max_refinement_iterations:
                l = r
                r *= 2
                print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
                torch.cuda.reset_peak_memory_stats()
                safe = self.verify_safety(example, embeddings, i, r)
                max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                iteration_num += 1

            if l == 0:
                while not safe and iteration_num < max_refinement_iterations:
                    r /= 2
                    print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
                    torch.cuda.reset_peak_memory_stats()
                    safe = self.verify_safety(example, embeddings, i, r)
                    max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                    iteration_num += 1

                l, r = r, r * 2
                print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
            for j in range(num_iters):
                m = (l + r) / 2
                torch.cuda.reset_peak_memory_stats()
                if self.verify_safety(example, embeddings, i, m):
                    l = m
                else:
                    r = m
                max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")

            print()

            self.logger.write(f"Position {i}: {l:.5f}")
            sum_eps += l
            min_eps = min(min_eps, l)
            end = time.time()
            print("Binary search for max eps took {:.4f} seconds\n".format(end - start))

            self.results_file.write(
                f"{example_num},{l},{end - start:f},{max_memory_used}\n"
            )
            self.results_file.flush()
        else:
            raise NotImplementedError

        if cnt == 0:
            return sum_eps, min_eps
        else:
            return sum_eps / cnt, min_eps

    def verify_safety(self, example, image, index, eps):
        zonotope_difference_scores = self.get_bounds_difference_in_scores(image, eps)
        if zonotope_difference_scores is None:
            model_is_robust = False  # There was an exception or some pre-conditions were not met
        else:
            model_is_robust = self.get_safety(example["label"], zonotope_difference_scores)

        return model_is_robust

    def get_bounds_difference_in_scores(self, image: torch.Tensor, eps: float) -> Optional[Zonotope]:
        if self.args.error_reduction_method == "None" and not self.showed_warning:
            START_WARNING = '\033[93m'
            END_WARNING = '\033[0m'
            print(START_WARNING + "Warning: No error reduction method for Zonotope verifier!" + END_WARNING)
            self.showed_warning = True

        cleanup_memory()
        errorType = OSError if self.debug else AssertionError

        try:
            with torch.no_grad():
                bounds = self._bound_input(image, eps=eps)  # hard-coded yet

                if self.args.log_error_terms_and_time: print("Input to attention layer 0 has %d error terms" % bounds.num_error_terms)

                for i, (attn, ff) in enumerate(self.target.transformer.layers):
                    if self.args.log_error_terms_and_time:
                        print()
                        print("Layer %d" % i)

                    start = time.time()
                    attention_scores, attention_probs, self_attention_output, bounds = self._bound_layer(bounds, attn, ff, layer_num=i)
                    end = time.time()

                    if self.args.log_error_terms_and_time:
                        print("Output of attention layer %d has %d error terms" % (i, bounds.num_error_terms))
                        print("Time to do attention layer %d is %.3f seconds" % (i, end - start))

                    if not self.args.keep_intermediate_zonotopes:
                        del attention_scores, attention_probs, self_attention_output
                        cleanup_memory()

                bounds = self._bound_pooling(bounds)
                bounds = self._bound_classifier(bounds)
                return bounds
        except errorType as err:  # for debug
            if self.verbose:
                print("Warning: failed assertion")
                print(err)
            # print("Warning: failed assertion", eps)
            # print(err)
            # raise err
            return None

    def _bound_input(self, image: torch.Tensor, eps: float) -> Zonotope:
        # Rearrange
        patch_size = self.target.patch_size
        rearrange = Rearrange('1 c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        image = rearrange(image)

        eps_scaled = eps / self.normalizer.std[0]  # Take into account the normalization
        bounds = Zonotope(self.args, p=self.p, eps=eps_scaled,
                          perturbed_word_index=None, value=image,
                          start_perturbation=0, end_perturbation=image.shape[0])

        # First linear
        bounds = bounds.dense(self.target.to_patch_embedding[1])

        e, n, _ = bounds.zonotope_w.shape

        # Get class tokens
        cls_tokens = repeat(self.target.cls_token, '() n d -> n d')
        cls_tokens_value_w = cls_tokens.unsqueeze(0)  # 1 n d
        cls_tokens_errors_w = torch.zeros_like(cls_tokens).unsqueeze(0).repeat(e - 1, 1, 1)  # (e - 1) n d
        cls_tokens_zonotope_w = torch.cat([cls_tokens_value_w, cls_tokens_errors_w], dim=0)  # e n d

        # Insert class tokens
        full_zonotope_w = torch.cat((cls_tokens_zonotope_w, bounds.zonotope_w), dim=1)
        bounds = make_zonotope_new_weights_same_args(full_zonotope_w, bounds, clone=False)

        # Add position embeddings
        bounds = bounds.add(self.target.pos_embedding[:, :(n + 1)])

        return bounds

    def is_in_fast_layer(self, layer_num: int):
        return True

    def _bound_layer(self, bounds_input: Zonotope, attn, ff, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        if self.args.log_error_terms_and_time:
            print("Bound_input before error reduction has %d error terms" % bounds_input.num_error_terms)

        # We recenter the zonotope and eliminate the error_term_ranges (the new zonotope is equivalent).
        # If we didn't do it, the Box or PCA error term reduction functions would have to take the errror ranges
        # into account and that would make things more complicated. By simply re-writing the zonotope in an equaivalent
        # fashion such that the error_ranges are eliminated, life is simpler and we don't have to worry about the error ranges anymore
        if bounds_input.error_term_range_low is not None:
            bounds_input = bounds_input.recenter_zonotope_and_eliminate_error_term_ranges()

        if self.args.error_reduction_method == 'box':
            bounds_input_reduced_box = bounds_input.reduce_num_error_terms_box(max_num_error_terms=self.args.max_num_error_terms)
            bounds_input = bounds_input_reduced_box

        layer_normed = bounds_input.layer_norm(get_layernorm(attn).norm, get_layernorm(attn).layer_norm_type)  # Layer norm 1

        # main self-attention
        attention_scores, attention_probs, context, attention = self._bound_attention(
            layer_normed, get_inner(attn), layer_num=layer_num
        )  # attention

        bounds_input = bounds_input.expand_error_terms_to_match_zonotope(attention)
        attention = attention.add(bounds_input)  # residual

        attention_layer_normed = attention.layer_norm(get_layernorm(ff).norm, get_layernorm(ff).layer_norm_type)  # prenorm 2

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        feed_forward = get_inner(ff)

        intermediate = attention_layer_normed.dense(feed_forward.net[0])  # FeedForward - Linear 1
        intermediate = intermediate.relu()  # FeedForward - ReLU
        dense = intermediate.dense(feed_forward.net[3])  # FeedForward - Linear 2

        attention = attention.expand_error_terms_to_match_zonotope(intermediate)
        dense = dense.add(attention)  # Residual 2

        if not self.args.keep_intermediate_zonotopes:
            del intermediate
            del attention

        output = dense


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

    def _bound_attention(self, bounds_input: Zonotope, attn, layer_num=-1) -> Tuple[Zonotope, Zonotope, Zonotope, Zonotope]:
        num_attention_heads = attn.heads

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
        query = bounds_input.dense(attn.to_q)
        key = bounds_input.dense(attn.to_k)

        query = query.add_attention_heads_dim(num_attention_heads)  # (A, 1 + #error, length, E)
        key = key.add_attention_heads_dim(num_attention_heads)  # (A, 1 + #error, length, E)

        attention_scores = self.do_dot_product(query, key, layer_num)
        attention_scores = attention_scores.multiply(attn.scale)

        if not self.args.keep_intermediate_zonotopes:
            del query
            del key

        attention_probs = attention_scores.softmax(verbose=self.verbose, no_constraints=not self.args.add_softmax_sum_constraint)

        value = bounds_input.dense(attn.to_v)

        if not self.args.keep_intermediate_zonotopes:
            del bounds_input

        value = value.add_attention_heads_dim(num_attention_heads)
        context = self.do_context(attention_probs, value, layer_num)

        if self.verbose:
            value.print("value")
            context.print("context")

        context = context.remove_attention_heads_dim()

        attention = context.dense(attn.to_out[0])

        return attention_scores, attention_probs, context, attention

    def _bound_pooling(self, bounds: Zonotope) -> Zonotope:
        bounds = make_zonotope_new_weights_same_args(new_weights=bounds.zonotope_w[:, :1, :], source_zonotope=bounds, clone=False)
        return bounds

    def _bound_classifier(self, bounds: Zonotope) -> Zonotope:
        bounds = bounds.layer_norm(self.target.mlp_head[0], self.target.layer_norm_type)
        bounds = bounds.dense(self.target.mlp_head[1])
        return bounds

    def get_safety(self, label: int, classifier_bounds: Zonotope) -> bool:
        pos = 0
        label = int(label)

        zonotope_w = classifier_bounds.zonotope_w
        diff = torch.zeros((zonotope_w.shape[0], 1, self.num_classes - 1), device=zonotope_w.device)
        for i in range(self.num_classes):
            if i != label:
                diff[:, 0, pos] = zonotope_w[:, 0, label] - zonotope_w[:, 0, i]
                pos += 1

        zonotope = make_zonotope_new_weights_same_args(diff, source_zonotope=classifier_bounds, clone=False)
        l, u = zonotope.concretize()
        return (torch.min(l) > 0).item()
