# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
from datetime import datetime
from pathlib import Path
import time

import os
import torch
from typing import Tuple, Optional

from termcolor import colored

from Verifiers.Zonotope import cleanup_memory
from Verifiers.synonym_attack import get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms, \
    get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms_l_norm
from data_utils import sample, get_all_correctly_classified_samples, compute_accuracy


# can only accept one example in each batch
class Verifier:
    def __init__(self, args, target, logger):
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

        if not args.with_lirpa_transformer:
            self.embeddings = target.model.bert.embeddings
            self.encoding_layers = target.model.bert.encoder.layer
            self.pooler = target.model.bert.pooler
            self.classifier = target.model.classifier
        else:  # LIRPA model
            self.embeddings = target.model.embeddings
            self.encoding_layers = target.model.model_from_embeddings.bert.encoder.layer
            self.pooler = target.model.model_from_embeddings.bert.pooler
            self.classifier = target.model.model_from_embeddings.classifier

        self.hidden_act = args.hidden_act
        self.layer_norm = target.model.config.layer_norm if hasattr(target.model.config, "layer_norm") else "standard"

        reduc_method = args.error_reduction_method

        if "smaller" in args.dir:
            size = "smaller"
        elif "small" in args.dir:
            size = "small"
        else:
            size = "big"

        details = "NoConstraint" if (args.method != "zonotope" or not args.add_softmax_sum_constraint) else "WithConstraint"
        if self.args.use_other_dot_product_ordering:
            details += "OtherDotProductOrder"
        if self.args.use_dot_product_variant3:
            details += "DotProduceVariant3"
        if self.args.num_fast_dot_product_layers_due_to_switch != -1:
            if self.args.variant2plus1:
                details += f"Variant21With{self.args.num_fast_dot_product_layers_due_to_switch}FastLayers"
            else:
                details += f"Variant12With{self.args.num_fast_dot_product_layers_due_to_switch}FastLayers"
            details += f"WithNoise{self.args.max_num_error_terms_fast_layers}Symbols"

        self.res_filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
            "resultsSynonym" if self.args.attack_type == "synonym" else "results",
            args.data if not self.args.one_word_per_sentence else args.data + "Subset",   # SST or YELP
            args.lirpa_ckpt.split("/")[1].split('_')[1] if self.args.attack_type == "synonym" else args.dir,  # 3, 6, 12
            size,
            "zonotopeSlow" if (args.method == "zonotope" and args.zonotope_slow) else args.method,  # zonotope vs baf
            args.p if args.p <= 10 else 'inf',  # 1, 2, inf
            "None_None" if reduc_method == "None" else f"{reduc_method}_{args.max_num_error_terms}",
            details,
            datetime.now().strftime('%b%d_%H-%M-%S')
        )

    def run_sentence_attacks(self, data) -> Tuple[int, int, float]:
        """
        Returns:
            number of safe sentences, number of examples, percentage of sentence that are safe
        """
        with torch.no_grad():
            # compute_accuracy(self.args, data, self.target)

            # Accuracy(small3, adversarially trained):  1436/1821 = 78.5%
            # Accuracy(small3, normally trained):       1481/1821 = 81.3%
            # Accuracy(small3, certifiability trained): 1487/1821 = 81.5%

            examples = get_all_correctly_classified_samples(self.args, data, self.target)

            if self.args.attack_type != "synonym":
                print([' '.join(x['sent_a']) for x in examples])
                print("{} valid examples".format(len(examples)))

            file_path = os.path.join(self.results_directory, self.res_filename)
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            self.results_file = open(file_path, "w")  # Erase stuff
            self.results_file.write("sentence,isSafe,timing\n")

            num_tested_sentences = 0
            num_safe_sentences = 0
            num_sentence = 0
            num_eligible_sentences = 0

            for i, example in enumerate(examples):
                should_test_sentence = num_eligible_sentences + 1 >= self.args.start_sentence
                cleanup_memory()
                is_safe, eligible_to_test, num_enumerations, timing = self.verify_against_synonym_attack(example, should_test_sentence)
                was_tested = eligible_to_test and should_test_sentence

                if eligible_to_test:
                    num_eligible_sentences += 1

                if was_tested:
                    num_tested_sentences += 1
                if was_tested and is_safe:
                    num_safe_sentences += 1

                if was_tested:
                    self.results_file.write(f"{num_eligible_sentences},{1 if is_safe else 0},{timing}\n")
                    self.results_file.flush()
                    if is_safe:
                        print(colored(f"Is safe: yes    - Num enumerations {num_enumerations} - sentence {num_sentence}", 'green'))
                    else:
                        print(colored(f"Is safe: no- Num enumerations {num_enumerations} - sentence {num_sentence}", 'red'))
                    print(f"It took {timing:.3f} seconds to verify against a synonym attack")

                    num_sentence += 1

            print(f"Num tested sentences (high permutation only): {num_tested_sentences}")
            print(f"Num safe sentences (high permutation only): {num_safe_sentences}")
            print(f"Percentage of high permutation sentences that could be verifier {num_safe_sentences / num_tested_sentences * 100}%")

            self.results_file.close()
            return num_safe_sentences, num_tested_sentences, num_safe_sentences / num_tested_sentences

    def run(self, data):
        examples = sample(self.args, data, self.target)

        if self.args.attack_type != "synonym":
            print([' '.join(x['sent_a']) for x in examples])
            print("Lengths: ", [len(x['sent_a']) for x in examples])

        file_path = os.path.join(self.results_directory, self.res_filename)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        self.results_file = open(file_path, "w")  # Erase stuff
        self.results_file.write("sentence,position,eps,timing,memory\n")

        print("{} valid examples".format(len(examples)))
        sum_avg, sum_min = 0, 0
        results = []
        start_sentence_0_indexed = self.args.start_sentence - 1
        for i, example in enumerate(examples):
            if i < start_sentence_0_indexed:  # i is 0-indexed, start_se
                continue

            if i == self.args.sentence_to_skip:
                continue

            self.logger.write("Sample", i)

            backup_max_num, backup_max_num_fast = self.args.max_num_error_terms, self.args.max_num_error_terms_fast_layers
            big_sentence = len(example['sent_a']) > 20
            if big_sentence and self.args.max_num_error_terms >= 7000:
                self.args.max_num_error_terms = int(0.5 * self.args.max_num_error_terms)
                print(colored(f'Temporarily setting max num error terms to {self.args.max_num_error_terms}', 'yellow'))

            if big_sentence and self.args.max_num_error_terms >= 10000:
                self.args.max_num_error_terms_fast_layers = int(0.5 * self.args.max_num_error_terms_fast_layers)
                print(colored(f'Temporarily setting max num error fast terms to {self.args.max_num_error_terms_fast_layers}', 'yellow'))

            res = self.verify(example, example_num=i, first_example=(i == start_sentence_0_indexed))

            if big_sentence:
                self.args.max_num_error_terms = backup_max_num
                self.args.max_num_error_terms_fast_layers = backup_max_num_fast
                print(colored(f'Reset num error terms to {backup_max_num} and {self.args.max_num_error_terms_fast_layers}', 'yellow'))

            if self.debug:
                continue
            results.append(res[0])
            sum_avg += res[1]
            sum_min += res[2]

        self.results_file.close()

    def start_verification_new_input(self):
        pass

    def reset(self):
        pass

    def verify_against_synonym_attack(self, example, should_test_sentence=True) -> Tuple[bool, bool, int, float]:
        """ Verify the example sentence against a synonym attack """

        if self.args.p > 10:   # L-inf
            if self.method in ["baf", "backward", "forward"]:
                zonotope_weights_for_synonym_region, number_neighbors, num_enumerations, candidate_words_per_position, center, radius = \
                    get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms(
                        example, model=self.target, max_synonyms_per_word=1000  # self.args.max_synonyms_per_word
                )
            else:
                zonotope_weights_for_synonym_region, number_neighbors, num_enumerations, candidate_words_per_position = \
                    get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms_l_norm(
                        example, model=self.target, max_synonyms_per_word=1000, p=self.p, use_solver=self.args.compute_synonym_region_using_solver
                    )
        elif self.args.p == 1 or self.args.p == 2:  # L1 and L2
            zonotope_weights_for_synonym_region, number_neighbors, num_enumerations, candidate_words_per_position = \
                get_zonotope_weights_representing_embeddings_for_sentence_and_synonyms_l_norm(
                    example, model=self.target, max_synonyms_per_word=1000, p=self.p, use_solver=self.args.compute_synonym_region_using_solver  # self.args.max_synonyms_per_word
                )
            center, radius = None, None
        else:
            assert NotImplementedError(f"Synonym attack can currently capture the embedding region with either L1, L2 or L-inf zonotopes,"
                                       f" but p={self.args.p}")

        # if num_enumerations > 100_000:
        num_words = len(example['sent_a'])
        embedding_words = zonotope_weights_for_synonym_region.size(1)
        self.args.num_input_error_terms *= embedding_words
        self.args.perturbed_words = embedding_words

        eligible_to_test = num_enumerations >= 32000 and num_words < 27
        if eligible_to_test and should_test_sentence:
            if self.args.attack_type == "synonym":
                self.args.num_perturbed_words = zonotope_weights_for_synonym_region.size(1)

                # Required for concretization of Bound and Layer in the CROWN verifiers
                if self.method in ["baf", "backward", "forward"]:
                    self.args.embedding_radii = radius

                # embeddings_previous_code = center.unsqueeze(0)
                embeddings = zonotope_weights_for_synonym_region[0:1]
            else:
                embeddings = None

            start = time.time()
            is_safe = self.verify_safety(example, embeddings=embeddings, index=1, eps="Synonym attack", initial_zonotope_weights=zonotope_weights_for_synonym_region)
            end = time.time()
            timing = end - start
        else:
            is_safe = False
            timing = -1.0
                # self.args.max_num_error_terms = self.args.max_num_error_terms // 2
            # if num_words > 25:
            #     self.args.max_num_error_terms = self.args.max_num_error_terms * 2

        self.args.num_input_error_terms = self.args.num_input_error_terms // embedding_words

        if eligible_to_test:
            print(f"\nNumber of neighbors: {number_neighbors}")
            print(f"Number of enumerations: {num_enumerations}")
            print(f"Sentence length:  {num_words}")
            print(f"Sentence: {example['sent_a']}")
            print(f"Num candidates per pos: {[len(x) for x in candidate_words_per_position]}")
            print(f"Replacements: {candidate_words_per_position}")

        return is_safe, eligible_to_test, num_enumerations, timing

    def get_bounds_difference_in_scores(
            self, embeddings: torch.Tensor, index: int, eps: float, gradient_enabled=False, *args, **kwargs
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    # These 2 methods below are specialized for the binary classification case
    def get_loss(self, embeddings: torch.Tensor, eps: float, label: int, perturbed_word_index: int) -> torch.Tensor:
        # Concrete lower and upper bounds of the expression (score(class 0) - score(class 1))
        l, u  = self.get_bounds_difference_in_scores(embeddings, perturbed_word_index, eps, gradient_enabled=True)

        if label == 0:
            loss = -l  # c0 - c1 in [-3, 5] => c1 - c0 in [-5, 3] => max(score(class 0) - score(class 1)) = -(-3)
        else:
            loss = u   # c0 - c1 in [-3, 5]                       => max(score(class 0) - score(class 1)) = 5

        return loss

    def train_diffai(self, example, eps: float):
        if self.args.with_lirpa_transformer:
            embeddings, _, _, _ = self.target.get_input([example])
        else:
            embeddings, _ = self.target.get_embeddings([example])

        start_word = 1
        length = embeddings.shape[1]
        for perturbed_word_index in range(start_word, length - 1):
            with torch.autograd.set_detect_anomaly(True):
                loss = self.get_loss(embeddings, eps, example["label"], perturbed_word_index)

                self.target.optimizer.zero_grad()
                loss.backward()
                self.target.optimizer.step()

    def verify(self, example, example_num: int, first_example=False):
        """ Verify the given example sentence """
        start_time = time.time()

        if self.args.with_lirpa_transformer:
            embeddings, extended_attention_mask, tokens, label_ids = self.target.get_input([example])
        else:
            embeddings, tokens = self.target.get_embeddings([example])

        embeddings = embeddings if self.args.cpu else embeddings.cuda()
        length = embeddings.shape[1]
        tokens = tokens[0]

        self.logger.write("tokens:", " ".join(tokens))
        self.logger.write("length:", length)
        self.logger.write("label:", example["label"])

        self.std = self.target.step([example])[-1]

        result = {
            "tokens": tokens,
            "label": float(example["label"]),
            "bounds": []
        }


        if False:  # self.debug:
            eps = self.eps
            index = self.debug_pos
            safety = self.verify_safety(example, embeddings, index, self.eps)
            self.logger.write("Time elapsed", time.time() - start_time)
            return eps
        else:
            eps = torch.zeros(length)
            num_iters = self.num_verify_iters

            cnt = 0
            sum_eps, min_eps = 0, 1e30

            max_refinement_iterations = 12
            iteration_num = 0

            if self.args.all_words:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.max_memory_allocated()

                start = time.time()
                self.start_verification_new_input()
                # skip OOV

                l, r = 0, self.max_eps
                print("{:.5f} {:.5f}".format(l, r), end="")

                torch.cuda.reset_peak_memory_stats()
                safe = self.verify_safety(example, embeddings, index=None, eps=r)
                max_memory_used = torch.cuda.max_memory_allocated()
                # print(f"   Max memory used: {max_memory_used}")

                while safe:
                    l = r
                    r *= 2
                    print("\r{:.5f} {:.5f}".format(l, r), end="")
                    torch.cuda.reset_peak_memory_stats()
                    safe = self.verify_safety(example, embeddings, index=None, eps=r)
                    max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                if l == 0:
                    while not safe:
                        r /= 2
                        print("\r{:.5f} {:.5f}".format(l, r), end="")
                        torch.cuda.reset_peak_memory_stats()
                        safe = self.verify_safety(example, embeddings, index=None, eps=r)
                        max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                    l, r = r, r * 2
                    print("\r{:.5f} {:.5f}".format(l, r), end="")
                for j in range(num_iters):
                    m = (l + r) / 2
                    torch.cuda.reset_peak_memory_stats()
                    if self.verify_safety(example, embeddings, index=None, eps=m):
                        l = m
                    else:
                        r = m
                    max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                    print("\r{:.5f} {:.5f}".format(l, r), end="")
                print()
                end = time.time()
                print("Binary search for max eps took {:.4f} seconds\n".format(end - start))

                self.results_file.write(
                    f"{example_num},{l},{end - start:f},{max_memory_used}\n"
                )
                self.results_file.flush()
            elif self.perturbed_words == 1:
                # warm up
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
                        while not went_down and iteration_num < max_refinement_iterations and self.verify_safety(example, embeddings, 1, self.max_eps):
                            print(f"eps {self.max_eps} - was safe - multiplying by 2")
                            self.max_eps *= 2
                            iteration_num += 1
                    self.warmed = True
                    print("Approximate maximum eps:", self.max_eps)

                # [CLS] and [SEP] cannot be perturbed
                start_word = 1 if not first_example else self.args.start_word

                # Consult generate_random_indices.py
                # These number were generated once in a pre-processing step and then re-used
                # for all verifiers, to ensure we are doing fair comparisons but without introducing
                # any bias
                PREGENERATED_RANDOM_INDEX = [10, 13, 4, 12, 7, 11, 3, 6, 3, 2]

                torch.cuda.reset_peak_memory_stats()
                torch.cuda.max_memory_allocated()

                for i in range(start_word, length - 1):
                    if self.args.one_word_per_sentence:
                        if PREGENERATED_RANDOM_INDEX[example_num] != i:
                            continue
                        else:
                            print(colored(f"For sentence {example_num}, only processing word {i}", 'yellow'))

                    iteration_num = 0
                    start = time.time()
                    self.start_verification_new_input()
                    # skip OOV
                    if tokens[i][0] == "#" or tokens[i + 1][0] == "#":
                        continue

                    cnt += 1

                    l, r = 0, self.max_eps
                    print("{} {:.5f} {:.5f}".format(i, l, r), end="")

                    torch.cuda.reset_peak_memory_stats()
                    safe = self.verify_safety(example, embeddings, i, r)
                    max_memory_used = torch.cuda.max_memory_allocated()
                    # print(f"   Max memory used: {max_memory_used}")

                    while safe and iteration_num < max_refinement_iterations:
                        l = r
                        r *= 2
                        print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
                        torch.cuda.reset_peak_memory_stats()
                        safe = self.verify_safety(example, embeddings, i, r)
                        max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                        # print(f"   Max memory used: {max_memory_used}")
                        iteration_num += 1

                    if l == 0:
                        while not safe and iteration_num < max_refinement_iterations:
                            r /= 2
                            print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
                            torch.cuda.reset_peak_memory_stats()
                            safe = self.verify_safety(example, embeddings, i, r)
                            max_memory_used = max(max_memory_used, torch.cuda.max_memory_allocated())
                            # print(f"   Max memory used: {max_memory_used}")
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
                        # print(f"   Max memory used: {max_memory_used}")
                        print("\r{} {:.5f} {:.5f}".format(i, l, r), end="")
                    print()
                    eps[i] = l
                    self.logger.write("Position {}: {} {:.5f}".format(i, tokens[i], eps[i], ))
                    sum_eps += eps[i]
                    min_eps = min(min_eps, eps[i])
                    norm = torch.norm(embeddings[0, i, :], p=self.p)
                    result["bounds"].append({
                        "position": i,
                        "eps": float(eps[i]),
                        "eps_normalized": float(eps[i] / norm)
                    })
                    end = time.time()
                    print("Binary search for max eps took {:.4f} seconds\n".format(end - start))

                    self.results_file.write(
                        f"{example_num},{i},{eps[i]},{end - start:f},{max_memory_used}\n"
                    )
                    self.results_file.flush()
            elif self.perturbed_words == 2:
                # warm up
                if not self.warmed:
                    self.start_verification_new_input()
                    print("Warming up...")
                    while not self.verify_safety(example, embeddings, [1, 2], self.max_eps):
                        self.max_eps /= 2
                    while self.verify_safety(example, embeddings, [1, 2], self.max_eps):
                        self.max_eps *= 2
                    self.warmed = True
                    print("Approximate maximum eps:", self.max_eps)

                for i1 in range(1, length - 1):
                    for i2 in range(i1 + 1, length - 1):
                        self.start_verification_new_input()
                        # skip OOV
                        if tokens[i1][0] == "#" or tokens[i1 + 1][0] == "#":
                            continue
                        if tokens[i2][0] == "#" or tokens[i2 + 1][0] == "#":
                            continue

                        cnt += 1

                        l, r = 0, self.max_eps
                        print("%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        safe = self.verify_safety(example, embeddings, [i1, i2], r)
                        while safe:
                            l = r
                            r *= 2
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                            safe = self.verify_safety(example, embeddings, [i1, i2], r)
                        if l == 0:
                            while not safe:
                                r /= 2
                                print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                                safe = self.verify_safety(example, embeddings, [i1, i2], r)
                            l, r = r, r * 2
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        for j in range(num_iters):
                            m = (l + r) / 2
                            if self.verify_safety(example, embeddings, [i1, i2], m):
                                l = m
                            else:
                                r = m
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        print()
                        eps = l
                        self.logger.write("Position %d %d: %s %s %.5f" % (
                            i1, i2, tokens[i1], tokens[i2], eps))
                        sum_eps += eps
                        min_eps = min(min_eps, eps)
                        result["bounds"].append({
                            "position": (i1, i2),
                            "eps": float(eps)
                        })
            else:
                raise NotImplementedError

            result["time"] = time.time() - start_time

            self.logger.write(f"Time elapsed for sentence {example_num}", result["time"])

            if cnt == 0:
                return result, sum_eps, min_eps
            else:
                return result, sum_eps / cnt, min_eps

    def verify_safety(self, example, embeddings, index, eps, **kwargs) -> bool:
        raise NotImplementedError
