import math
import random
import gc
import time
from argparse import Namespace
from math import sqrt
from typing import Union, Tuple, Optional, List

import numpy as np
import torch
import torch.linalg
import torch.nn.functional as F
from numpy.linalg.linalg import svd as numpy_svd
from opt_einsum import contract
from termcolor import colored

from Verifiers.error_range_updater import get_updated_error_ranges_using_LP

from Verifiers.utils import INFINITY, dual_norm, DUAL_INFINITY

epsilon = 1e-12


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_bigger_smaller_tensor(x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x1.size(0) >= x2.size(0):
        return x1, x2
    else:
        return x2, x1


def sum_tensor_with_different_dim0_(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    bigger, smaller = get_bigger_smaller_tensor(x1, x2)
    bigger[:smaller.size(0)] += smaller
    return bigger


def sum_tensor_with_different_dim0_1(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    A1, B1, C, D = x1.shape
    A2, B2, C, D = x2.shape
    result = torch.zeros(max(A1, A2), max(B1, B2), C, D, device=x1.device)

    result[:A1, :B1] += x1
    result[:A2, :B2] += x2

    return result


def tensor_and(*tensors: torch.Tensor) -> torch.Tensor:
    result = torch.logical_and(tensors[0], tensors[1])
    for i in range(2, len(tensors)):
        result = torch.logical_and(result, tensors[i])
    return result


def fillna(tensor: torch.Tensor, val: float) -> None:
    tensor[tensor != tensor] = val


def get_norm(tensor: torch.Tensor, p: float, dim: Union[int, List[int]]) -> torch.Tensor:
    return torch.linalg.norm(tensor, ord=p, dim=dim)
    # return torch.norm(tensor, p=p, dim=dim)


def get_num_error_terms(zonotope_weights: torch.Tensor) -> int:
    if zonotope_weights.ndim == 4:
        return zonotope_weights.size(1) - 1
    else:
        return zonotope_weights.size(0) - 1


def make_zonotope_new_weights_same_args(new_weights: torch.Tensor, source_zonotope: "Zonotope", clone=True) -> "Zonotope":
    # If there are more error terms in the new weights than in the source zonotope, we make sure the error ranges for
    # the new terms are [-1, 1]. To do so, we extend them.
    z = source_zonotope
    error_term_range_low, error_term_range_high = z.error_term_range_low, z.error_term_range_high

    if error_term_range_low is not None:
        assert (error_term_range_low <= error_term_range_high).all(), \
            "Error: some error ranges are empty. Low: %s High: %s" % (error_term_range_low, error_term_range_high)

        current_num_error_ranges = error_term_range_low.size(0)
        new_weights_num_error_ranges = get_num_error_terms(new_weights)
        num_error_ranges_there_should_be = new_weights_num_error_ranges - z.num_input_error_terms_special_norm

        if current_num_error_ranges < num_error_ranges_there_should_be:
            num_new_terms_needed = num_error_ranges_there_should_be - current_num_error_ranges
            error_term_range_low = torch.cat([error_term_range_low, -torch.ones(num_new_terms_needed, device=new_weights.device)])
            error_term_range_high = torch.cat([error_term_range_high, torch.ones(num_new_terms_needed, device=new_weights.device)])

        if current_num_error_ranges > new_weights_num_error_ranges:
            raise Exception("Current number of errors (%d) is higher than the error in the new weights (%d)" % (current_num_error_ranges, new_weights_num_error_ranges))

    return Zonotope(args=z.args,
                    p=z.p,
                    eps=z.eps,
                    perturbed_word_index=z.perturbed_word_index,
                    zonotope_w=new_weights,
                    error_term_range_low=error_term_range_low,
                    error_term_range_high=error_term_range_high,
                    clone=clone)


def sample_vector_with_p_norm_below_or_equal_to_1(number_values: int, p: float, binary_value=False) -> torch.Tensor:
    if p == 1:
        raise NotImplementedError("sample vector for p=1 not implemented")
    elif p == 2:
        values = torch.normal(mean=0.0, std=1.0, size=[number_values])
        norm2 = get_norm(values, p=2.0, dim=0).item()
        values_normalized = values / norm2

        radius = torch.rand(1).item()
        error_values = values_normalized * radius

        assert get_norm(error_values, p=2.0, dim=0).item() <= 1.0

        return error_values
    elif p > 10:
        if binary_value:
            error_term_values = torch.rand(number_values)
            error_term_binary_values = torch.zeros_like(error_term_values)
            error_term_binary_values[error_term_values < 0.5] = -1.0
            error_term_binary_values[error_term_values >= 0.5] = 1.0
            return error_term_binary_values
        else:
            val = torch.rand(number_values) * 2 - 1
            val = val.clamp(-1.0, 1.0)
            return val
    else:
        raise Exception("Invalid p value")


class Zonotope:
    def __init__(self, args: Namespace, p: float, eps: float, perturbed_word_index: int, value: torch.Tensor = None,
                 zonotope_w: torch.Tensor = None,
                 error_term_range_low: torch.Tensor = None, error_term_range_high: torch.Tensor = None, clone=True,
                 start_perturbation=None, end_perturbation=None):
        self.args = args
        self.device = value.device if value is not None else zonotope_w.device
        self.p = p
        self.dual_p = dual_norm(p)
        self.eps = eps

        assert args.perturbed_words == 1 or self.args.attack_type == "synonym", "We assume there is only one perturbed word in the Zonotope implementation"
        self.perturbed_word_index = perturbed_word_index

        # If we do the equality constraint in the softmax, this range might be reduced for some error terms
        self.error_term_range_low: torch.Tensor = error_term_range_low  # Should be -1 normally
        self.error_term_range_high: torch.Tensor = error_term_range_high  # Should be +1 normally

        if zonotope_w is not None:
            assert not torch.isnan(zonotope_w).any(), "Zonotope creation: Some values in zonotope_w are NaNs"

            self.zonotope_w = zonotope_w.clone() if clone else zonotope_w
            if zonotope_w.ndim == 3:
                self.num_error_terms = zonotope_w.shape[0] - 1
                self.num_words = zonotope_w.shape[1]
                self.word_embedding_size = zonotope_w.shape[2]
            else:
                self.num_error_terms = zonotope_w.shape[1] - 1
                self.num_words = zonotope_w.shape[2]
                self.word_embedding_size = zonotope_w.shape[3]
        else:
            self.num_words, self.word_embedding_size = value.shape

            start = 1 if start_perturbation is None else start_perturbation
            end = self.num_words - 1 if end_perturbation is None else end_perturbation

            if self.args.all_words:
                self.num_error_terms = self.word_embedding_size * (end - start)
            else:
                self.num_error_terms = self.word_embedding_size

            self.zonotope_w = torch.zeros([
                1 + self.num_error_terms,  # Bias + error terms for the perturbed word
                self.num_words,
                self.word_embedding_size,
            ], device=args.device)

            # fills biases (here we don't know the range of the embedding, so we don't do the "move eps" trick we did in the RIAI project)
            self.zonotope_w[0, :, :] = value

            # Create error weights. Only the perturbed word has error terms, not the others
            # zonotope[0, 2, 0]: first dim of 2nd word embedding should have bias equal to its value
            # zonotope[1, 2, 0]: first dim of 2nd word embedding should have weight eps for its own error
            # zonotope[2, 2, 0]: first dim of 2nd word embedding should have weight 0 for error of other perturbed dims
            # zonotope[..., 2, 0]: first dim of 2nd word embedding should have weight 0 for error of other perturbed dims
            # zonotope[embedding_dim + 1, 2, 0]: first dim of 2nd word embedding should have weight 0 for error of other perturbed dims

            # in general, the only indices that should have weight=eps are [i + 1, perturbed_word, i]
            # which means that the i-th coordinate of the embedding of the perturbed word should depend on the i-th error term
            if self.args.all_words:
                error_index = 1

                for w in range(start, end):
                    for j in range(self.word_embedding_size):
                        self.zonotope_w[error_index, w, j] = self.eps
                        error_index += 1
            else:
                for i in range(1, 1 + self.num_error_terms):
                    self.zonotope_w[i, perturbed_word_index, i - 1] = self.eps

        self.num_input_error_terms: int = args.num_input_error_terms
        self.num_input_error_terms_special_norm = self.num_input_error_terms if (self.p == 1 or self.p == 2) else 0

    def to_device(self, val: torch.Tensor):
        return val if self.args.cpu else val.cuda()

    ######################
    # Shape manipulation #
    ######################

    def has_error_terms(self) -> bool:
        error_terms = self.zonotope_w[1:] if self.zonotope_w.ndim <= 3 else self.zonotope_w[:, 1:]
        return (error_terms != 0).any().item()

    def get_num_error_terms(self) -> int:
        error_dim = 0 if self.zonotope_w.ndim == 3 else 1

        assert self.num_error_terms == self.zonotope_w.shape[error_dim] - 1, \
            "self.num_error_terms value (%d) doesn't match actual number of error terms (%d)" \
            % (self.num_error_terms, self.zonotope_w.shape[error_dim])

        return self.num_error_terms

    def compact_errors(self) -> "Zonotope":
        assert self.error_term_range_low is None, "compact error: low"
        assert self.error_term_range_high is None, "compact error: high"

        if self.zonotope_w.ndim == 4:
            error_terms = self.zonotope_w[:, 1:]
            num_non_zero_coeffs_per_error_term = (error_terms != 0).sum(dim=[0, 2, 3])
            to_keep = (num_non_zero_coeffs_per_error_term > 0)
            error_terms_to_keep = error_terms[:, to_keep]

            zonotope_w = torch.cat([self.zonotope_w[:, 0], error_terms_to_keep], dim=1)
        else:
            error_terms = self.zonotope_w[1:]
            num_non_zero_coeffs_per_error_term = (error_terms != 0).sum(dim=[1, 2])
            to_keep = (num_non_zero_coeffs_per_error_term > 0)
            error_terms_to_keep = error_terms[to_keep]

            zonotope_w = torch.cat([self.zonotope_w[0], error_terms_to_keep], dim=0)

        return make_zonotope_new_weights_same_args(zonotope_w, source_zonotope=self, clone=False)

    def expand_error_terms_to_match_zonotope(self, other: "Zonotope") -> "Zonotope":
        difference_error_terms = other.get_num_error_terms() - self.get_num_error_terms()
        if difference_error_terms < 0:
            raise Exception("Other zonotope has less error terms!")
        elif difference_error_terms == 0:
            return self

        if self.zonotope_w.ndim == 3:
            extra_zeros = torch.zeros(difference_error_terms, self.zonotope_w.shape[1], self.zonotope_w.shape[2], device=self.args.device)
            new_zonotope_w = torch.cat([self.zonotope_w, extra_zeros], dim=0)
        else:
            extra_zeros = torch.zeros(self.zonotope_w.shape[0], difference_error_terms, self.zonotope_w.shape[2], self.zonotope_w.shape[3], device=self.args.device)
            new_zonotope_w = torch.cat([self.zonotope_w, extra_zeros], dim=1)

        return make_zonotope_new_weights_same_args(new_zonotope_w, source_zonotope=other, clone=False)  # We use other because it has the correct error ranges

    def pick_error_terms_to_remove(self, weights: torch.Tensor, final_number_terms: int) -> Tuple[torch.Tensor, torch.Tensor]:
        G = weights
        num_values, number_error_terms = G.size(0), G.size(1)

        # compute metric of generators
        # vecnorm = norm of each column = torch.norm(dim=0)
        # h = torch.norm(G, p=1, dim=0) - torch.norm(G, float("inf"), dim=0)
        h = G.abs().sum(0)

        # number of generators that are not reduced
        # Before -> Original number of terms = N = N_Unreduced + N_reduced
        # After  -> N_Unreduced + N_Words * N_Embedding = final_number_terms
        #
        # =====>  N_unreduced = final_number_terms - N_Words * N_Embedding
        # =====>  N_reduced   = N - N_unreduced

        number_unreduced = final_number_terms - self.num_words * self.word_embedding_size  # math.floor(num_values * (final_number_terms - 1))
        # number of generators that are reduced
        number_reduced = number_error_terms - number_unreduced

        return self.pick_indices(h, number_reduced)

    def pick_indices(self, coefficients: torch.Tensor, number_reduced: int):
        assert number_reduced <= coefficients.nelement(), "pick_indices: number reduced higher than the number of coefficients"
        # pick generators with smallest coefficients values to be reduced
        _, indices_to_reduce = torch.topk(coefficients, k=number_reduced, largest=False, sorted=False)

        # unreduced generators
        indices_not_to_reduce = torch.ones(coefficients.size(0), dtype=torch.bool)
        indices_not_to_reduce[indices_to_reduce] = False

        return indices_not_to_reduce, indices_to_reduce

    def reduce_num_error_terms_box(self, max_num_error_terms: int):
        # Initial = N = N_input_to_keep + N_others_unreduced + N_others_reduced
        # Final   = F = N_input_to_keep + N_others_unreduced + N_words * N_embedding
        #
        # ======> N_others_unreduced = F - N_input_to_keep - N_words * N_embedding
        # ======> N_others_reduced   = N - N_input_to_keep - N_others_unreduced
        #
        # with constraints N_others_unreduced >= 0 and N_others_reduced > 0
        # N_others_unreduced >= 0 is equivalent to F >= N_input_to_keep + N_words * N_embedding, which makess ense
        # N_others_reduced > 0    is equivalent to N - N_input_to_keep - F + N_input_to_keep + N_words * N_embedding > 0
        #                                          N > F - N_words * N_embedding
        # which tries to ensure the reducing process will actually be useful for anything
        #
        # N = 128 + UNREDUCED + REDUCED = 5000
        # F = 128 + UNREDUCED + 2560 = 3000
        #   -> UNREDUCED = 3000 - 2560 - 128 = 312
        #   -> REDUCED   = 5000 - 128  - 312 = 4560
        if self.num_error_terms <= max_num_error_terms:
            return self

        n_input_to_keep = self.num_input_error_terms_special_norm
        N_others_unreduced = max_num_error_terms  - n_input_to_keep - self.num_words * self.word_embedding_size
        N_others_reduced   = self.num_error_terms - n_input_to_keep - N_others_unreduced

        if N_others_reduced > 0 and N_others_unreduced < 0:
            print(colored(f'Warning: max num error terms {max_num_error_terms} too low to do a reduction', 'red'))
            print(colored(f'Warning: reducing instead to min num {self.num_words*self.word_embedding_size + 1} noise symbols', 'red'))
            return self.reduce_num_error_terms_box(max_num_error_terms=self.num_words*self.word_embedding_size + n_input_to_keep + 1)

        if N_others_reduced > 0 and N_others_unreduced >= 0:
            return self.remove_error_terms_box(num_terms_to_reduce=N_others_reduced, num_original_error_terms_to_keep=n_input_to_keep)
        else:
            return self

    def remove_error_terms_box(self, num_terms_to_reduce: int, num_original_error_terms_to_keep: int) -> "Zonotope":
        if self.zonotope_w.ndim == 4:
            num_attention_heads = self.zonotope_w.shape[0]
            return self.remove_attention_heads_dim() \
                .remove_error_terms_box(num_terms_to_reduce, num_original_error_terms_to_keep) \
                .add_attention_heads_dim(num_attention_heads, clone=False)

        candidate_error_terms = self.zonotope_w[1 + num_original_error_terms_to_keep:]
        abs_sum_error_coeffs = candidate_error_terms.abs().sum(dim=[1, 2])
        indices_not_to_reduce, indices_to_reduce = self.pick_indices(abs_sum_error_coeffs, num_terms_to_reduce)

        preserved_error_terms = candidate_error_terms[indices_not_to_reduce]

        # Aggregated error terms (1 new error term per value; doing only one new term globally would be incorrect)
        aggregated_errors_coeffs = candidate_error_terms[indices_to_reduce].abs().sum(dim=0)
        aggregated_errors_coeffs_good_shape = torch.zeros(
            self.num_words * self.word_embedding_size, self.num_words, self.word_embedding_size, device=self.device
        )

        # Check if new version behaves identically
        errors, rows, cols = [], [], []
        pos = 0
        for word_index in range(self.num_words):
            for embedding_index in range(self.word_embedding_size):
                errors.append(pos)
                rows.append(word_index)
                cols.append(embedding_index)
                pos += 1

        # still inplace
        aggregated_errors_coeffs_good_shape[errors, rows, cols] = aggregated_errors_coeffs[rows, cols]

        # pos = 0
        # for word_index in range(self.num_words):
        #     for embedding_index in range(self.word_embedding_size):
        #         aggregated_errors_coeffs_good_shape[pos, word_index, embedding_index] = aggregated_errors_coeffs[word_index, embedding_index]
        #         pos += 1

        new_zonotope_w = torch.cat([
            self.zonotope_w[0:1 + num_original_error_terms_to_keep],
            preserved_error_terms,
            aggregated_errors_coeffs_good_shape
        ])

        # print("Reduce error terms... Original shape: %s       New shape: %s" % (self.zonotope_w.shape, new_zonotope_w.shape))
        return make_zonotope_new_weights_same_args(new_zonotope_w, source_zonotope=self, clone=False)

    def add_attention_heads_dim(self, A: int, clone=True) -> "Zonotope":
        """
            Original shape                      (1 + n_error_terms, length, embedding_size)
            New shape      (num_attention_heads, 1 + n_error_terms, length, embedding_size / num_attention_heads)
        """
        assert self.zonotope_w.ndim == 3, "Zonotope_w doesn't have 3 dims (it has %d) so we can't add a new dim" % self.zonotope_w.ndim
        error_dim, length, embedding_times_attention_size = self.zonotope_w.shape
        new_zonotope_w = self.zonotope_w.reshape(error_dim, length, A, embedding_times_attention_size // A).permute(2, 0, 1, 3)
        return make_zonotope_new_weights_same_args(new_zonotope_w, source_zonotope=self, clone=clone)

    def remove_attention_heads_dim(self, clone=True) -> "Zonotope":
        """
            Original shape  (num_attention_heads, 1 + n_error_terms, length, embedding_size)
            New shape                            (1 + n_error_terms, length, num_attention_heads * embedding_size)
        """
        assert self.zonotope_w.ndim == 4, "Zonotope_w doesn't have 4 dims (it has %d) so we can't remove a new dim" % self.zonotope_w.ndim

        torch.cuda.empty_cache()

        A, error_dim, length, embedding_size = self.zonotope_w.shape
        new_zonotope_w = self.zonotope_w.permute(1, 2, 0, 3).reshape(error_dim, length, A * embedding_size)
        return make_zonotope_new_weights_same_args(new_zonotope_w, source_zonotope=self, clone=clone)

    def print(self, message: str) -> None:
        """ Print a message and then statistics about the bounds and norms of the error weights"""
        print("Zonotope Stats: ", message)
        l, u = self.concretize()
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(l)).item(), torch.mean(torch.abs(u)).item()))
        print("diff %.5f %.5f %.5f" % (torch.min(u - l).item(), torch.max(u - l).item(), torch.mean(u - l).item()))
        print("error weights norm", torch.mean(torch.norm(self.zonotope_w[1:], dim=-2)))
        print("min", torch.min(l))
        print("max", torch.max(u))
        print()

    def set_error_term_ranges(self, error_term_low: torch.Tensor, error_term_high: torch.Tensor) -> None:
        num_error_terms_whose_range_can_be_set = self.num_error_terms - self.num_input_error_terms_special_norm

        assert error_term_low.nelement() == num_error_terms_whose_range_can_be_set, "set_error_term_ranges: invalid number of low error ranges"
        assert error_term_high.nelement() == num_error_terms_whose_range_can_be_set, "set_error_term_ranges: invalid number of high error ranges"

        self.error_term_range_low = error_term_low
        self.error_term_range_high = error_term_high

    def sample_point(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.error_term_range_low is None and self.error_term_range_high is None, "Not supported"
        assert self.num_input_error_terms_special_norm == 0 or self.num_input_error_terms_special_norm == self.num_error_terms, "Sampling with multi-norm isn't supported yet"
        error_terms_values = sample_vector_with_p_norm_below_or_equal_to_1(number_values=self.num_error_terms, p=self.p, binary_value=random.random() < 0.2)
        return self.compute_val_given_error_terms(error_terms_values), error_terms_values

    def compute_val_given_error_terms(self, error_term_values: torch.Tensor) -> torch.Tensor:
        """ Given the values of the error terms e, concretize the zonotope into a specific value """
        assert self.error_term_range_low is None and self.error_term_range_high is None, "Not supported"
        assert self.num_error_terms == error_term_values.nelement(), "Number of error terms doesn't match"

        if self.zonotope_w.ndim == 3:
            # Shape: (1 + num_error_terms, length, embedding)
            center = self.zonotope_w[0]
            weighted_error_term_vals = (self.zonotope_w[1:].permute(1, 2, 0) * error_term_values).permute(2, 0, 1).sum(dim=0)
        else:
            # Shape: (A, 1 + num_error_terms, length, embedding)
            center = self.zonotope_w[:, 0]
            weighted_error_term_vals = (self.zonotope_w[:, 1:].permute(0, 2, 3, 1) * error_term_values).permute(0, 3, 1, 2).sum(dim=1)

        result = center + weighted_error_term_vals
        if self.zonotope_w.ndim <= 3: assert result.shape == self.zonotope_w[0].shape, "compute_val_given_error_terms: result does not have the right shape"
        if self.zonotope_w.ndim == 4: assert result.shape == self.zonotope_w[:, 0].shape, "compute_val_given_error_terms: result does not have the right shape"

        return result

    def get_error_terms(self) -> torch.Tensor:
        if self.zonotope_w.ndim == 4:
            return self.zonotope_w[:, 1:]
        else:
            return self.zonotope_w[1:]

    def get_input_special_errors_and_new_error_terms(self, error_terms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if error_terms.ndim == 4:
            return error_terms[:, :self.num_input_error_terms_special_norm], error_terms[:, self.num_input_error_terms_special_norm:]
        else:
            return error_terms[:self.num_input_error_terms_special_norm], error_terms[self.num_input_error_terms_special_norm:]

    def concretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes the concretized scalar upper and lower bound values for each perturbed word
        and them sums the results, obtaining a total concretize lower and upper bound """
        if self.zonotope_w.ndim <= 3:  # (1 + n_error_terms, length, embedding_size)
            center, error_terms = self.zonotope_w[0], self.zonotope_w[1:]
        elif self.zonotope_w.ndim == 4:  # (num_attention_heads, 1 + n_error_terms, length, embedding_size)
            center, error_terms = self.zonotope_w[:, 0], self.zonotope_w[:, 1:]
        else:
            raise Exception("Zonotope W should have 3 or 4 dimensions")

        if self.error_term_range_low is None:
            assert self.error_term_range_high is None, "error_term_range_low is None but error_term_range_high is not None"
            sum_dim = 0 if self.zonotope_w.ndim <= 3 else 1  # if there are 4 dims, the error terms are in the dim 1 and not 0
            input_special_norm_errors, infinity_errors = self.get_input_special_errors_and_new_error_terms(error_terms)

            lower = center - get_norm(infinity_errors, p=DUAL_INFINITY, dim=sum_dim)
            upper = center + get_norm(infinity_errors, p=DUAL_INFINITY, dim=sum_dim)

            if self.p == 1 or self.p == 2:
                width = self.get_width_for_special_terms(input_special_norm_errors)
                lower = lower - width
                upper = upper + width
        else:
            assert self.error_term_range_high is not None, "error_term_range_low is not None but error_term_range_high is None"

            input_special_norm_errors, infinity_errors = self.get_input_special_errors_and_new_error_terms(error_terms)

            if self.zonotope_w.ndim == 4:
                A, length = error_terms.size(0), error_terms.size(2)

                # ORIGINAL ERRORS (A, num_original_error_terms, length, width) -> (num_original_error_terms, A * length, width)
                input_special_norm_errors = input_special_norm_errors.permute(1, 0, 2, 3).reshape(self.num_input_error_terms_special_norm, A * length, self.word_embedding_size)

                # NEW ERRORS (A, num_new_error_terms, length, width) -> (num_new_error_terms, A * length, width)
                num_new_error_terms = infinity_errors.size(1)
                infinity_errors = infinity_errors.permute(1, 0, 2, 3).reshape(num_new_error_terms, -1, self.word_embedding_size)

                # CENTER (A, length, width) -> (A * length, width)
                center = center.reshape(-1, self.word_embedding_size)

            # COMPUTE MAXIMUM/MIN VALUE OF THE NEW ERROR TERMS (INF NORM) AND TAKE INTO ACCOUNT THE ERROR RANGES
            # (num_error_terms, length, width) -> (length, width, num_error_terms) -> (num_error_terms, length, width)
            extreme_1 = (infinity_errors.permute(1, 2, 0) * self.error_term_range_low).permute(2, 0, 1)
            extreme_2 = (infinity_errors.permute(1, 2, 0) * self.error_term_range_high).permute(2, 0, 1)
            new_errors_min, new_errors_max = torch.min(extreme_1, extreme_2), torch.max(extreme_1, extreme_2)

            lower = center + new_errors_min.sum(dim=0)
            upper = center + new_errors_max.sum(dim=0)
            if input_special_norm_errors.nelement() > 0:
                width = self.get_width_for_special_terms(input_special_norm_errors)
                lower = lower - width
                upper = upper + width

            if self.zonotope_w.ndim == 4:
                # (A * length, width) -> (A, length, width)
                lower = lower.reshape(-1, self.num_words, self.word_embedding_size)
                upper = upper.reshape(-1, self.num_words, self.word_embedding_size)

        if self.zonotope_w.ndim <= 3: assert upper.shape == self.zonotope_w[0].shape, "Concretize: upper does not have the right shape"
        if self.zonotope_w.ndim == 4: assert upper.shape == self.zonotope_w[:, 0].shape, "Concretize: upper does not have the right shape"

        return lower, upper

    def get_width_for_special_terms(self, input_special_norm_errors: torch.Tensor, q: float = None) -> torch.Tensor:
        if q is None:
            q = self.dual_p

        error_dim = 0 if input_special_norm_errors.ndim <= 3 else 1
        if self.args.concretize_special_norm_error_together:
            width = get_norm(input_special_norm_errors, p=q, dim=error_dim)
        elif self.args.perturbed_words == 1:
            width = get_norm(input_special_norm_errors, p=q, dim=error_dim)
        else:
            N = self.args.perturbed_words
            E = self.num_input_error_terms_special_norm // self.args.perturbed_words

            if input_special_norm_errors.ndim <= 3:
                _, X, Y = input_special_norm_errors.shape
                special_terms_good_shape = input_special_norm_errors.reshape(N, E, X, Y)  # (N * E, X, Y) -> (N, E, X, Y)
                norm_per_word = get_norm(special_terms_good_shape, p=q, dim=1)  # (N, X, Y)
                width = norm_per_word.abs().sum(dim=0)  # (X, Y)
            else:
                A, _, X, Y = input_special_norm_errors.shape
                special_terms_good_shape = input_special_norm_errors.reshape(A, N, E, X, Y)  # (A, N * E, X, Y) -> (A, N, E, X, Y)
                norm_per_word = get_norm(special_terms_good_shape, p=q, dim=2)  # (A, N, X, Y)
                width = norm_per_word.abs().sum(dim=1)  # (A, X, Y)
        return width

    def clone(self) -> "Zonotope":
        return Zonotope(
            self.args, self.p, self.eps, self.perturbed_word_index,
            zonotope_w=self.zonotope_w, error_term_range_low=self.error_term_range_low, error_term_range_high=self.error_term_range_high,
            clone=True
        )

    def t(self) -> "Zonotope":
        """ Transposes the length and the dim_out dimensions of the bounds """
        assert self.zonotope_w.ndim in [3, 4], "Zonotope weights must have 3 or 4 dimensions"

        if self.zonotope_w.ndim == 3:
            new_zonotope_w = self.zonotope_w.transpose(1, 2)
        else:
            new_zonotope_w = self.zonotope_w.transpose(2, 3)

        return make_zonotope_new_weights_same_args(new_zonotope_w, source_zonotope=self)

    def add(self, delta: Union["Zonotope", float, torch.Tensor]) -> "Zonotope":
        if type(delta) == Zonotope:
            return make_zonotope_new_weights_same_args(new_weights=self.zonotope_w + delta.zonotope_w, source_zonotope=self, clone=False)
        else:  # Add constant values
            new_weights = self.zonotope_w.clone()
            new_weights[0] = new_weights[0] + delta  # INPLACE
            return make_zonotope_new_weights_same_args(new_weights, source_zonotope=self, clone=False)

    def matmul(self, W) -> "Zonotope":
        if type(W) == Zonotope:
            raise NotImplementedError()
        elif len(W.shape) == 2:
            output_zonotope_w = F.linear(self.zonotope_w, W, None)  # no bias for the moment
            return make_zonotope_new_weights_same_args(output_zonotope_w, source_zonotope=self, clone=False)
        else:
            assert False, "Matmul with weight matrix with %d dimensions not supported (exact shape = %s)" % (W.dim(), W.shape)

    def multiply(self, other: Union[float, "Zonotope", torch.Tensor]) -> "Zonotope":
        """ Multiply by either a float, Bounds or a matrix. Takes into account the signs to
        ensure the compute lower and upper bounds weights and biases are correct. """
        if isinstance(other, float):
            return make_zonotope_new_weights_same_args(new_weights=self.zonotope_w * other, source_zonotope=self, clone=False)
        elif isinstance(other, Zonotope):
            assert self.zonotope_w.ndim == 3, "Multiply: self should have 3 dims"
            assert other.zonotope_w.ndim == 3, "Multiply: other should have 3 dims"
            assert self.zonotope_w.shape[1:] == other.zonotope_w.shape[1:], "Multiply: Zonotope should have the same shape"

            return self.do_multiply(other)
        else:  # Matrix
            # Important: in this case, it's NOT a matrix multiplication, it's an ELEMENT-WISE multiplication
            return make_zonotope_new_weights_same_args(new_weights=self.zonotope_w * other, source_zonotope=self, clone=False)

    def dot_product(self, other, *args, **kwargs) -> "Zonotope":
        if self.args.zonotope_slow:
            return self.dot_product_precise(other, *args, **kwargs)
        else:
            return self.dot_product_fast(other, *args, **kwargs)

    def dot_product_precise(self, other, zonotopes_can_have_different_number_noise_symbols=True, **kwargs) -> "Zonotope":
        # Input:
        #    A: (1 + n_error_terms, nA)
        #    B: (1 + n_error_terms, nB)
        # Output:
        #    C: (1 + n_error_terms, nA, nB)
        #    C[i, j] = A[i] * B[j] = A[i]_0 * B[j]_0  +
        #                          + sum_k [ ( A[i]_0 * B[j]_k  +    B[j]_0 * A[i]_k )  * e_k ]
        #                          + sum_k_m [ A[i]_k * B[j]_m                          * e_k * e_m)
        if not zonotopes_can_have_different_number_noise_symbols:
            if self.num_error_terms < other.num_error_terms:
                self = self.expand_error_terms_to_match_zonotope(other)
                print("Dot product: Increasing number of error terms in self")
            elif other.num_error_terms < self.num_error_terms:
                other = other.expand_error_terms_to_match_zonotope(self)
                print("Dot product: Increasing number of error terms in other")

        assert self.zonotope_w.ndim == 4, "self should have 4 dims"
        assert other.zonotope_w.ndim == 4, "other should have 4 dims"

        num_attention_heads = self.zonotope_w.shape[0]
        nA = self.num_words
        nB = other.num_words

        n_new_error_terms = nA * nB * num_attention_heads
        has_error_terms = torch.ones(nA, nB, dtype=torch.bool, device=self.args.device)

        self_zonotope_w = self.zonotope_w
        other_zonotope_w = other.zonotope_w

        min_error_terms = min(self.num_error_terms, other.num_error_terms)
        max_error_terms = max(self.num_error_terms, other.num_error_terms)

        if zonotopes_can_have_different_number_noise_symbols:
            C = torch.zeros(num_attention_heads, 1 + max_error_terms + n_new_error_terms, nA, nB, device=self.args.device)
        else:
            C = torch.zeros(num_attention_heads, self.zonotope_w.shape[1] + n_new_error_terms, nA, nB, device=self.args.device)

        for attention_head in range(num_attention_heads):
            half_term = torch.zeros_like(C[attention_head, 0])  # (nA, nB)
            big_term = torch.zeros_like(C[attention_head, 0])  # (nA, nB)

            # SECTION: NEW NORMAL BIAS
            # First row of C[A, 0] = first vector of self x all other vector of other
            # First col of C[A, 0] =   all vector of self x first vector of other
            if zonotopes_can_have_different_number_noise_symbols:
                C[attention_head, 0] = self_zonotope_w[attention_head, 0] @ other_zonotope_w[attention_head, 0].t()
            else:
                C[attention_head, 0] = self.zonotope_w[attention_head, 0] @ other.zonotope_w[attention_head, 0].t()

            # SECTION: MIXED ERROR TERMS e_i * e_i (update bias + error coefficient of new error term)
            # For the e_i * e_i error terms that appear when doing the multiplication, we compute the value
            # of the center of the inverval (which we add to the bias of the new zonotope) and the error coefficients
            # of the new error coefficient (we do abs value)

            # ORIGINAL
            if zonotopes_can_have_different_number_noise_symbols:
                values_i_i = torch.bmm(
                    self_zonotope_w[attention_head, 1:1 + min_error_terms],  # (n_error_termsMin, nA, embedding_size)
                    other_zonotope_w[attention_head, 1:1 + min_error_terms].permute(0, 2, 1)  # (n_error_termsMin, embedding_size, nB)
                )  # Output shape: (embedding_size, nA, nB)
            else:
                values_i_i = torch.bmm(
                    self.zonotope_w[attention_head, 1:],  # (n_error_terms, nA, embedding_size)
                    other.zonotope_w[attention_head, 1:].permute(0, 2, 1)  # (n_error_terms, embedding_size, nB)
                )  # Output shape: (embedding_size, nA, nB)

            # The value for e_i * e_i will depend on all the term of the vector, which is computed in the BMM above
            # For the error coefficients, I need to do abs().sum() instead of the reverse, because the formula is
            # sum_a[ abs(sum_i v_i(a) * w_i(a)) ]
            C[attention_head, 0] += 0.5 * values_i_i.sum(dim=0)  # inplace
            half_term += 0.5 * values_i_i.abs().sum(dim=0)  # inplace

            # SECTION: EXISTING ERROR TERMS - Compute the new coefficients for the existing error terms
            if zonotopes_can_have_different_number_noise_symbols:
                els_A = self_zonotope_w[attention_head, 1:]  # (n_errorA, nA, embedding_size)
                els_B = other_zonotope_w[attention_head, 1:]  # (n_errorB, nB, embedding_size)

                els_A_0 = self_zonotope_w[attention_head, 0].repeat(els_B.size(0), 1, 1)  # (n_errorB, nA, embedding_size)
                els_B_0 = other_zonotope_w[attention_head, 0].repeat(els_A.size(0), 1, 1)  # (n_errorA, nB, embedding_size)

                sum_errors_1 = torch.bmm(els_A_0, els_B.transpose(1, 2))  # (n_errorB, nA, embedding_size)  bmm  (n_errorB, embedding_size, nB)
                sum_errors_2 = torch.bmm(els_A, els_B_0.transpose(1, 2))  # (n_errorA, nA, embedding_size)  bmm  (n_errorA, embedding_size, nB)

                sum_errors = sum_tensor_with_different_dim0_(sum_errors_1, sum_errors_2)
                C[attention_head, 1:1 + self.num_error_terms] = sum_errors  # (max_num_error_terms, nA, nB)
            else:
                els_A = self.zonotope_w[attention_head, 1:]  # (n_error, nA, embedding_size)
                els_A_0 = self.zonotope_w[attention_head, 0].repeat(self.num_error_terms, 1, 1)  # (n_error, nA, embedding_size)

                els_B = other.zonotope_w[attention_head, 1:]  # (n_error, nB, embedding_size)
                els_B_0 = other.zonotope_w[attention_head, 0].repeat(self.num_error_terms, 1, 1)  # (n_error, nB, embedding_size)

                sum_errors_1 = torch.bmm(els_A_0, els_B.transpose(1, 2))  # (n_error, nA, embedding_size)  bmm  (n_error, embedding_size, nB)
                sum_errors_2 = torch.bmm(els_A, els_B_0.transpose(1, 2))  # (n_error, nA, embedding_size)  bmm  (n_error, embedding_size, nB)
                C[attention_head, 1:1 + self.num_error_terms] = sum_errors_1 + sum_errors_2  # (num_error_terms, nA, nB)

            # SECTION: MIXED ERROR TERMS e_i * e_j (compute the error coefficient of new error terms)
            if zonotopes_can_have_different_number_noise_symbols:
                for a in range(1, 1 + min_error_terms):
                    # there's no point in taking a in the range [min_errors + 1, last_error]
                    # because then b > min_error + 1
                    # and so either
                    #    1) errors_other_b and errors_other_a is filled with 0's OR
                    #    1) errors_self_b  and errors_self_a  is filled with 0's
                    # then, in the matrix multiplications between self and other would be 0
                    # and so this is wasted computation
                    b_min = a + 1

                    # shape: (other_error_terms_after_a, nB, embedding_size)
                    errors_other_b = other_zonotope_w[attention_head, b_min:, :, :]
                    # shape: (self_error_terms_after_a, nA, embedding_size)
                    errors_self_b = self_zonotope_w[attention_head, b_min:, :, :]

                    # num_repeats = errors_other_b.shape[0]

                    # shape: (num_repeats, nA, embedding_size)
                    errors_self_a = self_zonotope_w[attention_head, a, :, :].repeat(errors_other_b.size(0), 1, 1)
                    # shape: (num_repeats, nB, embedding_size)
                    errors_other_a = other_zonotope_w[attention_head, a, :, :].repeat(errors_self_b.size(0), 1, 1)

                    # shape: (size1, nA, nB)
                    mixed_values1 = torch.bmm(errors_self_a, errors_other_b.transpose(1, 2))
                    # shape: (size2, nA, nB)
                    mixed_values2 = torch.bmm(errors_self_b, errors_other_a.transpose(1, 2))

                    mixed_values = sum_tensor_with_different_dim0_(mixed_values1, mixed_values2)
                    big_term += mixed_values.abs().sum(0)  # inplace, does not matter
            else:
                for a in range(1, 1 + self.num_error_terms):
                    b_min = a + 1

                    errors_other_b = other.zonotope_w[attention_head, b_min:, :, :]
                    errors_self_b = self.zonotope_w[attention_head, b_min:, :, :]

                    num_repeats = errors_other_b.shape[0]
                    errors_self_a = self.zonotope_w[attention_head, a, :, :].repeat(num_repeats, 1, 1)
                    errors_other_a = other.zonotope_w[attention_head, a, :, :].repeat(num_repeats, 1, 1)

                    mixed_values1 = torch.bmm(errors_self_a, errors_other_b.transpose(1, 2))
                    mixed_values2 = torch.bmm(errors_self_b, errors_other_a.transpose(1, 2))
                    mixed_values = mixed_values1 + mixed_values2

                    big_term += mixed_values.abs().sum(0)  # inplace, does not matter

            new_weights = half_term + big_term

            start = 1 + self.num_error_terms + attention_head * nA * nB
            indices = torch.arange(start, start + nA * nB)
            C[attention_head, indices, has_error_terms] = new_weights[has_error_terms]

        return make_zonotope_new_weights_same_args(C, source_zonotope=self, clone=False)

    def dot_product_fast(self, other: "Zonotope", **kwargs) -> "Zonotope":
        # Details for the dot product between two vectors. Here there are N1 vectors in self and N2 vectors in other, so
        # the code generalizes this below
        #
        # Input:
        #    A: (1 + n_error_terms, nA)
        #    B: (1 + n_error_terms, nB)
        # Output:
        #    C: (1 + n_error_terms, nA, nB)
        #    C[i, j] = A[i] * B[j] = A[i]_0 * B[j]_0  +
        #                          + sum_k [ ( A[i]_0 * B[j]_k  +    B[j]_0 * A[i]_k )  * e_k ]
        #                          + sum_k_m [ A[i]_k * B[j]_m                          * e_k * e_m)
        if self.args.use_dot_product_variant3:
            A_inf_original_all = self.zonotope_w[:, 1 + self.num_input_error_terms_special_norm:]
            B_inf_original_all = other.zonotope_w[:, 1 + other.num_input_error_terms_special_norm:]

        if self.num_error_terms < other.num_error_terms:
            self = self.expand_error_terms_to_match_zonotope(other)
            # print("Dot product: Increasing number of error terms in self")
        elif other.num_error_terms < self.num_error_terms:
            other = other.expand_error_terms_to_match_zonotope(self)
            # print("Dot product: Increasing number of error terms in other")

        assert self.zonotope_w.ndim == 4, "self should have 4 dims"
        assert other.zonotope_w.ndim == 4, "other should have 4 dims"

        num_attention_heads = self.zonotope_w.shape[0]
        nA = self.num_words
        nB = other.num_words

        n_new_error_terms = num_attention_heads * nA * nB
        has_error_terms = torch.ones(nA, nB, dtype=torch.bool, device=self.args.device)
        C = torch.zeros(num_attention_heads, self.zonotope_w.shape[1] + n_new_error_terms, nA, nB, device=self.args.device)

        def multiply_matrices(left: torch.Tensor, right_with_special_norm: torch.Tensor, p_right: float) -> torch.Tensor:
            dual_norm_right = dual_norm(p_right)

            E1, N1, v1 = left.shape
            E2, N2, v2 = right_with_special_norm.shape
            assert v1 == v2, "bad v"
            E = E1

            A = self.get_width_for_special_terms(right_with_special_norm, q=dual_norm_right)  # Shape: (N2, v)

            # for the matmul, we need (N1, 1, E, v) x (N2, v, 1) -> (N1, N2, E, 1)
            left_good_shape = left.transpose(0, 1).unsqueeze(1)  # left: (E, N1, v) -> (N1, 1, E, v)
            A_good_shape = A.unsqueeze(-1).unsqueeze(0)  # A:    (N2, v)    -> (1, N2, v, 1)

            result = torch.matmul(left_good_shape.abs(), A_good_shape)  # We accumulate the intervals
            assert result.shape == torch.Size([N1, N2, E, 1])

            # We want to update the shape of result from (N1, N2, E, 1) to (E, N1, N2)
            result_good_shape = result.squeeze(-1).permute(2, 0, 1)

            assert (result_good_shape >= 0).all(), "result has some negative numbers"

            return result_good_shape

        def multiply_matrices_p_p(left: torch.Tensor, right: torch.Tensor, p: float) -> torch.Tensor:
            q = dual_norm(p)
            E1, N1, v1 = left.shape
            E2, N2, v2 = right.shape
            assert v1 == v2, "bad v"
            E = E1

            A = self.get_width_for_special_terms(right, q=q)  # Shape: (N2, v)
            # A = get_norm(right, p=dual_norm_right, dim=0)

            # for the matmul, we need (N1, 1, E, v) x (N2, v, 1) -> (N1, N2, E, 1)
            left_good_shape = left.transpose(0, 1).unsqueeze(1)  # left: (E, N1, v) -> (N1, 1, E, v)
            A_good_shape = A.unsqueeze(-1).unsqueeze(0)  # A:    (N2, v)    -> (1, N2, v, 1)

            result = torch.matmul(left_good_shape.abs(), A_good_shape)  # We accumulate the intervals
            assert result.shape == torch.Size([N1, N2, E, 1])

            # We want to update the shape of result from (N1, N2, E, 1) to (E, N1, N2)
            result_good_shape = result.squeeze(-1).permute(2, 0, 1)

            assert (result_good_shape >= 0).all(), "result has some negative numbers"

            return result_good_shape

        for attention_head in range(num_attention_heads):
            #### SECTION: A0 * B0
            # First row of C[A, 0] = first vector of self x all other vector of other
            # First col of C[A, 0] =   all vector of self x first vector of other
            C[attention_head, 0] = self.zonotope_w[attention_head, 0] @ other.zonotope_w[attention_head, 0].t()

            #### SECTION: A0 * B_errors   +     B0 * A_errors
            els_A = self.zonotope_w[attention_head, 1:]  # (n_error, nA, embedding_size)
            els_A_0 = self.zonotope_w[attention_head, 0].repeat(self.num_error_terms, 1, 1)  # (n_error, nA, embedding_size)

            els_B = other.zonotope_w[attention_head, 1:]  # (n_error, nB, embedding_size)
            els_B_0 = other.zonotope_w[attention_head, 0].repeat(self.num_error_terms, 1, 1)  # (n_error, nB, embedding_size)

            sum_errors_1 = torch.bmm(els_A_0, els_B.transpose(1, 2))  # (n_error, nA, embedding_size)  bmm  (n_error, embedding_size, nB)
            sum_errors_2 = torch.bmm(els_A, els_B_0.transpose(1, 2))  # (n_error, nA, embedding_size)  bmm  (n_error, embedding_size, nB)
            C[attention_head, 1:1+self.num_error_terms] = sum_errors_1 + sum_errors_2  # (num_error_terms, nA, nB)

            #### SECTION: A_errors * B_errors
            #
            # We will decompose the errors into two groups: E_p and E_inf
            # where E_p contains the error terms bounded by the 1-norm or 2-norm and E_inf contains the error terms bounded by the inf-norm
            #
            # Therefore,
            #      A_errors = A_p + A_inf
            #      B_errors = B_p + B_inf
            # and
            #      A_errors * B_errors = (A_p + A_inf) * (B_p + B_inf)
            #                          = A_p * B_p  + A_p * B_inf + B_p * A_inf +  A_inf * B_inf
            A_p, A_inf = self.get_input_special_errors_and_new_error_terms(self.zonotope_w[attention_head, 1:])  # (E_p, N1, v)
            B_p, B_inf = other.get_input_special_errors_and_new_error_terms(other.zonotope_w[attention_head, 1:]) # (E_inf, N2, v)

            assert A_p.size(0) == B_p.size(0), "A and B don't have the same number of non-inf error terms"
            assert A_inf.size(0) == B_inf.size(0), "A and B don't have the same number of inf error terms"
            assert A_p.size(2) == B_p.size(2), "A and B don't have the same number of elements in the vectors - p"
            assert A_inf.size(2) == B_inf.size(2), "A and B don't have the same number of elements in the vectors - inf"
            assert self.p == other.p, "Self and other don't have the same norm"

            new_weights = torch.zeros(nA, nB, device=self.device)

            ### A_p * B_p
            # TODO(multi noise symbol groups): adapt
            if A_p.size(0) > 0 and B_p.size(0) > 0:
                assert self.p == 1 or self.p == 2, "there are 1-norm or 2-norm bound error terms but the norm is INF"
                error_1_coeffs = multiply_matrices_p_p(A_p, B_p, self.p)  # (e_errors_1, NA, NB)
                new_weights += self.get_width_for_special_terms(error_1_coeffs, q=dual_norm(self.p))  # inplace
            else:
                assert self.p > 10, "there are no 1-norm or 2-norm bound error terms but the norm is 1 or 2"

            def do_mixed_multiplication(inf_errors: torch.Tensor, p_errors: torch.Tensor, p: float, use_other_dot_product_ordering: bool, inf_is_self: bool):
                if not use_other_dot_product_ordering:  # v2 is better
                    errors_inf_coeffs1_v2 = multiply_matrices(
                        p_errors, right_with_special_norm=inf_errors, p_right=INFINITY
                    )  # (e_error_1, NB, NA)

                    result_v2 = self.get_width_for_special_terms(errors_inf_coeffs1_v2, q=dual_norm(p))
                    # result_v2 = get_norm(errors_inf_coeffs1_v2, p=dual_norm(p), dim=0)

                    if inf_is_self:
                        result_v2 = result_v2.t()
                    assert (result_v2 >= 0).all(), "v2"
                    return result_v2
                else:  # v1 is worse
                    assert self.args.num_perturbed_words == 1, "Only 1 word perturbations are supported in this mode"
                    errors_inf_coeffs1_v1 = multiply_matrices(inf_errors, right_with_special_norm=p_errors, p_right=p)  # (e_errors_inf, N1, N2)
                    result_v1 = get_norm(errors_inf_coeffs1_v1, p=dual_norm(INFINITY), dim=0)
                    if not inf_is_self:
                        result_v1 = result_v1.t()
                    assert (result_v1 >= 0).all(), "v1"
                    return result_v1

            ### A_p * B_inf + B_p * A_inf
            if A_p.size(0) > 0 and B_p.size(0) > 0 and A_inf.size(0) > 0 and B_inf.size(0):
                new_weights += do_mixed_multiplication(A_inf, B_p, self.p, self.args.use_other_dot_product_ordering, inf_is_self=True)  # inplace
                new_weights += do_mixed_multiplication(B_inf, A_p, self.p, self.args.use_other_dot_product_ordering, inf_is_self=False)  # inplace

            ### A_inf * B_inf
            if A_inf.size(0) > 0 and B_inf.size(0):
                if self.args.use_dot_product_variant3:
                    center_change, weights_change = self.dot_product_precise_inf_inf_multiplicationv1(
                        A_inf_original_all[attention_head], B_inf_original_all[attention_head]
                    )
                    # center_change, weights_change = self.dot_product_inf_inf_sparse_more_basic(
                    #     A_inf, B_inf
                    # )
                    new_weights += weights_change  # inplace, does not matter
                    C[attention_head, 0] += center_change  # inplace, does not matter
                else:
                    errors_inf_coeffs3 = multiply_matrices(A_inf, B_inf, p_right=INFINITY)  # (e_errors_inf, NA, NB)
                    new_weights += get_norm(errors_inf_coeffs3, p=dual_norm(INFINITY), dim=0)  # inplace

            start_index = 1 + self.num_error_terms + attention_head * nA * nB
            indices = torch.arange(start_index, start_index + nA * nB)
            C[attention_head, indices, has_error_terms] = new_weights[has_error_terms]

        return make_zonotope_new_weights_same_args(C, source_zonotope=self, clone=False)

    def dot_product_fast_batch(self, other: "Zonotope", **kwargs) -> "Zonotope":
        if self.args.use_dot_product_variant3:
            assert "use_dot_product_variant3 not supported yet for Zonotope.dot_product_fast_batch()"

        assert self.args.perturbed_words == 1, "Multiple perturbed words not supported yet for Zonotope.dot_product_fast_batch()"

        if self.num_error_terms < other.num_error_terms:
            self = self.expand_error_terms_to_match_zonotope(other)
        elif other.num_error_terms < self.num_error_terms:
            other = other.expand_error_terms_to_match_zonotope(self)

        assert self.zonotope_w.ndim == 4, "self should have 4 dims"
        assert other.zonotope_w.ndim == 4, "other should have 4 dims"

        num_attention_heads = self.zonotope_w.shape[0]
        nA = self.num_words
        nB = other.num_words

        n_new_error_terms = num_attention_heads * nA * nB
        C = torch.zeros(num_attention_heads, self.zonotope_w.shape[1] + n_new_error_terms, nA, nB, device=self.args.device)

        def multiply_matrices(left: torch.Tensor, right_with_special_norm: torch.Tensor, p_right: float) -> torch.Tensor:
            dual_norm_right = dual_norm(p_right)

            _, E1, N1, v1 = left.shape
            _, E2, N2, v2 = right_with_special_norm.shape
            assert v1 == v2, "bad v"
            E = E1

            A = get_norm(right_with_special_norm, p=dual_norm_right, dim=1)  # Shape: (A, N2, v)

            # for the matmul, we need (A, N1, 1, E, v) x (A, N2, v, 1) -> (A, N1, N2, E, 1)
            left_good_shape = left.transpose(1, 2).unsqueeze(2)  # left: (A, E, N1, v) -> (A, N1, 1, E, v)
            A_good_shape = A.unsqueeze(-1).unsqueeze(1)  # A:    (A, N2, v)    -> (A, 1, N2, v, 1)

            result = torch.matmul(left_good_shape.abs(), A_good_shape)  # We accumulate the intervals
            assert result.shape == torch.Size([num_attention_heads, N1, N2, E, 1])

            # We want to update the shape of result from (A, N1, N2, E, 1) to (A, E, N1, N2)
            result_good_shape = result.squeeze(-1).permute(0, 3, 1, 2)

            assert (result_good_shape >= 0).all(), "result has some negative numbers"

            return result_good_shape

        def multiply_matrices_p_p(left: torch.Tensor, right: torch.Tensor, p: float) -> torch.Tensor:
            q = dual_norm(p)
            _, E1, N1, v1 = left.shape
            _, E2, N2, v2 = right.shape
            assert v1 == v2, "bad v"
            E = E1

            A = get_norm(right, q, dim=1)  # Shape: (A, N2, v)

            # for the matmul, we need (A, N1, 1, E, v) x (A, N2, v, 1) -> (A, N1, N2, E, 1)
            left_good_shape = left.transpose(1, 2).unsqueeze(2)  # left: (A, E, N1, v) -> (A, N1, 1, E, v)
            A_good_shape = A.unsqueeze(-1).unsqueeze(1)  # A:    (A, N2, v)    -> (A, 1, N2, v, 1)

            result = torch.matmul(left_good_shape.abs(), A_good_shape)  # We accumulate the intervals
            assert result.shape == torch.Size([num_attention_heads, N1, N2, E, 1])

            # We want to update the shape of result from (A, N1, N2, E, 1) to (A, E, N1, N2)
            result_good_shape = result.squeeze(-1).permute(0, 3, 1, 2)

            assert (result_good_shape >= 0).all(), "result has some negative numbers"

            return result_good_shape

        #### SECTION: A0 * B0
        # First row of C[A, 0] = first vector of self x all other vector of other
        # First col of C[A, 0] =   all vector of self x first vector of other
        C[:, 0] = torch.bmm(self.zonotope_w[:, 0], other.zonotope_w[:, 0].transpose(1, 2))

        #### SECTION: A0 * B_errors   +     B0 * A_errors
        els_A = self.zonotope_w[:, 1:]  # (A, n_error, nA, embedding_size)
        els_A_0 = self.zonotope_w[:, 0:1].repeat(1, self.num_error_terms, 1, 1)  # (A, n_error, nA, embedding_size)

        els_B = other.zonotope_w[:, 1:]  # (A, n_error, nB, embedding_size)
        els_B_0 = other.zonotope_w[:, 0:1].repeat(1, self.num_error_terms, 1, 1)  # (A, n_error, nB, embedding_size)

        sum_errors_1 = torch.matmul(els_A_0, els_B.transpose(2, 3))  # (A, n_error, nA, embedding_size)  matmul  (A, n_error, embedding_size, nB)
        sum_errors_2 = torch.matmul(els_A, els_B_0.transpose(2, 3))  # (A, n_error, nA, embedding_size)  matmul  (A, n_error, embedding_size, nB)
        C[:, 1:1+self.num_error_terms] = sum_errors_1 + sum_errors_2  # (A, num_error_terms, nA, nB)

        #### SECTION: A_errors * B_errors
        #
        # We will decompose the errors into two groups: E_p and E_inf
        # where E_p contains the error terms bounded by the 1-norm or 2-norm and E_inf contains the error terms bounded by the inf-norm
        #
        # Therefore,
        #      A_errors = A_p + A_inf
        #      B_errors = B_p + B_inf
        # and
        #      A_errors * B_errors = (A_p + A_inf) * (B_p + B_inf)
        #                          = A_p * B_p  + A_p * B_inf + B_p * A_inf +  A_inf * B_inf
        A_p, A_inf = self.get_input_special_errors_and_new_error_terms(self.zonotope_w[:, 1:])  # (A, E_p, N1, v)
        B_p, B_inf = other.get_input_special_errors_and_new_error_terms(other.zonotope_w[:, 1:])  # (A, E_inf, N2, v)

        assert A_p.size(1) == B_p.size(1), "A and B don't have the same number of non-inf error terms"
        assert A_inf.size(1) == B_inf.size(1), "A and B don't have the same number of inf error terms"
        assert A_p.size(3) == B_p.size(3), "A and B don't have the same number of elements in the vectors - p"
        assert A_inf.size(3) == B_inf.size(3), "A and B don't have the same number of elements in the vectors - inf"
        assert self.p == other.p, "Self and other don't have the same norm"

        new_weights = torch.zeros(num_attention_heads, nA, nB, device=self.device)

        ### A_p * B_p
        # TODO(multi noise symbol groups): adapt
        if A_p.size(1) > 0 and B_p.size(1) > 0:
            assert self.p == 1 or self.p == 2, "there are 1-norm or 2-norm bound error terms but the norm is INF"
            error_1_coeffs = multiply_matrices_p_p(A_p, B_p, self.p)  # (A, e_errors_1, NA, NB)
            new_weights += get_norm(error_1_coeffs, p=dual_norm(self.p), dim=1)  # inplace
        else:
            assert self.p > 10, "there are no 1-norm or 2-norm bound error terms but the norm is 1 or 2"

        def do_mixed_multiplication(inf_errors: torch.Tensor, p_errors: torch.Tensor, p: float, use_other_dot_product_ordering: bool, inf_is_self: bool):
            assert not use_other_dot_product_ordering, "use_other_dot_product_ordering not supported yet"
            errors_inf_coeffs1_v2 = multiply_matrices(
                p_errors, right_with_special_norm=inf_errors, p_right=INFINITY
            )  # (A, e_error_1, NB, NA)

            # (A, NB, NA)
            result_v2 = get_norm(errors_inf_coeffs1_v2, p=dual_norm(p), dim=1)

            if inf_is_self:
                result_v2 = result_v2.transpose(1, 2)   # (A, NA, NB)
            assert (result_v2 >= 0).all(), "v2"
            return result_v2

        ### A_p * B_inf + B_p * A_inf
        if A_p.size(1) > 0 and B_p.size(1) > 0 and A_inf.size(1) > 0 and B_inf.size(1):
            new_weights += do_mixed_multiplication(A_inf, B_p, self.p, self.args.use_other_dot_product_ordering, inf_is_self=True)  # inplace
            new_weights += do_mixed_multiplication(B_inf, A_p, self.p, self.args.use_other_dot_product_ordering, inf_is_self=False)  # inplace

        ### A_inf * B_inf
        if A_inf.size(1) > 0 and B_inf.size(1):
            errors_inf_coeffs3 = multiply_matrices(A_inf, B_inf, p_right=INFINITY)  # (A, e_errors_inf, NA, NB)
            new_weights += get_norm(errors_inf_coeffs3, p=dual_norm(INFINITY), dim=1)  # inplace

        # num_old_errors = 1 + self.num_error_terms
        # all_attention_heads = torch.ones(num_attention_heads, dtype=torch.bool, device=self.args.device)
        # has_error_terms = torch.ones(nA, nB, dtype=torch.bool, device=self.args.device)
        # indices = torch.arange(num_old_errors, num_old_errors + num_attention_heads * nA * nB)
        # C[all_attention_heads, indices, has_error_terms] = new_weights[all_attention_heads, has_error_terms]

        has_error_terms = torch.ones(nA, nB, dtype=torch.bool, device=self.args.device)
        for attention_head in range(num_attention_heads):
            start_index = 1 + self.num_error_terms + attention_head * nA * nB
            indices = torch.arange(start_index, start_index + nA * nB)
            C[attention_head, indices, has_error_terms] = new_weights[attention_head, has_error_terms]

        return make_zonotope_new_weights_same_args(C, source_zonotope=self, clone=False)

    def do_multiply(self, other: "Zonotope", **kwargs) -> "Zonotope":
        if self.args.use_dot_product_variant3:
            assert "use_dot_product_variant3 not supported yet for Zonotope.do_multiply()"
        assert not self.args.use_other_dot_product_ordering, "use_other_dot_product_ordering not supported yet"

        assert self.args.perturbed_words == 1, "Multiple perturbed words not supported yet for Zonotope.dot_product_fast_batch()"

        if self.num_error_terms < other.num_error_terms:
            self = self.expand_error_terms_to_match_zonotope(other)
        elif other.num_error_terms < self.num_error_terms:
            other = other.expand_error_terms_to_match_zonotope(self)

        assert self.zonotope_w.ndim == 3, "self should have 4 dims"
        assert other.zonotope_w.ndim == 3, "other should have 4 dims"
        assert self.zonotope_w.shape[1] == other.zonotope_w.shape[1], "Dim 1 doesn't match"
        assert self.zonotope_w.shape[2] == other.zonotope_w.shape[2], "Dim 2 doesn't match"

        _, nA, nB = self.zonotope_w.shape

        n_new_error_terms = nA * nB
        C = torch.zeros(self.zonotope_w.shape[0] + n_new_error_terms, nA, nB, device=self.args.device)

        def multiply_matrices(left: torch.Tensor, right_with_special_norm: torch.Tensor, p_right: float) -> torch.Tensor:
            dual_norm_right = dual_norm(p_right)
            assert left.shape[1:] == right_with_special_norm.shape[1:], "Incorrect shape"
            E1, N1, N2 = left.shape

            A = get_norm(right_with_special_norm, p=dual_norm_right, dim=0)  # Shape: (N1, N2)

            # (E1, N1, N2) * (N1, N2) = (E1, N1, N2)
            result = left.abs() * A

            assert result.shape == torch.Size([E1, N1, N2]), "Incorrect shape"
            assert (result >= 0).all(), "result has some negative numbers"

            return result

        def multiply_matrices_p_p(left: torch.Tensor, right: torch.Tensor, p: float) -> torch.Tensor:
            return multiply_matrices(left, right, p)

        #### SECTION: A0 * B0
        C[0] = self.zonotope_w[0] * other.zonotope_w[0]

        #### SECTION: A0 * B_errors   +     B0 * A_errors
        els_A = self.zonotope_w[1:]  # (n_error, nA, nB)
        els_A_0 = self.zonotope_w[0:1].repeat(self.num_error_terms, 1, 1)  # (n_error, nA, nB)

        els_B = other.zonotope_w[1:]  # (n_error, nA, nB)
        els_B_0 = other.zonotope_w[0:1].repeat(self.num_error_terms, 1, 1)  # (n_error, nA, nB)

        sum_errors_1 = els_A_0 * els_B  # (n_error, nA, nB) * (n_error, nA, nB)
        sum_errors_2 = els_B_0 * els_A  # (n_error, nA, nB) * (n_error, nB, nB)
        C[1:1+self.num_error_terms] = sum_errors_1 + sum_errors_2  # (A, num_error_terms, nA, nB)

        #### SECTION: A_errors * B_errors
        #
        # We will decompose the errors into two groups: E_p and E_inf
        # where E_p contains the error terms bounded by the 1-norm or 2-norm and E_inf contains the error terms bounded by the inf-norm
        #
        # Therefore,
        #      A_errors = A_p + A_inf
        #      B_errors = B_p + B_inf
        # and
        #      A_errors * B_errors = (A_p + A_inf) * (B_p + B_inf)
        #                          = A_p * B_p  + A_p * B_inf + B_p * A_inf +  A_inf * B_inf
        A_p, A_inf = self.get_input_special_errors_and_new_error_terms(self.zonotope_w[1:])  # (E_p, N1, v)
        B_p, B_inf = other.get_input_special_errors_and_new_error_terms(other.zonotope_w[1:])  # (E_inf, N2, v)

        assert A_p.size(0) == B_p.size(0), "A and B don't have the same number of non-inf error terms"
        assert A_inf.size(0) == B_inf.size(0), "A and B don't have the same number of inf error terms"
        assert A_p.size(2) == B_p.size(2), "A and B don't have the same number of elements in the vectors - p"
        assert A_inf.size(2) == B_inf.size(2), "A and B don't have the same number of elements in the vectors - inf"
        assert self.p == other.p, "Self and other don't have the same norm"

        new_weights = torch.zeros(nA, nB, device=self.device)

        ### A_p * B_p
        # TODO(multi noise symbol groups): adapt
        if A_p.size(0) > 0 and B_p.size(0) > 0:
            assert self.p == 1 or self.p == 2, "there are 1-norm or 2-norm bound error terms but the norm is INF"
            error_1_coeffs = multiply_matrices_p_p(A_p, B_p, self.p)  # (e_errors_1, nA, nB)
            new_weights += get_norm(error_1_coeffs, p=dual_norm(self.p), dim=0)  # inplace, does not matter for now
        else:
            assert self.p > 10, "there are no 1-norm or 2-norm bound error terms but the norm is 1 or 2"

        def do_mixed_multiplication(inf_errors: torch.Tensor, p_errors: torch.Tensor, p: float):
            errors_inf_coeffs1 = multiply_matrices(
                p_errors, right_with_special_norm=inf_errors, p_right=INFINITY
            )  # (e_error_1, NA, NB)

            # (NA, NB)
            result = get_norm(errors_inf_coeffs1, p=dual_norm(p), dim=0)
            assert (result >= 0).all(), "v2"
            return result

        ### A_p * B_inf + B_p * A_inf
        if A_p.size(0) > 0 and B_p.size(0) > 0 and A_inf.size(0) > 0 and B_inf.size(0):
            new_weights += do_mixed_multiplication(A_inf, B_p, self.p)  # inplace, does not matter for now
            new_weights += do_mixed_multiplication(B_inf, A_p, self.p)  # inplace, does not matter for now

        ### A_inf * B_inf
        if A_inf.size(0) > 0 and B_inf.size(0):
            errors_inf_coeffs3 = multiply_matrices(A_inf, B_inf, p_right=INFINITY)  # (e_errors_inf, NA, NB)
            new_weights += get_norm(errors_inf_coeffs3, p=dual_norm(INFINITY), dim=0)  # inplace, does not matter for now

        num_old_errors = 1 + self.num_error_terms
        has_error_terms = torch.ones(nA, nB, dtype=torch.bool, device=self.args.device)
        indices = torch.arange(num_old_errors, num_old_errors + nA * nB)
        C[indices, has_error_terms] = new_weights[has_error_terms]

        return make_zonotope_new_weights_same_args(C, source_zonotope=self, clone=False)

    def divide(self, W: "Zonotope", use_original_reciprocal=True, y_positive_constraint=False) -> "Zonotope":
        if type(W) == Zonotope:
            if self.zonotope_w.ndim == 4:
                assert W.zonotope_w.ndim == 4, "Divide: if self has 4 dims, then the other zonotope must have 4 dims too"
                assert self.zonotope_w.shape[0] == W.zonotope_w.shape[0], "Divide: both terms must have same number of attention heads"
                num_attention_heads = self.zonotope_w.shape[0]

                W_reshaped = W.remove_attention_heads_dim()
                self_reshaped = self.remove_attention_heads_dim()
                result = self_reshaped.divide(W_reshaped)
                return result.add_attention_heads_dim(num_attention_heads, clone=False)

            W_reciprocal = W.reciprocal(original_implementation=use_original_reciprocal, y_positive_constraint=y_positive_constraint)
            # l, u = W_reciprocal.concretize()
            # assert (l > 0).all(), "Divide: reciprocal isn't positive (min l = %f)" % l.min()

            self_with_extra_error_terms = self.expand_error_terms_to_match_zonotope(W_reciprocal)
            res = self_with_extra_error_terms.multiply(W_reciprocal)
            # l, u = res.concretize()
            # assert (l > 0).all(), "Divide: multiplication with reciprocal isn't positive (min l = %f)" % l.min()

            return res
        else:
            raise NotImplementedError()

    def context(self, value: "Zonotope") -> "Zonotope":
        return self.dot_product(other=value.t())

    def relu(self, lambdas=None) -> "Zonotope":
        # x: (1 + word_embedding_dim) x n_words x word_embedding_dim
        # computes minimum and maximum boundaries for every input value
        # upper and lower bound are tensor of size n_features x (h x w) || vector_size
        # creates the lambda values
        lower, upper = self.concretize()

        if lambdas is None:
            # print("Zonotope ReLU: lambdas not provided, using lambdas that minimize the area")
            lambdas = upper / (upper - lower + epsilon)

        # terms that have new error weights
        overlapping_cases = (lower * upper < 0).float()
        n_new_error_terms = int(torch.sum(overlapping_cases).item())

        shape = list(self.zonotope_w.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape, device=self.args.device)

        # delta is the difference in height between top and bottom lines
        # how to compute delta ?
        #   - bottom border is lambda * x
        #   - top border has the following lambda * x + delta, with delta to determine
        #     top border is above ReLU curve in u, so lambda * u + delta >= u
        #     top border is above ReLU curve in l, so lambda * l + delta >= 0
        #     delta >= (1 - lambda) * u
        #     delta >= -lambda * l
        #     so delta >= max((1 - lambda) * u, -lambda * l), and we take equality
        #   - difference between the two lines is delta = max((1 - lambda) * u, -lambda * l)
        # the new bias is therefore (in crossing border cases) delta/2
        delta = torch.max(-lambdas * lower, (1 - lambdas) * upper)

        # for crossing border cases, we modify bias
        # for negative case, 0 is the new bias
        # for positive case, we don't change anything
        transformed_x[0] = (lambdas * self.zonotope_w[0] + delta / 2) * overlapping_cases \
                           + self.zonotope_w[0] * (lower >= 0).float()

        # for crossing border cases, we multiply by lambda error weights
        # for positive cases, we don't change anything
        # for negative cases, it is 0
        # modifying already existing error weights
        transformed_x[1:1 + self.num_error_terms] = self.zonotope_w[1:] * lambdas * overlapping_cases \
                                                    + self.zonotope_w[1:] * (lower >= 0).float()

        # Add the new errors terms efficiently using slicing
        halfDelta = delta / 2
        overlapping_cases = (lower * upper < 0)
        values = halfDelta[overlapping_cases]

        i_error = self.zonotope_w.shape[0]
        indices = torch.arange(i_error, i_error + n_new_error_terms)
        transformed_x[indices, overlapping_cases] = values

        return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def exp_minimal_area(self, return_raw_tensors_separately=False, allow_modifying_input_zonotope=False):
        if self.zonotope_w.ndim == 4:
            A = self.zonotope_w.shape[0]
            return self.remove_attention_heads_dim(clone=False).exp_minimal_area(return_raw_tensors_separately, allow_modifying_input_zonotope).add_attention_heads_dim(A, clone=False)

        l, u = self.concretize()

        # terms that have new error weights
        different_bool = has_new_error_term = (l != u)
        n_new_error_terms = different_bool.sum().item()
        equal_bool = (l == u)

        shape = list(self.zonotope_w.shape)
        if not return_raw_tensors_separately:
            shape[0] += n_new_error_terms

        t_crit = ((u.exp() - l.exp()) / (u - l)).log()
        t_crit[u == l] = float('inf')  # Replace NaNs by infinity
        t_crit[t_crit == float('-inf')] = (0.5 * l + 0.5 * u)[t_crit == float('-inf')]  # Replace -Inf that arise from the log to avg(l, u)
        # t_crit[t_crit == float('-inf')] = float('+inf')  # Replace -Inf that arise from the log to avg(l, u)
        t_crit2 = l + 0.95
        t_opt = torch.min(torch.min(t_crit, t_crit2), u)  # Idea: has to be below L + 1 (and <= U)

        if (t_opt == float('-inf')).any():
            t_opt = torch.min(torch.min(t_crit, t_crit2), u)  # Idea: has to be below L + 1 (and <= U)

        # λ = f'(t) = e^t
        lambdas = t_opt.exp()

        # Final zonotope = line bottom + region above of uncertainty to fit the upper bound
        # Line bottom touches at the point (t, f(t)), therefore line_bottom:   y = λx + b
        # We have that λ = f'(t) = e^t     and also that b = y -  λx = f(t) - λt
        #
        # so line_bottom has equation: y =  λx   +   (f(t) - λt)
        #
        # for the region above, what's the maximum uncertainty?
        # in our case, we have that t <= t_crit and therefore the line has a low slope and therefore the final upper bound touches f(u)
        # the maximum distance between f(x) and the upper bound happens at t
        # here we have that f has value LOWER = f(t) and that line has value   UPPER : f(u) + λ(t - u)
        # therefore the max range of uncertainty has
        #      WIDTH = UPPER - LOWER = (f(u) + λt - λu) - f(t) = λt - f(t) + (f(u) - λu)
        #      CENTER = RADIUS = WIDTH / 2 = 0.5 * (λt - f(t) + (f(u) - λu))
        #
        # so the final equation is
        #    y = line_bottom                + uncertainty region
        #      = (λx   +   (f(t) - λt))     +     CENTER + RADIUS * eps
        #      = λx     +    (f(t) - λt) + CENTER)   + RADIUS * eps
        #
        #   so the NEW_COEFF = RADIUS = 0.5 * (λt - f(t) + (f(u) - λu))
        #  and the NEW_CONST = f(t) - λt + CENTER =
        #                    = f(t) - λt + 0.5 * (λt - f(t) + (f(u) - λu)) =
        #                    = 0.5 (f(t) - λt +  (f(u) - λu))
        #
        # This leads to the same equation as the one in Mark's paper:
        #    X = e^u - λu
        #    E = 0.5 * (e^t - λt  +  X )
        #    u = 0.5 * (λt - e^t  +  X )

        # V1
        # NEW_CONSTS_OLD = 0.5 * (t_opt.exp() - lambdas * t_opt + u.exp() - lambdas * u)
        # NEW_COEFFS_OLD = 0.5 * (lambdas * t_opt - t_opt.exp() + u.exp() - lambdas * u)

        # V2 to try to avoid fp problems
        # NEW_CONSTS = 0.5 * (t_opt.exp() - lambdas * t_opt - lambdas * u)
        # NEW_CONSTS = 0.5 * (t_opt.exp() - lambdas * (t_opt + u))
        NEW_CONSTS = 0.5 * (lambdas * (1 - t_opt - u))
        NEW_CONSTS += 0.5 * u.exp()  # inplace

        # NEW_COEFFS = 0.5 * (lambdas * t_opt - t_opt.exp() - lambdas * u)
        # NEW_COEFFS = 0.5 * (- t_opt.exp() + lambdas * (t_opt  - u))
        NEW_COEFFS = 0.5 * (lambdas * (t_opt  - u - 1))
        NEW_COEFFS += 0.5 * u.exp()  # inplace

        INTERCEPT = (t_opt.exp() - lambdas * t_opt)
        if not ((NEW_CONSTS - INTERCEPT)[different_bool] >= -1e-4).all():
            a = 5

        assert ((NEW_CONSTS - INTERCEPT)[different_bool] >= -1e-4).all(), \
            f"exp_mark: diff < 0. diff min = {(NEW_CONSTS - INTERCEPT)[different_bool].min()}, " \
            f"const min = {NEW_CONSTS[different_bool].min()},  intercept.max = {INTERCEPT.max()}"
        assert (NEW_COEFFS[different_bool] >= -1e-4).all(), "exp_mark: NEW_COEFFS is negative. min = %f" % NEW_COEFFS[different_bool].min().item()

        cleanup_memory()

        if return_raw_tensors_separately:
            assert allow_modifying_input_zonotope, "exp_mark: special case must acknowledge input zonotope is modified"

            # Re-use the input zonotope tensor
            transformed_x = self.zonotope_w

            # Center
            transformed_x[0] *= lambdas  # inplace
            transformed_x[0] += NEW_CONSTS  # inplace
            transformed_x[0, equal_bool] = l[equal_bool].exp()

            # transformed_x[1:1 + self.num_error_terms] = self.zonotope_w[1:]  # Copy everthing
            # inplace
            transformed_x[1:, equal_bool] = 0.0  # But set the error terms of the exact values (l = u) to 0
            transformed_x[1:] *= lambdas  # Then multiply by lambda. This basically multiplies everything by lambda

            NEW_COEFFS[equal_bool] = 0
            return transformed_x, NEW_COEFFS
        else:
            transformed_x = torch.zeros(shape, device=self.args.device)

            # Step 1) new bias
            # for the equality case, we put the value 1/x
            # for the difference case, we put the center λ a0 + (T + B)/2
            # inplace
            transformed_x[0] = lambdas * self.zonotope_w[0] + NEW_CONSTS
            # inplace
            transformed_x[0, equal_bool] = l[equal_bool].exp()

            # Step 2) updated error weights
            # for the equality case, there are no error terms
            # for the difference case, we multiply the current error terms by λ
            import gc; gc.collect(); torch.cuda.empty_cache()

            # This is done this way to avoid creating unneeded tensors, which blow up the memory requirement
            # inplace
            transformed_x[1:1 + self.num_error_terms] = self.zonotope_w[1:]  # Copy everthing
            # inplace
            transformed_x[1:1 + self.num_error_terms, equal_bool] = 0.0      # But set the error terms of the exact values (l = u) to 0
            # inplace
            transformed_x[1:1 + self.num_error_terms] *= lambdas             # Then multiply by lambda. This basically multiplies everything by lambda

            # Step 3) new error weights
            # Add the new errors terms efficiently using slicing

            # We use two tensor as indices, because then it does pair-wise indexing
            # E.g. it matches the indices with the indices of the True values of has_new_error_term (skipping the False values)
            # So x[torch.tensor([4, 5]), torch.tensor([[True, False, True]])] will get the values at index [4, 0] and [5, 2]
            #
            # c = 0
            # i = 0
            # for index in range(indices):
            #     while has_new_error_term[c] == False:
            #         c += 1
            #
            #     x[index, c] = val[i]
            #     c += 1
            #     i += 1
            #
            # If we did transformed_x[num_error_terms:num_error_terms + n_new_error_terms, has_new_error_term]
            # then it would do a cross product, i.e. indices X has_new_error_term
            indices = torch.arange(1 + self.num_error_terms, 1 + self.num_error_terms + n_new_error_terms)
            # inplace
            transformed_x[indices, has_new_error_term] = NEW_COEFFS[different_bool]

            return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def exp_simple(self) -> "Zonotope":
        # lower bound: y >= f(l) + λ * (x - l)
        # upper bound: y <= f(u) + λ * (x - u)
        #
        #  we must also have f(l) + λ * (x - l) <= f(u) + λ * (x - u)
        #  which implies that f(l) - f(u) <= λ ((x - u) - (x - l))
        #                     f(l) - f(u) <= λ (l - u)
        #                     (f(l) - f(u)) / ((l - u)) >= λ
        #                     (f(u) - f(l)) / ((u - l)) >= λ
        #                     λ <= (f(u) - f(l)) / ((u - l))
        # when we did min(avg(l, u), l + 0.99)), we might have picked a point
        # such that the slope doesn't respect this law above, or in other
        # word a point above t_crit!
        #
        # New logic (experiment):
        #   The point we'll pick to get the slope is d = min(average(l, u), l + 1 - 0.01)
        #   We will have that λ = exp(min(d, 12.0))
        # Previous logic:
        #   λ = f'(l) = exp^(l)
        #
        # And so we will have that f(l) + λ * (x - l) <= y <= f(u) + λ * (x - u)
        # Let's seperate this into things that depend on X and things that don't:
        # (f(l) - λl) + λx <= y <= (f(u) - λu) + λx
        # We will call B = (f(l) - λl) and T = (f(u) - λu) and so B + λx <= y <= T + λx
        #
        # And so if c in the range [0, 1] then y = λx + B + c (T - B)
        # However, since eps is in the range [-1, 1], then c = (eps + 1) / 2
        # Substituting and simplifying, we obtain
        # y = λx + (T + B) / 2 + eps (T - B) / 2
        #
        # And since x = a0 + sum(eps_i error_i)
        # Then the final y will be
        # y = λ (a0 + sum(eps_i error_i)) + (T + B) / 2 + eps (T - B)/2
        #   = (λ a0 + (T + B)/2)  + sum(λ eps_i error_i)  + eps_new (T - B)/2
        #
        # In summary:
        #   1) new bias = λ a0 + (T + B)/2
        #   2) old error terms: multiplied by λ
        #   3) new error term has weight (T - B)/2
        if self.zonotope_w.ndim == 4:
            A = self.zonotope_w.shape[0]
            return self.remove_attention_heads_dim().exp_simple().add_attention_heads_dim(A, clone=False)

        l, u = self.concretize()

        # terms that have new error weights
        different_bool = has_new_error_term = (l != u)
        n_new_error_terms = different_bool.sum().item()

        different = different_bool.float()
        equal = (l == u).float()

        shape = list(self.zonotope_w.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape, device=self.args.device)

        lambdas = l.exp()
        B = (l.exp() - lambdas * l)
        T = (u.exp() - lambdas * u)

        # Step 1) new bias
        # for the equality case, we put the value 1/x
        # for the difference case, we put the center λ a0 + (T + B)/2
        transformed_x[0] = equal * u.exp() \
                         + different * (lambdas * self.zonotope_w[0] + (T + B) / 2)

        # Step 2) updated error weights
        # for the equality case, there are no error terms
        # for the difference case, we multiply the current error terms by λ
        transformed_x[1:1 + self.num_error_terms] = different * self.zonotope_w[1:] * lambdas

        # Step 3) new error weights
        # Add the new errors terms efficiently using slicing
        halfDelta = (T - B) / 2
        values = halfDelta[different_bool]

        indices = torch.arange(1 + self.num_error_terms, 1 + self.num_error_terms + n_new_error_terms)
        transformed_x[indices, has_new_error_term] = values

        return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def exp(self, return_raw_tensors_separately=False, allow_modifying_input_zonotope=False, minimal_area=True) -> "Zonotope":
        if minimal_area:
            return self.exp_minimal_area(return_raw_tensors_separately, allow_modifying_input_zonotope)
        else:
            return self.exp_simple()

    def softmax(self, verbose=False,
                use_new_softmax=True,
                no_constraints=True,                    # TODO: think if this should be enabled or disable by default
                add_value_positivity_constraint=False,   # TODO: think if this should be enabled or disabled by default
                use_new_reciprocal=True
        ) -> "Zonotope":
        num_rows = self.zonotope_w.size(-2)
        num_values = self.zonotope_w.size(-1)
        A = self.zonotope_w.size(0)
        if use_new_softmax:
            # To encode the softmax, the following approach will be taken:
            # softmax_i(x1, ..., xn) = exp(xi) / (exp(x1) + ... exp(xn))
            #                        = 1 / ((exp(x1) + ... + exp(xn)) / exp(xi))
            #                        = 1 / ((exp(x1 - xi) + ... + exp(xn - xi))
            # so we will have to have subtractions, exponentials and in the top formulation also divisions
            # We'll begin by using the bottom formulation
            # cleanup_memory()

            if self.args.batch_softmax_computation:
                sum_w_list, new_error_terms_collapsed_list = [], []
                for a in range(A):
                    sum_exp_diffs_w, new_error_terms_collapsed = process_values(
                        self.zonotope_w[a:a+1], self, A=1, num_rows=num_rows, num_values=num_values,
                        keep_intermediate_zonotopes=self.args.keep_intermediate_zonotopes
                    )
                    sum_w_list.append(sum_exp_diffs_w)
                    new_error_terms_collapsed_list.append(new_error_terms_collapsed)
                    cleanup_memory()

                sum_exp_diffs_w = torch.cat(sum_w_list, dim=0)
                new_error_terms_collapsed = torch.cat(new_error_terms_collapsed_list, dim=0)
            else:
                sum_exp_diffs_w, new_error_terms_collapsed = process_values(
                    self.zonotope_w, self, A, num_rows, num_values,
                    keep_intermediate_zonotopes=self.args.keep_intermediate_zonotopes
                )

            # (A * num_rows * num_values, A, num_rows, num_values)
            new_error_terms_collapsed_intermediate_shape = torch.zeros(A * num_rows * num_values, A, num_rows, num_values, device=self.device)
            indices = torch.arange(A * num_rows * num_values, device=self.device)
            to_add = torch.ones_like(new_error_terms_collapsed, dtype=torch.bool, device=self.device)
            new_error_terms_collapsed_intermediate_shape[indices, to_add] = new_error_terms_collapsed[to_add]

            # (A, A * num_rows * num_values, num_rows, num_values)
            new_error_terms_collapsed_good_shape = new_error_terms_collapsed_intermediate_shape.permute(1, 0, 2, 3)

            if not self.args.keep_intermediate_zonotopes:
                del new_error_terms_collapsed_intermediate_shape
                cleanup_memory()

            # (A, error dims that existed before exp, num_rows, num_values)
            final_sum_exps_zonotope_w = torch.cat([sum_exp_diffs_w, new_error_terms_collapsed_good_shape], dim=1)

            zonotope_sum_exp_diffs = make_zonotope_new_weights_same_args(final_sum_exps_zonotope_w, source_zonotope=self, clone=False)
            # return zonotope_sum_exp_diffs

            ### Step 4: Compute the inverse for all of these sums, thus obtaining all the softmax values
            zonotope_softmax = zonotope_sum_exp_diffs.reciprocal(original_implementation=not use_new_reciprocal, y_positive_constraint=add_value_positivity_constraint)
        else:
            # zonotope_with_adjusted_center = self.subtract_max_from_bias()
            zonotope_exp = self.exp_minimal_area()
            # Input: (n_attention_heads, 1 + n_error terms, num_rows, num_values)
            # Output: (n_attention_heads, 1 + n_error terms, num_rows, num_values)

            l, u = zonotope_exp.concretize()
            if torch.isnan(l).any():
                print("Bound have NaN values: Lower - %s values   Upper - %s values" % (torch.isnan(l).sum().item(), torch.isnan(u).sum().item()))
                l, u = zonotope_exp.concretize()
            assert (l > -1e-9).all(), "Softmax: Exp is negative or 0 (min l = %.9f)" % l.min()

            # Are the scores organized row-wise or column wise? given that the old code used dim=-1, I think it's column wise
            zonotope_sum_w = zonotope_exp.zonotope_w.sum(dim=-1, keepdim=True).repeat(1, 1, 1, num_values)
            zonotope_sum = make_zonotope_new_weights_same_args(zonotope_sum_w, self, clone=False)

            # l, u = zonotope_sum.concretize()
            # assert (l > 0).all(), "Softmax: Exp Sum isn't positive (min l = %f)" % l.min()

            zonotope_softmax = zonotope_exp.divide(zonotope_sum, use_original_reciprocal=not use_new_reciprocal, y_positive_constraint=add_value_positivity_constraint)
            # l, u = zonotope_softmax.concretize()
            # assert (l > 0).all(), "Softmax: Softmax result pre-constraint isn't positive (min l = %f)" % l.min()

        if no_constraints:
            return zonotope_softmax

        u, l = zonotope_softmax.concretize()

        # Some are already good, most other are almost perfect (happens in the 12 layer networks)
        if (l.sum(dim=-1) - 1).abs().max().item() < 1e-6 and (u.sum(dim=-1) - 1).abs().max().item() < 1e-6:
            del u, l
            cleanup_memory()
            return zonotope_softmax

        zonotope_softmax_sum_constrained = zonotope_softmax.add_equality_constraint_on_softmax()
        return zonotope_softmax_sum_constrained

    def add_equality_constraint_on_softmax(self) -> "Zonotope":
        # The shape of the softmax will be (1 + num error terms, num_words, num_words)
        # The softmax values will arranged in the last dimensions, e.g. the first softmax is zonotope_w[:, 0, :]

        # We will have softmax1 + softmax2 + ... + softmax_n = 1
        # And rewriting we will have that softmax1 = 1 - (softmax_2 + ... + softmax_n)
        A, num_rows, num_softmaxes = self.zonotope_w.size(0), self.zonotope_w.size(2), self.zonotope_w.size(3)
        permuted_weights = self.zonotope_w.permute(1, 0, 2, 3)
        flattened_weights = permuted_weights.reshape(permuted_weights.size(0), -1, permuted_weights.size(3))

        ### Setup LHS and RHS x1 = 1 - (x2 + x3 + ... + xn) which is equivalent to x1 + ... + xn = 1
        left_hand_side = flattened_weights[:, :, 0]  # (num_error_terms, A * num_rows)
        right_hand_side = -flattened_weights[:, :, 1:].sum(dim=-1)  # INFO: not sure the clone is needed but I do it for safety
        right_hand_side[0] += 1

        differences = left_hand_side - right_hand_side
        pivot_error_index_per_equation = get_last_nonzero_index_per_row(differences.t())
        equations_already_good = (pivot_error_index_per_equation[0] == 0)
        equations_to_fix = (pivot_error_index_per_equation[0] != 0)
        equations_should_be_0 = differences[:, equations_already_good]
        lower, upper = equations_should_be_0[0] - equations_should_be_0[1:].abs().sum(), equations_should_be_0[0] + equations_should_be_0[1:].abs().sum()
        if (lower.abs() > 1e-6).any() and (upper.abs() > 1e-6).any():
            assert False, "Problem with values"

        ### Find the new contraint zonotope for x1 and receive the indices necessary to update the other variables
        resulting_zonotope, indices_of_removed_vars = add_equality_constraints(
            left_hand_side[:, equations_to_fix], right_hand_side[:, equations_to_fix],
            z=self, min_removable_error_term=self.num_input_error_terms_special_norm, number_initial_terms_whose_range_should_not_change=self.num_input_error_terms_special_norm
        )
        # Edge case: only one equation to fix, need to add extra dim
        if indices_of_removed_vars.ndim == 0:
            indices_of_removed_vars = indices_of_removed_vars.unsqueeze(0)


        ### Update the weights of the other terms (x2 ... xn) by doing a substitution on the var not used in x1
        # left_hand_size = right_hand_side is equivalent to left_hand_side - right_hand_side = 0
        constraints = left_hand_side[:, equations_to_fix] - right_hand_side[:, equations_to_fix]  # (num_error_terms, A * num_rows)
        coeffs_for_pivot_terms = constraints.gather(0, indices_of_removed_vars.unsqueeze(0))  # (A * num_rows)
        replacement_for_pivot_term = constraints / (-coeffs_for_pivot_terms)  # (num_error_terms, A * num_rows)

        # replacement_for_pivot_term[i][j] -> in equation j, the coeffs should be added to eliminate 1 * error_term_to_remove
        # x2 = c1 e1 + c2 e2 + c3 e3
        # x3 = d1 e1 + d2 e2 + d3 e3
        #
        # e1 = 3 e2 - 4 e3
        #
        # x2 = c1 (3 e2 - 4e3) + c2 e2 + c3 e3
        #     = x2 + c1 (-1 e1 + 3 e2 - 4 e3)
        #     = x2 + c1 (replacement_for_pivot_term)
        #
        # x3 = x3 + d1 (replacement_for_pivot_term)
        relevant_coeffs = flattened_weights[:, equations_to_fix, 1:]  # (num_error_terms, A * num_rows, num_softmaxes - 1)

        # initial: (A * num_rows)     final: (1, A * num_rows, num_softmaxes - 1)
        indices_of_removed_vars_right_shape = indices_of_removed_vars.reshape(1, -1, 1).repeat(1, 1, num_softmaxes - 1)
        coeffs_of_variable_to_eliminate = relevant_coeffs.gather(0, indices_of_removed_vars_right_shape)[0]

        # coeffs_of_variable_to_eliminate:                  (A * num_rows, num_softmaxes - 1)
        # replacement_for_pivot_term:      (num_error_terms, A * num_rows)
        # delta:                           (num_error_terms, A * num_rows, num_softmaxes - 1)
        change_other_vars_due_to_substitution = replacement_for_pivot_term.unsqueeze(2) * coeffs_of_variable_to_eliminate

        ### Create the final zonotope weight matrix and Zonotope object
        new_weights = flattened_weights.clone()
        # LHS: (num_error_terms, A * num_softmaxes)    RHS: (num_error_terms, A * num_rows, 1) -> (num_error_terms, A * num_rows)
        new_weights[:, equations_to_fix, 0] = resulting_zonotope.zonotope_w.squeeze(2)
        # Shape of changed elements: (num_error_terms, A * num_rows, num_softmaxes - 1)
        new_weights[:, equations_to_fix, 1:] += change_other_vars_due_to_substitution  # inplace, does not matter for now

        # Before: (num_error_terms, A * num_rows, num_softmaxes)     After:  (num_error_terms, A, num_rows, num_softmaxes)
        new_weights_right_shape = new_weights.reshape(new_weights.size(0), A, num_rows, num_softmaxes)
        # Before: (num_error_terms, A, num_rows, num_softmaxes)      After:  (A, num_error_terms, num_rows, num_softmaxes)
        new_weights_right_shape = new_weights_right_shape.permute(1, 0, 2, 3)

        return make_zonotope_new_weights_same_args(new_weights_right_shape, source_zonotope=resulting_zonotope, clone=False)

    def dense(self, dense) -> "Zonotope":
        result = self.matmul(dense.weight)
        if dense.bias is None:
            return result
        else:
            return result.add(dense.bias)

    def tanh(self) -> "Zonotope":
        lower, upper = self.concretize()
        tanh = torch.tanh

        different_bool = has_new_error_term = (lower != upper)
        different = different_bool.float()
        equal_bool = (lower == upper)
        equal = equal_bool.float()

        # Final step: creating the new weight matrix
        n_new_error_terms = int(torch.sum(different).item())

        # Step 1: Create the zonotope weight of the right size
        shape = list(self.zonotope_w.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape, device=self.args.device)

        # print("tanh shape", transformed_x.shape)

        # Step 2: put bias, old weights, new weights
        lambda_optimal = torch.min(1 - tanh(lower).square(), 1 - tanh(upper).square())
        new_biases = lambda_optimal * self.zonotope_w[0] + 0.5 * (tanh(upper) + tanh(lower) - lambda_optimal * (upper + lower))
        new_error_weights = 1 / 2. * (tanh(upper) - tanh(lower) - lambda_optimal * (upper - lower))

        old_num_error_terms = self.zonotope_w.shape[0]
        transformed_x[0] = torch.tanh(lower) * equal + new_biases * different
        transformed_x[1:old_num_error_terms] = self.zonotope_w[1:] * different * lambda_optimal

        indices = torch.arange(old_num_error_terms, old_num_error_terms + n_new_error_terms)
        transformed_x[indices, has_new_error_term] = new_error_weights[different_bool]

        return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def layer_norm(self, normalizer, layer_norm) -> "Zonotope":
        if layer_norm == "no":
            return self

        # Layer normalization: normalize each sample independently according to its mean and variance
        dim_out = self.word_embedding_size
        w_avg = torch.ones((dim_out, dim_out), device=self.device) / dim_out  # avg per = sum followed by division
        zonotope_minus_avg = self.add(self.matmul(w_avg).multiply(-1.))

        if layer_norm == "standard":
            variance = zonotope_minus_avg.square_and_sum_and_repeat().multiply(1 / dim_out)
            # variance = zonotope_minus_avg.multiply(zonotope_minus_avg).matmul(w_avg)
            sqrt = variance.add(epsilon).sqrt()
            normalized = zonotope_minus_avg.divide(sqrt)
        elif layer_norm == "no_var":
            normalized = zonotope_minus_avg
        else:
            raise ValueError(f"Unknown layer norm mode '{layer_norm}'")

        normalized = normalized.multiply(normalizer.weight).add(normalizer.bias)
        return normalized

    def square_and_sum_and_repeat(self) -> "Zonotope":
        # input: (1 + #error, length, E)
        # output: (1 + #error, length, E)
        assert self.zonotope_w.ndim == 3, "Expected 3-Dim zonotope weights"
        E = self.zonotope_w.shape[-1]

        # (1 + #error, length, E) -> (1, 1 + #error, length, E) -> (length, 1 + #error, 1, E)
        zonotope_w_reshaped = self.zonotope_w.unsqueeze(0).transpose(0, 2)
        zonotope_reshaped = make_zonotope_new_weights_same_args(zonotope_w_reshaped, self, clone=False)

        # (length, 1 + #error + #new error, 1, 1)
        result = zonotope_reshaped.dot_product_fast(zonotope_reshaped)
        # result = zonotope_reshaped.dot_product_fast_batch(zonotope_reshaped)

        assert result.zonotope_w.ndim == 4
        assert result.zonotope_w.shape[-1] == 1
        assert result.zonotope_w.shape[-2] == 1

        # (length, 1 + new #errors, 1, 1) -> (1, 1 + new #errors, length, 1) -> (1 + new #errors, length, 1)
        result_w = result.zonotope_w.transpose(0, 2).squeeze(0)
        result_w_repeated = result_w.repeat((1, 1, E))  # (1 + new #errors, length, E)
        return make_zonotope_new_weights_same_args(result_w_repeated, result, clone=False)

    def sqrt(self) -> "Zonotope":
        l, u = self.concretize()

        if torch.min(l) <= epsilon:
            num_negative_elements = (l <= epsilon).float().sum().item()
            num_elements = l.nelement()

            message = "sqrt: Bounds must be positive but %d elements out of %d were < 1e-12 (min value = %.12f, iszero = %s)" % (
                num_negative_elements, num_elements, torch.min(l).item(), torch.min(l).item() == 0)
            assert False, message

        different_bool = has_new_error_term = (l != u)
        equal_bool = (l == u)

        # Final step: creating the new weight matrix
        n_new_error_terms = int(torch.sum(different_bool).item())

        # Step 1: Create the zonotope weight of the right size
        shape = list(self.zonotope_w.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape, device=self.args.device)

        # Step 2: Finding t_crit

        # f(x) = sqrt(x)      therefore      f'(x) = 1 / (2 * sqrt(x))
        # f'(t_crit) = (sqrt(u) - sqrt(l)) / (u - l) = A
        # 1 / (2 * sqrt(t_crit)) = A
        # sqrt(t_crit) = 1/(2A) = (u - l) / (2 * (sqrt(u) - sqrt(l)))    <-  careful here with divisions
        t_crit = ((u - l) / (2 * (u.sqrt() - l.sqrt()))).square()

        def f(x):
            return x.sqrt()

        # Step 3: put bias, old weights, new weights
        lambda_optimal = (f(u) - f(l)) / (u - l)  # lambda = f'(t_crit) = (sqrt(u) - sqrt(l)) / (u - l)
        X = f(l) - lambda_optimal * l
        new_consts = 0.5 * (f(t_crit)               - lambda_optimal * t_crit  + X)
        new_coeffs = 0.5 * (lambda_optimal * t_crit - f(t_crit)                + X)

        old_num_error_terms = self.zonotope_w.shape[0]
        transformed_x[0, equal_bool] = f(l[equal_bool])
        transformed_x[0, different_bool] = lambda_optimal[different_bool] * self.zonotope_w[0, different_bool] + new_consts[different_bool]
        transformed_x[1:old_num_error_terms, different_bool] = self.zonotope_w[1:, different_bool] * lambda_optimal[different_bool]

        indices = torch.arange(old_num_error_terms, old_num_error_terms + n_new_error_terms)
        transformed_x[indices, has_new_error_term] = new_coeffs[different_bool]

        return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def reciprocal(self, original_implementation=True, y_positive_constraint=False) -> "Zonotope":
        """
        Returns a new zonotope representing the reciprocal of the values in this zonotope.
        Requirement: x should be guaranteed to be positive
        """
        y_positive_constraint = True

        if self.zonotope_w.ndim == 4:
            A = self.zonotope_w.size(0)
            return self.remove_attention_heads_dim().reciprocal(original_implementation, y_positive_constraint).add_attention_heads_dim(A, clone=False)

        l, u = self.concretize()

        if torch.min(l) <= epsilon:
            num_negative_elements = (l <= epsilon).float().sum().item()
            num_elements = l.nelement()

            message = "reciprocal: Bounds must be positive but %d elements out of %d were < 1e-12 (min value = %.12f, iszero = %s)" % (
                num_negative_elements, num_elements, torch.min(l).item(), torch.min(l).item() == 0)
            assert False, message

        # terms that have new error weights
        different_bool = has_new_error_term = (l != u)
        equal_bool = (l == u)

        n_new_error_terms = int(has_new_error_term.sum().item())

        shape = list(self.zonotope_w.shape)
        shape[0] += n_new_error_terms
        transformed_x = torch.zeros(shape, device=self.args.device)

        if original_implementation:
            # upper bound: y <= f(l) + λ * (x - l)
            # lower bound: y >= f(u) + λ * (x - u)
            # λ = f'(u) = -1 / u²
            # And so we will have that f(u) + λ * (x - u) <= y <= f(l) + λ * (x - l)
            # Let's seperate this into things that depend on X and things that don't:
            # (f(u) - λu) + λx <= y <= (f(l) - λl) + λx
            # We will call B = (f(u) - λu) and T = (f(l) - λl) and so B + λx <= y <= T + λx
            #
            #  we must also have f(l) + λ * (x - l) >= f(u) + λ * (x - u)
            #  which implies that f(l) - f(u) >= λ ((x - u) - (x - l))
            #                     f(l) - f(u) >= λ (l - u)
            #                     (f(l) - f(u)) / ((l - u)) <= λ
            #                     (f(u) - f(l)) / ((u - l)) <= λ
            #                     λ >= (f(u) - f(l)) / ((u - l))
            # when we did min(avg(l, u), l + 0.99)), we might have picked a point
            # such that the slope doesn't respect this law above, or in other
            # word a point above t_crit!
            #
            # And so if c in the range [0, 1] then y = λx + B + c (T - B)
            # However, since eps is in the range [-1, 1], then c = (eps + 1) / 2
            # Substituting and simplifying, we obtain
            #   y = λx + B + (eps + 1) / 2 * (T - B)
            #   y = λx + B     + 1 / 2 * (T - B)         + eps / 2 * (T - B)
            #   y = λx + (T + B) / 2                     + eps (T - B) / 2
            #
            # And since x = a0 + sum(eps_i error_i)
            # Then the final y will be
            # y = λ (a0 + sum(eps_i error_i)) + (T + B) / 2 + eps (T - B)/2
            #   = (λ a0 + (T + B)/2)  + sum(λ eps_i error_i)  + eps_new (T - B)/2
            #
            # In summary:
            #   1) new bias = λ a0 + (T + B)/2
            #   2) old error terms: multiplied by λ
            #   3) new error term has weight (T - B)/2
            lambdas = -1 / (u * u)
            B = (1 / u - lambdas * u)
            T = (1 / l - lambdas * l)

            NEW_CONSTS = 0.5 * (T + B)
            NEW_COEFFS = 0.5 * (T - B)
        else:
            t_crit2 = u / 2.0
            mean_slope = (u.reciprocal() - l.reciprocal()) / (u - l)
            t_crit = (-mean_slope.reciprocal()).sqrt()

            if y_positive_constraint:
                t_opt = torch.max(t_crit, t_crit2 + 0.01)  # the + 0.01 is there to ensure strict positivity
            else:
                t_opt = t_crit

            lambdas = -t_opt.reciprocal().square()  # -1/t²
            X = l.reciprocal() - lambdas * l    # here we have that t_opt >= t_crit, and therefore we have to use L since we connect to that endpoint
            NEW_CONSTS = 0.5 * (t_opt.reciprocal() - lambdas * t_opt + X)
            NEW_COEFFS = 0.5 * (lambdas * t_opt - t_opt.reciprocal() + X)

        # INTERCEPT = (t_opt.reciprocal() - lambdas * t_opt)
        # isNan = torch.isnan(NEW_COEFFS[different_bool])
        # if isNan.any().item():
        #     print(l[different_bool][isNan])
        #     print(u[different_bool][isNan])
        #     a = 5

        assert not torch.isnan(NEW_COEFFS[different_bool]).any(), "Reciprocal: there are NaNs in the new COEFFS, pre-condition not met"
        # assert ((NEW_CONSTS - INTERCEPT)[different_bool] > -1-e5).all()
        assert (NEW_COEFFS[torch.logical_and(different_bool, NEW_COEFFS == NEW_COEFFS)]  >= -1e-4).all(), \
            "Reciprocal: there is a bad negative coeff: %f" % NEW_COEFFS[torch.logical_and(different_bool, NEW_COEFFS == NEW_COEFFS)].max().item()
        assert (lambdas[different_bool] <= 0).all(), "Reciprocal: There is a positive lambda %f" % lambdas[different_bool].max().item()

        # Step 1) new bias
        # for the equality case, we put the value 1/x
        # for the difference case, we put the center λ a0 + (T + B)/2
        transformed_x[0, equal_bool] = l[equal_bool].reciprocal()
        transformed_x[0, different_bool] = lambdas[different_bool] * self.zonotope_w[0, different_bool] + NEW_CONSTS[different_bool]

        # Step 2) updated error weights
        # for the equality case, there are no error terms
        # for the difference case, we multiply the current error terms by λ
        transformed_x[1:1 + self.num_error_terms, different_bool] = self.zonotope_w[1:, different_bool] * lambdas[different_bool]

        # Step 3) new error weights
        # Add the new errors terms efficiently using slicing
        indices = torch.arange(1 + self.num_error_terms, 1 + self.num_error_terms + n_new_error_terms)
        transformed_x[indices, has_new_error_term] = NEW_COEFFS[different_bool]

        return make_zonotope_new_weights_same_args(transformed_x, source_zonotope=self, clone=False)

    def recenter_zonotope_and_eliminate_error_term_ranges(self) -> "Zonotope":
        """ Eliminate the error_range_low and error_range_high variables
        by changing the variables so that all of them are in [-1, 1] e.g.
        if the k-th error term e_k has range [a, b], replace it by the term e_k_new
        such that c_k e_k = c_k ( (a + b) / 2 + (b - a) / 2 * e_k_new).

        We'll have center += c_k (a + b) / 2 and coefficient[k] = c_k (b - a) / 2 and del error_term_ranges """
        assert self.error_term_range_low is not None, "recenter_zonotope_and_eliminate_error_term_ranges: self.error_term_range_low is None"
        assert self.error_term_range_high is not None, "recenter_zonotope_and_eliminate_error_term_ranges: self.error_term_range_high is None"

        low, high = self.error_term_range_low, self.error_term_range_high
        center_range = (low + high) / 2  # n_error_terms
        radius_range = (high - low) / 2  # n_error_terms

        # Don't modify the error terms that are bound by a 1-norm or 2-norm, only those bound by an infinity-norm
        num_error_terms_to_skip = self.num_input_error_terms_special_norm
        error_terms = self.zonotope_w[1 + num_error_terms_to_skip:]  # (n_error_terms, num_words, embedding size)
        error_terms_reshaped = error_terms.permute(1, 2, 0)  # (num_words, embedding size, n_error_terms)

        center_coeffs = (error_terms_reshaped * center_range).sum(dim=-1)  # (num_words, embedding size)
        radius_coeffs = (error_terms_reshaped * radius_range)  # (num_words, embedding size, n_error_terms)

        new_zonotope_w = self.zonotope_w.clone()
        new_zonotope_w[0] += center_coeffs  # inplace, does not matter for now
        new_zonotope_w[1 + num_error_terms_to_skip:] = radius_coeffs.permute(2, 0, 1)  # (n_error_terms, num_words, embedding size)

        # We dont use make_zonotope_new_weights_same_args() on purpose, because we don't want to copy the error ranges to the new zonotope!
        return Zonotope(self.args, self.p, self.eps, self.perturbed_word_index, zonotope_w=new_zonotope_w)


def to_original_indices(new_indices: torch.Tensor, mapping_original_to_new_indices: torch.Tensor):
    """
    Example:
          new_indices     [3, 1, 3]
          mapping_original_to_new_indices:
                    [[0, 3, 1, 2],
                     [1, 2, 0, 3],
                     [0, 2, 3, 1]].T()
    Returns:
          [2, 2, 1]

    new_indices: N
    mapping_original_to_new_indices: E x N
    result: N
    """
    return mapping_original_to_new_indices.gather(0, new_indices.unsqueeze(0)).squeeze()


def get_slopes(linear_terms: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # linear_terms: (1 + num_error_terms, num_softmaxes)
    # indices: (num_softmaxes)
    num_indices, num_softmaxes = linear_terms.shape
    abs_linear_terms = linear_terms.abs()

    all_indices = torch.arange(num_indices, device=linear_terms.device)  # (1 + num_error_terms)
    all_indices = all_indices.unsqueeze(1).repeat(1, num_softmaxes)  # (1 + num_error_terms, num_softmaxes)

    # indices -> (num_softmaxes)
    # indices_good_shape -> (1 + num_error_terms, num_softmaxes)
    indices_good_shape = indices.unsqueeze(0).repeat(num_indices, 1)
    mask = ((all_indices < indices_good_shape) * 1.0) + ((all_indices >= indices_good_shape) * (-1.0))

    left_slope = (mask * abs_linear_terms).sum(dim=0)
    right_slope = left_slope.clone()
    right_slope += 2 * abs_linear_terms.gather(0, indices.unsqueeze(0)).squeeze()

    return left_slope, right_slope


def compute_width_heuristic(linear_terms: torch.Tensor, min_positions_per_term: torch.Tensor, chosen_indices: torch.Tensor) -> torch.Tensor:
    # all_positions: shape - (num_error_terms, num_equations)
    # indices:       shape - (num_equations)
    # linear_terms:  shape - (num_error_terms, num_equations)

    # |10 + 5x|   optimal x = -2
    # let's say the real value is x=3, so |10 + 5 * x| = |10 + 5*3| = 25
    # and we have that |5 * (3 - (-2))| = |5 * 5| = 25

    # |a + bx| = |a + b (x - x_min_for_that_term) + b(x_min_for_that_term)| = |b (x - x_min)|
    min_positions_chosen_points = min_positions_per_term.gather(0, chosen_indices.unsqueeze(0))  # (1, num_equations)
    distances = min_positions_per_term - min_positions_chosen_points  # (num_error_terms, num_equations)
    # Some error terms have width |a - 0e|, which leads to a ratio of NaN. since these terms don't matter, put 0 in the distance
    fillna(distances, 0)

    width_increase = (linear_terms * distances).abs().sum(dim=0)  # (num_error_terms, num_equations) -> (num_equations)
    return width_increase


def change_mid_to_only_have_removable_error_terms(current_indices_for_sorted: torch.Tensor, sorted_ratios: torch.Tensor,
                                                  sorted_linear_terms: torch.Tensor, sorting_indices: torch.Tensor, min_removable_error_term: int) -> None:
    # current_indices_for_sorted: shape - (num_equations)
    # sorted_ratios:              shape - (num_error_terms, num_equations)
    # sorted_linear_terms:        shape - (num_error_terms, num_equations)
    # sorting_indices:            shape - (num_error_terms, num_equations)
    num_error_terms, num_equations = sorted_linear_terms.shape

    # Search to the right
    current_indices_for_right = current_indices_for_sorted.clone()

    term_has_non_0_coeff = (sorted_linear_terms != 0)  # (num_error_terms, num_equations)

    term_has_inf_bound_original = torch.zeros_like(sorted_linear_terms, device=sorted_ratios.device, dtype=torch.bool)
    term_has_inf_bound_original[min_removable_error_term:] = True
    term_has_inf_bound = term_has_inf_bound_original.gather(0, sorting_indices)  # (num_error_terms, num_equations)

    all_indices = torch.arange(num_error_terms, device=sorted_linear_terms.device)  # (num_error_terms)
    all_indices = all_indices.unsqueeze(1).repeat(1, num_equations)  # (num_error_terms, num_equations)
    current_indices_for_sorted_good_shape = current_indices_for_sorted.unsqueeze(0).repeat(num_error_terms, 1)  # (num_softmaxes) -> (num_error_terms, num_equations)

    # MAX returns the first valid element
    term_is_to_the_right = (all_indices >= current_indices_for_sorted_good_shape) # (num_error_terms, num_equations)
    valid_terms_right = tensor_and(term_has_non_0_coeff, term_has_inf_bound, term_is_to_the_right)  # (num_error_terms, num_equations)
    is_valid_right, current_indices_for_right = valid_terms_right.max(dim=0)  # (num_equations)

    # MAX gives the first, here we want the last so we have to use reversing to fix this
    term_is_to_the_left = (all_indices <= current_indices_for_sorted_good_shape)  # (num_error_terms, num_equations)
    valid_terms_left = tensor_and(term_has_non_0_coeff, term_has_inf_bound, term_is_to_the_left)  # (num_error_terms, num_equations)
    valid_terms_left_reverse_order = torch.flip(valid_terms_left, [0])  # e.g. valid_terms_left[::-1, :]

    is_valid_left, current_indices_for_left_reversed = valid_terms_left_reverse_order.max(dim=0)  # (num_equations)
    current_indices_for_left = (num_error_terms - 1) - current_indices_for_left_reversed

    right_is_invalid = (is_valid_right == 0)
    left_is_invalid = (is_valid_left == 0)

    # Pick the right element
    values_left  = compute_width_heuristic(linear_terms=sorted_linear_terms, min_positions_per_term=sorted_ratios, chosen_indices=current_indices_for_left)
    values_right = compute_width_heuristic(linear_terms=sorted_linear_terms, min_positions_per_term=sorted_ratios, chosen_indices=current_indices_for_right)

    result = current_indices_for_sorted.clone()
    both_valid = tensor_and(torch.logical_not(left_is_invalid), torch.logical_not(right_is_invalid))

    result[right_is_invalid] = current_indices_for_left[right_is_invalid]   # Left better than right, because right is invalid
    result[left_is_invalid]  = current_indices_for_right[left_is_invalid]  # Right better than left, because left  is invalid
    result[torch.logical_and(both_valid, values_left <= values_right)] = current_indices_for_left[torch.logical_and(both_valid, values_left <= values_right)]
    result[torch.logical_and(both_valid, values_left > values_right)]  = current_indices_for_right[torch.logical_and(both_valid, values_left > values_right)]

    return result


def find_values_that_minimizes_width(constant_terms: torch.Tensor, linear_terms: torch.Tensor, min_removable_error_term: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Uses a binary search mechanism to find the optimal value fast
    Idea: call the constant terms 'a' and the linear terms 'b', then the optimal value will be one of the possible -a/b values
          if we order these -a/b values, we will see that the slope starts negative, monotonically increases and ends positive
          the minimum is at the position where before it the slope is negative and after it the slope is positive
          since the slopes increase monotonically, we can use binary search, and so we'll need O(log(error terms))
          computations of the slope to find the optimal value. Each computation of the slope is in O(error terms)
          so the total complexity per equation is O(error terms * log(error terms)). If there are N equality constraints
          then the total complexity will be O(N * error terms * log(error terms)) which is almost linear, so pretty good
          since N is quite small (it's equal to the sentence length in the case of Transformers)

    Each parameters has width |a + be|

    Parameters:
        - constant_terms: value a in the width
        - linear_terms: value b in the width
        - min_removable_error_term: disallow eliminating the first X terms. Can only remove terms whose index >= min_removable_error_term

    Returns:
        - optimal value for the variables (for each equation)
        - index of the variable (for each equation)
    """
    assert constant_terms.dtype == torch.float, "Constant term should be float"
    assert linear_terms.dtype == torch.float, "Linear term should be float"

    num_softmaxes = constant_terms.size(1)
    low = torch.zeros(num_softmaxes, dtype=torch.long, device=constant_terms.device)
    high = torch.ones(num_softmaxes, dtype=torch.long, device=constant_terms.device) * (constant_terms.size(0) - 1)

    ratios = -constant_terms / linear_terms  # (num_error_terms, num_softmaxes)

    sorted_ratios, indices_of_sorted = torch.sort(ratios, dim=0)  # sort per column (e.g. per softmax equation)
    sorted_linear_terms = linear_terms.gather(0, indices_of_sorted)

    converged = False
    while not converged:
        mid = (low + high) // 2
        slope_left, slope_right = get_slopes(sorted_linear_terms, mid)

        mask_too_much_left = (slope_left < 0) * (slope_right < 0)
        mask_too_much_right = (slope_left > 0) * (slope_right > 0)

        low[mask_too_much_left] = mid[mask_too_much_left] + 1
        high[mask_too_much_right] = mid[mask_too_much_right] - 1

        converged = ((mask_too_much_left.sum() + mask_too_much_right.sum()) == 0).item()

    if min_removable_error_term > 0:
        mid = change_mid_to_only_have_removable_error_terms(mid, sorted_ratios, sorted_linear_terms, indices_of_sorted, min_removable_error_term)

    # https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497/2?u=harokw
    optimal_values = sorted_ratios.gather(0, mid.view(1, -1))
    assert not torch.isnan(optimal_values).any(), "find_values_that_minimizes_width: Some optimal values are NaNs"

    indices_in_original_term_order = to_original_indices(mid, indices_of_sorted)

    return optimal_values, indices_in_original_term_order


def get_last_nonzero_index_per_row(x: torch.Tensor) -> torch.Tensor:
    """ Gets the index of the last non-zero value per row
    Input shape: (3, 5) -> 3 rows with 5 values
    Output shape: (3) -> 3 indexes (one per row) indicating the first non-zero value in that row
    """
    # Adapted from https://stackoverflow.com/a/60202801/3861529
    assert x.ndim == 2, "X should have two dimensions"

    row_size = x.shape[1]  # in our example, 5
    numbers_range = torch.arange(1, row_size + 1, device=x.device)  # in our example [1, 2, 3, 4, 5]

    # If the first row is [0, 3, 0, 7, 100], then (x != 0) = [0, 1, 0, 1, 0]
    # and therefore (x != 0) * numbers_range = [0, 2, 0, 4, 0]
    numbers_range_with_zeros = (x != 0) * numbers_range

    # the MAXIMUM value here will correspond to the first non-zero index in the original tensor
    last_non_zero_indices = torch.argmax(numbers_range_with_zeros, dim=1, keepdim=True)

    return last_non_zero_indices.t()


def add_equality_constraints(left_vars_w: torch.Tensor, right_vars_w: torch.tensor, z: Zonotope, min_removable_error_term: int, number_initial_terms_whose_range_should_not_change: int = 0) -> Tuple["Zonotope", torch.Tensor]:
    assert min_removable_error_term == number_initial_terms_whose_range_should_not_change, "add_equality constraints: invalid value"

    # The shape of the weight tensors should be (1 + num_error_terms, num_softmaxes)
    differences = left_vars_w - right_vars_w  # (1 + num_error_terms, num_softmaxes)
    pivot_error_index_per_equation = get_last_nonzero_index_per_row(differences.t())

    if not (pivot_error_index_per_equation >= 1 + min_removable_error_term).all():
        a = 5
    assert (pivot_error_index_per_equation >= 1 + min_removable_error_term).all(), "At least one equation has all allowed pivot terms equal to 0"

    # Let's figure out the logic
    # k -> denominator, left vars
    #   -> constant_terms, linear_terms_on_k
    #   =====> optimal_value_for_k, indices_removed_error_terms  (given constant_terms and linear_terms_on_k), all that follows is independ of k
    #   -> term_values -> constrained_zonotope
    #   -> indices_removed_error_terms later used to do the pivot in the RHS

    # Do index_select() -> use on left_vars_w[k] and differences[k]
    denominator = differences.gather(dim=0, index=pivot_error_index_per_equation).squeeze()
    assert (denominator != 0).all(), "add_equality_constraints: Some values for the denominator are 0!"

    ### Step 1: build the new zonotope
    # Find the a,b terms

    # left_vars_w: (error_dim, equation_dim)   left_vars_w[k]: (equation_dim)
    left_vars_at_pivot_terms = left_vars_w.gather(dim=0, index=pivot_error_index_per_equation).squeeze()

    constant_terms = left_vars_w - left_vars_at_pivot_terms * differences / denominator  # (1 + num_error_terms, num_softmaxes)
    linear_terms_on_k = differences / denominator  # (1 + num_error_terms, num_softmaxes)

    # Find the optimal value for the k-th term and deduce the values of all other terms

    # Important: we cannot allow pivoting on the 0-th term, since that term isn't an error term!
    # Doing so would imply the center is always 0, which generally we don't want to impose (e.g. the softmax sum must be 1)
    # and it just doesn't make any sense mathematically or logically

    # Therefore, we have to exclude the value for the 0-th term (the center) in the function below
    # and then increment the received optimal index
    optimal_value_for_k, indices_of_removed_error_terms = find_values_that_minimizes_width(constant_terms[1:], linear_terms_on_k[1:], min_removable_error_term)  # (num_softmaxes)
    indices_of_removed_error_terms += 1
    assert (indices_of_removed_error_terms != 0).all(), "Can't remove the constant term"

    terms_values = constant_terms + linear_terms_on_k * optimal_value_for_k  # (1 + num_error_terms, num_softmaxes)

    # Make zonotope (1 + num_error_terms, num_softmaxes, 1)
    constrained_zonotope = Zonotope(z.args, z.p, z.eps, z.perturbed_word_index, zonotope_w=terms_values.unsqueeze(2))

    ### Step 2: find the new error space for the constraints
    # Can't do more than 1, results becomes wrong
    error_term_low, error_term_high = iterated_alpha2(equality_eq=differences, iterations=3, number_initial_terms_whose_range_should_not_change=number_initial_terms_whose_range_should_not_change)
    assert (error_term_low <= error_term_high).all(), "add_equality_constraint: Error term high < error term low"

    constrained_zonotope.set_error_term_ranges(
        error_term_low[number_initial_terms_whose_range_should_not_change:],
        error_term_high[number_initial_terms_whose_range_should_not_change:]
    )  # (num_error_terms)

    return constrained_zonotope, indices_of_removed_error_terms


def add_inequality_constraints(vars_w: torch.Tensor, source_zonotope: Zonotope, lower_bound: Optional[float], upper_bound: Optional[float], iterations=5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    vars_w: Tensor representing the variables to be bounded, of the shape (num_vars, 1 + num error terms)
    source_zonotope: original Zonotope from which the variables originate
    lower_bound: lower bound. If None, no lower bound constraint is added.
    upper_bound: upper bound. If None, no upper bound constraint is added.
    iterations: number of alpha 2 iterations

    Returns
    -------
    Returns a new Zonotope where the vars respect the constraint
        lower_bound <= vars <= upper_bound
    lower_bound or upper_bound can be None if you don't want to set a bound on one of the sides.
    """
    assert lower_bound is not None or upper_bound is not None, "Illegal Arguments: both lower_bound and upper_bound is None"
    assert lower_bound < upper_bound, "Lower bound must be smaller than the upper bound"

    ##
    # x = c1 e1 + c2 e2 + ... + cn en <= 0
    #
    # Let's pick the term i and see how its range it updated. We get:
    #     ci ei <= -(c1 e1 + ... + c_i-1 e_i_1 + c_i+1 e_i+1 + ... + cn en)
    #           <= D
    # where that has the range [A, B]. When then have that ci ei <= D and therefore ei <= [A, B] / c
    # and therefore that ei <= max(A / c, B / c).
    #
    # Let the range after the division, while taking into acccount the sign of c, is [C, D]
    # Then, we have that ei <= [C, D]. What should be the range of ei?
    # We want to have [l, u] <= [C, D]. We can force u <= D and l <= C, but can we force u <= C without losing important information?

    ### Setup the inequalities
    vars_w = vars_w.permute(1, 0)  # New shape: (1 + num error terms, num vars)
    lower_bound_exprs, upper_bound_exprs = None, None

    # lower bound <= x   or in other words          lower bound - x <= 0
    if lower_bound is not None:
        lower_bound_exprs = -vars_w.clone()
        lower_bound_exprs[0, :] += lower_bound

    # x <= upper bound   or in other words          x - upper bound <= 0
    if upper_bound is not None:
        upper_bound_exprs = vars_w.clone()
        upper_bound_exprs[0, :] -= upper_bound

    expressions = torch.cat([x for x in [lower_bound_exprs, upper_bound_exprs] if x is not None], dim=1)

    ### Get the current values of the ranges
    current_error_range_low, current_error_range_high = source_zonotope.error_term_range_low, source_zonotope.error_term_range_high
    if current_error_range_low is None:
        current_error_range_low = -torch.ones(source_zonotope.num_error_terms, device=source_zonotope.device)
    if current_error_range_high is None:
        current_error_range_high = torch.ones(source_zonotope.num_error_terms, device=source_zonotope.device)

    ### Improve the ranges
    c, e = expressions[0], expressions[1:]  # (num_equations) and (num_error_terms, num_equations)
    for i in range(iterations):

        # Let's say that the range of e1 and e2 are originally [-1, 1] and the equations are
        #    x1 = 1 +  2 e1 + 3e2 <= 0
        #    x2 = 1 + -3 e1 - 1e2 <= 0
        # from which we deduce
        #     2 e1 <= -1 - 3e2       RHS has range [-4, 2]
        #    -3 e1 <= -1 + e2        RHS has range [-2, 0]
        # from which we deduce
        #     e1 <= -0.5  - 1.5 e2        RHS has range [-2, 1]
        #     e1 >= 1/3 - 1/3 e2          RHS has range [0, 2/3]
        #
        # and therefore the lower bound of e1 has to be at least -2
        #           and the upper bound of e1 has to be at most 2/3
        # so the new bound is [max(-1, -2), min(1, 2/3)] = [-1, 2/3]

        # the variable e contains the coeffs
        # center + c1 e1 + ... + ci ei + ... + cn en <= 0
        # ci ei <= -(center + c1 e1 + ... + c_(i-1) e_i-1 + c_i+1 e_i+1 + ... + cn en)
        # the variables low_without_self and high_without_self are the bounds of the RHS of the equation above

        ### Get the values of the RHS
        if current_error_range_low is None and current_error_range_high is None:
            low_without_self, high_without_self = -e.abs().sum(dim=0) + e.abs(), e.abs().sum(dim=0) - e.abs()  # (num_softmaxes) and (num_softmaxes)
        else:
            extreme1, extreme2 = e * current_error_range_low.unsqueeze(1), e * current_error_range_high.unsqueeze(1)
            smaller, bigger = torch.min(extreme1, extreme2), torch.max(extreme1, extreme2)
            low_without_self = smaller.sum(dim=0) - smaller
            high_without_self = bigger.sum(dim=0) - bigger

            # smaller = [1, 2, 3]
            # bigger = [4, 5, 7]

            # low_without_self    = [5, 4, 3]
            # bigger_without_self = [12, 11, 9]

            # smaller.sum() = 6
            # bigger.sum() = 16
            # smaller.sum() - smaller = 6 - [1, 2, 3] = [5, 4, 3]
            # bigger.sum() - bigger = 16 - [4, 5, 7] = [12, 11, 9]

        ### Compute the ranges after the division
        extreme1, extreme2 = -(c + low_without_self) / e, -(c + high_without_self) / e  # (num_error_terms, num_softmaxes)
        lower_bound_rest = torch.min(extreme1, extreme2)
        upper_bound_rest = torch.max(extreme1, extreme2)

        # If in the equation we had  5e_n <= val, this becomes e_n <= val /  5, then we'll deduce upper_bound(e_n) <= val / 5
        # If in the equation we had -5e_n <= val, this becomes e_n >= val / -5, then we'll deduce lower_bound(e_n) >= val / -5
        # Note: the sign change is very important there.
        # The variables lower_bound_rest and upper_bound_rest are what contain the range of the val / coeffs on the RHS.
        # Whether the minimum or the maximum of e_i is modified depends on the whether the coefficient was positive or negative
        #
        # e > 0:    upper bound := torch.min(upper bound, new upper bound)
        # e < 0:    lower bound := torch.max(lower bound, new lower bound)
        # e = 0:    no change, variable not present

        # 5 e_n - val <= 0
        # e_n - val / 5 <= 0
        # upper_bound(e_n) - lower_bound(val / 5) <= 0
        #

        # Shape: (number error terms, number expressions)
        number_expressions = expressions.size(1)
        updated_error_range_low_all = current_error_range_low.unsqueeze(1).repeat(1, number_expressions)
        updated_error_range_high_all = current_error_range_high.unsqueeze(1).repeat(1, number_expressions)

        updated_error_range_high_all[e > 0] = torch.min(updated_error_range_high_all[e > 0], upper_bound_rest[e > 0])
        updated_error_range_low_all[e < 0] = torch.max(updated_error_range_low_all[e < 0], lower_bound_rest[e < 0])

        updated_error_range_low, _ = torch.max(updated_error_range_low_all, dim=1)
        updated_error_range_high, _ = torch.min(updated_error_range_high_all, dim=1)

        assert (updated_error_range_low <= updated_error_range_high).all(), "Updated error range has low > high"

        should_stop = torch.allclose(updated_error_range_high, current_error_range_high) and \
                      torch.allclose(updated_error_range_low, current_error_range_low)

        current_error_range_low, current_error_range_high = updated_error_range_low, updated_error_range_high
        if should_stop:
            print("add_inequality_constraints: Converged at the end of iteration %i" % i)
            break

    return current_error_range_low, current_error_range_high


def iterated_alpha2(equality_eq: torch.Tensor, iterations: int = 1,
                    current_error_range_low: torch.Tensor = None,
                    current_error_range_high: torch.Tensor = None,
                    number_initial_terms_whose_range_should_not_change: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    for i in range(iterations):
        new_error_range_low, new_error_range_high = alpha2(equality_eq, current_error_range_low, current_error_range_high, number_initial_terms_whose_range_should_not_change)
        assert (new_error_range_low <= new_error_range_high).all(), "iterated_alpha2: high < low"

        if current_error_range_low is not None and torch.allclose(new_error_range_low, current_error_range_low) and torch.allclose(new_error_range_high, current_error_range_high):
            # print("iterated alpha2: Converged at the end of iteration %i" % i)
            break

        if i == 2:
            a = 5

        current_error_range_low, current_error_range_high = new_error_range_low, new_error_range_high

    return new_error_range_low, new_error_range_high


def alpha2(equality_eq: torch.Tensor,
           current_error_range_low: torch.Tensor = None,
           current_error_range_high: torch.Tensor = None,
           number_initial_terms_whose_range_should_not_change: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses the alpha2 method to update the error ranges
    Parameters
    ----------
    equality_eq: Tensor with shape(1 + num_error_terms, number_equations)
    current_error_range_low
    current_error_range_high
    number_initial_terms_whose_range_should_not_change: useful if some terms are bound by non-zero

    Returns
    -------
    The new error ranges (low and high tensor)
    """
    c, e = -equality_eq[0], equality_eq[1:]  # (num_softmaxes) and (num_error_terms, num_softmaxes)
    if current_error_range_low is None and current_error_range_high is None:
        low, high = -e.abs().sum(dim=0) + e.abs(), e.abs().sum(dim=0) - e.abs()     # (num_softmaxes) and (num_softmaxes)
    else:
        assert (current_error_range_low <= current_error_range_high).all(), "alpha2: high < low"
        extreme1, extreme2 = e * current_error_range_low.unsqueeze(1), e * current_error_range_high.unsqueeze(1)
        smaller, bigger = torch.min(extreme1, extreme2), torch.max(extreme1, extreme2)
        low = smaller.sum(dim=0) + smaller
        high = bigger.sum(dim=0) - bigger

    # Example: range y = [-2, 4] and c = 3    y + 3x = c = 3 -> x = 3 - y = 3 - [-2, 4] = [-1, 5] -> [-1, 5] INTERSECT [-1, 1] = [-1, 1]
    # Example: range y = [-4, 4] and c 4      y + 3x = c = 4 -> x = 4 - y = 4 - [-4, 4] = [0, 8] -> [0, 8] INTERSECT [-1, 1] = [0, 1]
    # Example: y + kx = c   kx = c - y   x = c - y / k = (c - [ly, uy]) / k = [c - uy, c - ly] / k
    # if     k >= 0, then x in [c - uy, c - ly] / k = [(c - uy) / k, (c - ly) / k]
    # else if k < 0, then x in [c - uy, c - ly] / k = [(c - ly) / k, (c - uy) / k]
    bound_with_l, bound_with_u = (c - low) / e, (c - high) / e  # (num_error_terms, num_softmaxes)

    all_error_term_low, all_error_term_high = torch.empty_like(bound_with_l), torch.empty_like(bound_with_u)
    all_error_term_high[e >= 0] = bound_with_l[e >= 0]
    all_error_term_low[e >= 0] = bound_with_u[e >= 0]
    all_error_term_high[e < 0] = bound_with_u[e < 0]
    all_error_term_low[e < 0] = bound_with_l[e < 0]

    if all_error_term_high.min() < 1:
        a = 5

    all_error_term_high = all_error_term_high.clamp(min=-1, max=1)
    all_error_term_low = all_error_term_low.clamp(min=-1, max=-1)

    fillna(all_error_term_high, 1)  # (num_error_terms, num_softmaxes)
    fillna(all_error_term_low, -1)  # (num_error_terms, num_softmaxes)

    error_term_high, _ = torch.min(all_error_term_high, dim=1)
    error_term_low, _ = torch.max(all_error_term_low, dim=1)

    if current_error_range_low is not None:
        assert current_error_range_high is not None, "current_error_range_low is not None but current_error_range_high is None"
        error_term_high = torch.min(error_term_high, current_error_range_high)
        error_term_low = torch.min(error_term_low, current_error_range_low)

    error_term_low[:number_initial_terms_whose_range_should_not_change] = -1.0
    error_term_high[:number_initial_terms_whose_range_should_not_change] = 1.0

    assert (error_term_low <= error_term_high).all(), \
        "Error: some error ranges are empty. Low: %s High: %s" % (error_term_low, error_term_high)

    return error_term_low, error_term_high


def process_values(input_zonotope_w: torch.Tensor, source_zonotope: "Zonotope", A: int, num_rows: int, num_values: int, keep_intermediate_zonotopes: bool) -> Tuple[torch.Tensor, int]:
    ### Step 1: compute all the xj - xi
    # Self shape: (A, 1 + num_error_terms, num_rows, num_values)
    # Middle shape: (A, 1 + num_error_terms, num_rows, num_values, num_values)
    # such that w[A, E, N, i, j] = rowN_j - rowN_i
    vals_w = input_zonotope_w.unsqueeze(-1)  # (A, 1 + num_error_terms, num_rows, num_values, 1)
    vals_w_repeated = vals_w.repeat(1, 1, 1, 1, num_values)  # (A, 1 + num_error_terms, num_rows, num_values, num_values)

    diffs_w = vals_w_repeated.transpose(3, 4) - vals_w_repeated

    if not keep_intermediate_zonotopes:
        del vals_w_repeated; cleanup_memory()

    # End shape: (1 + num_error_terms, A * num_rows, num_values * num_values)?
    diffs_w_reshaped = diffs_w.permute(1, 0, 2, 3, 4).reshape(1 + source_zonotope.num_error_terms, A * num_rows, num_values * num_values)

    zonotope_diffs = make_zonotope_new_weights_same_args(diffs_w_reshaped, source_zonotope, clone=False)
    cleanup_memory()

    ### Step 2: compute all the exp(xj - xi)
    # End shape: (1 + num_error_terms, A * num_rows, num_values * num_values)?
    incomplete_diffs_exp_w, exp_new_error_term = zonotope_diffs.exp_minimal_area(return_raw_tensors_separately=True,
                                                                                 allow_modifying_input_zonotope=True)
    # full_exp_zonotope = zonotope_diffs.exp_mark(return_raw_tensors_separately=False)

    ### Step 3: sum all the exp(xj - xi) for each i
    # Start shape: (1 + num_error_terms, A * num_rows, num_values * num_values)?
    # Mid shape: (1 + num_error_terms, A, num_rows, num_values, num_values)
    # End shape: (A, 1 + num_error_terms, num_rows, num_values)
    diffs_exp_w_reshaped = incomplete_diffs_exp_w.reshape(-1, A, num_rows, num_values, num_values)

    if not keep_intermediate_zonotopes:
        del incomplete_diffs_exp_w

    sum_exp_diffs_w = diffs_exp_w_reshaped.sum(dim=-1).permute(1, 0, 2, 3)  # (A, 1 + num_error_terms, num_rows, num_values)

    if not keep_intermediate_zonotopes:
        del diffs_exp_w_reshaped

    # (A * num_rows, num_values * num_values) -> (A * num_rows, num_values)
    new_error_terms_collapsed = exp_new_error_term.reshape(A, num_rows, num_values, num_values).sum(dim=-1)

    if not keep_intermediate_zonotopes:
        del exp_new_error_term

    return sum_exp_diffs_w, new_error_terms_collapsed
