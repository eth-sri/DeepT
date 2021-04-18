import unittest
from random import randint

import torch
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F

from typing import Tuple, Callable, Any, Union, List, Iterable

from Verifiers.Zonotope import Zonotope, make_zonotope_new_weights_same_args, get_last_nonzero_index_per_row

from math import exp, tanh


def unzip(x):
    return list(zip(*x))

def check_zonotope_positivity(zonotope: Zonotope):
    l, u = zonotope.concretize()
    rel_problem_lower_bound = -l / abs(u)
    assert (l <= u).all()
    assert (l >= 0).all(), "Lower bound of zonotope <= 0.  l.min() = %.12f" % l.min().item()
    assert (rel_problem_lower_bound < 1e-6).all(), "Bad lower bound.  Val = %.12f" % rel_problem_lower_bound.min().item()


def check_zonotope_in_0_1_range(zonotope: Zonotope):
    l, u = zonotope.concretize()
    assert (l <= u).all()
    # TODO
    # assert (l > 0).all(), "Lower bound of zonotope <= 0  (min = %f)" % l.min().item()
    # assert (u <= 1).all(), "Upper bound of zonotope >= 0  (max = %f)" % u.max().item()


"""
Testing suite for transformers
Unit tests to quickly verify that the transformers have no obvious bugs
"""

INFINITY = float("inf")

# norm = inf    dual norm = 1
# norm = 1      dual norm = inf
# norm = 2      dual norm = 2


class Args:
    def __init__(self, num_input_error_terms):
        self.perturbed_words = 1
        self.device = 'cpu'
        self.timeout = 1.0
        self.num_processes = 6
        self.num_input_error_terms = num_input_error_terms


class ZonotopeTest(unittest.TestCase):
    def setUp(self):
        pass

    def make_zonotope(self, zonotope_w, p=500):
        if zonotope_w.ndim == 4:
            num_errors = zonotope_w.size(1) - 1
        else:
            num_errors = zonotope_w.size(0) - 1
        return Zonotope(args=Args(num_errors), p=p, eps=1.0, perturbed_word_index=0, zonotope_w=zonotope_w)

    def assert_tensors_equal(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue(torch.equal(a, b))

    def assert_tensors_almost_equal(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue(torch.allclose(a, b))

    def assert_tensor_is_smaller(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue((a - 1e-5 <= b).all())

    def assert_tensor_is_bigger(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue((a + 1e-5 >= b).all())

    def test_create_zonotope_from_value(self):
        value = torch.tensor(
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]]
        )
        zonotope = Zonotope(Args(), p=INFINITY, eps=1., perturbed_word_index=1, value=value)

        expected_weights = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]],
        ])

        self.assert_tensors_almost_equal(zonotope.zonotope_w, expected_weights)

    def test_concretize_no_errors(self):
        # First word was perturbed
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w)
        l, u = zonotope.concretize()
        self.assert_tensors_equal(l, u)
        self.assert_tensors_equal(l, zonotope_w[0])

    def test_concretize_with_error_first_word_p_inf(self):
        # First word was perturbed
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w, p=INFINITY)
        l, u = zonotope.concretize()

        expected_lower = torch.tensor(
            [[-2.0, 0.0, 2.0],
             [3.0, 4.0, 5.0]]
        )
        expected_higher = torch.tensor(
            [[4.0, 4.0, 4.0],
             [3.0, 4.0, 5.0]]
        )
        self.assert_tensors_equal(l, expected_lower)
        self.assert_tensors_equal(u, expected_higher)

    def test_concretize_with_error_first_word_p_1(self):
        # First word was perturbed
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w, p=1)
        l, u = zonotope.concretize()

        expected_lower = torch.tensor(
            [[0.0, 1.0, 2.0],
             [3.0, 4.0, 5.0]]
        )
        expected_higher = torch.tensor(
            [[2.0, 3.0, 4.0],
             [3.0, 4.0, 5.0]]
        )
        self.assert_tensors_almost_equal(l, expected_lower)
        self.assert_tensors_almost_equal(u, expected_higher)

    def test_concretize_with_error_first_word_p_2(self):
        # First word was perturbed
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w, p=2)
        l, u = zonotope.concretize()

        expected_lower = torch.tensor(
            [[1.0 - 3**0.5, 2.0 - 2**0.5, 2.0],
             [3.0, 4.0, 5.0]]
        )
        expected_higher = torch.tensor(
            [[1.0 + 3**0.5, 2.0 + 2**0.5, 4.0],
             [3.0, 4.0, 5.0]]
        )
        self.assert_tensors_almost_equal(l, expected_lower)
        self.assert_tensors_almost_equal(u, expected_higher)

    def test_relu(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.relu(lambdas=None)

        # First: [-1, 3], λ = 3/4 = 0.75
        #   Bias: λ a0 - λ * l / 2
        #   Error weights: λ w_i
        #   New error weights: - λ * l / 2
        # Second: positive, should stay unchanged
        # Third: negative, should all be 0
        lamb = 0.75
        expected_weights = torch.tensor([
            [[lamb * 1.0 - lamb * (-1) / 2, 2.0, 0.0]],
            [[lamb * 1.0                  , 1.0, 0.0]],
            [[lamb * 1.0                  , 0.0, 0.0]],
            [[0.0                         , 0.0, 0.0]],
            [[- lamb * (-1) / 2, 0.0, 0.0]],
        ])

        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assertTrue((trans_l <= torch.relu(l)).all())  # Ensure soundness
        self.assertTrue((trans_u >= torch.relu(u)).all())  # Ensure soundness

        # Can't use equality because the value might go below
        # For the relu, we don't always have that should have f(bounds(z)) = bounds(f(z))
        # The relation shold except in the overlapping case
        non_overlapping = (l >= 0).float() * (u <= 0).float()
        self.assert_tensors_almost_equal(torch.exp(l) * non_overlapping, trans_l * non_overlapping)
        self.assert_tensors_almost_equal(torch.exp(u) * non_overlapping, trans_u * non_overlapping)

    def test_exp(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])
        # Ranges: [-1, 3], [1, 3], [-3, -3]

        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.exp()

        # B = (f(u) - λu) and T = (f(l) - λl)
        # y = λx + (T + B) / 2 + eps (T - B) / 2
        # y = (λ a0 + (T + B)/2)  + sum(λ eps_i error_i)  + eps_new (T - B)/2
        #
        #   Bias: λ a0 + (T + B)/2
        #   Error weights: λ w_i
        #   New error weights: (T - B)/2
        lamb1 = exp(-1)
        lamb2 = exp(1)
        B1, T1 = exp(-1) - lamb1 * (-1), exp(3) - lamb1 * 3
        B2, T2 = exp(1) - lamb2 * 1, exp(3) - lamb2 * 3,
        expected_weights = torch.tensor([
            [[lamb1 * 1.0 + (T1 + B1) / 2, lamb2 * 2.0 + (T2 + B2) / 2        , exp(-3.0)]],
            [[lamb1 * 1.0                   , lamb2 * 1.0, 0.0]],
            [[lamb1 * 1.0                   , 0.0, 0.0]],
            [[0.0                           , 0.0, 0.0]],
            [[(T1 - B1) / 2                 , 0.0, 0.0]],
            [[0.0                  , (T2 - B2) / 2, 0.0]],
        ])
        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, torch.exp(l))  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, torch.exp(u))  # Ensure soundness

        # For the exp, we should have f(bounds(z)) = bounds(f(z))
        self.assert_tensors_almost_equal(torch.exp(l), trans_l)
        self.assert_tensors_almost_equal(torch.exp(u), trans_u)

    def test_reciprocal(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[3.0, 3.0, 3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])
        # Ranges: [1, 5], [2, 4], [3, 3]

        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.reciprocal()

        # B = (f(u) - λu) and T = (f(l) - λl)
        # y = λx + (T + B) / 2 + eps (T - B) / 2
        # y = (λ a0 + (T + B)/2)  + sum(λ eps_i error_i)  + eps_new (T - B)/2
        #
        #   Bias: λ a0 + (T + B)/2
        #   Error weights: λ w_i
        #   New error weights: (T - B)/2
        lamb1 = -1 / 5 ** 2
        lamb2 = -1 / 4 ** 2
        B1, T1 = 1. / 5 - lamb1 * 5, 1. / 1 - lamb1 * 1
        B2, T2 = 1. / 4 - lamb2 * 4, 1. / 2 - lamb2 * 2
        expected_weights = torch.tensor([
            [[lamb1 * 3.0 + (T1 + B1) / 2, lamb2 * 3.0 + (T2 + B2) / 2        , 1. / 3]],
            [[lamb1 * 1.0                   , lamb2 * 1.0, 0.0]],
            [[lamb1 * 1.0                   , 0.0, 0.0]],
            [[0.0                           , 0.0, 0.0]],
            [[(T1 - B1) / 2                 , 0.0, 0.0]],
            [[0.0                  , (T2 - B2) / 2, 0.0]],
        ])

        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, torch.reciprocal(l))  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, torch.reciprocal(u))  # Ensure soundness

        self.assert_tensors_almost_equal(torch.reciprocal(l), trans_u)  # largest value obtained by doing 1 / l
        self.assert_tensors_almost_equal(torch.reciprocal(u), trans_l)  # smallest value obtained by doing 1 / u

    def test_reduce_error_terms(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[100.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[10.0, -20.0, 10.0]],
            [[1.0, -30.0, 0.0]],
            [[20.0, 0.0, 20.0]],
            [[100.0, 0.0, 0.0]],
        ])
        # Ranges: [-1, 3], [1, 3], [-3, -3]

        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.remove_error_terms_box(num_terms_to_reduce=2)

        expected_weights = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[100.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[31.0, 50.0, 30.0]],
        ])
        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensors_almost_equal(trans_l, l)  # Ensure soundness
        self.assert_tensors_almost_equal(trans_u, u)  # Ensure soundness

    def test_subtract_mean(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([[
            [[1.0, 2.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[1.0, -2.0, 1.0]],
            [[1.0, -3.0, 0.0]],
            [[2.0, 0.0, 2.0]],
            [[1.0, 0.0, 0.0]],
        ]])
        # Ranges: [-1, 3], [1, 3], [-3, -3]

        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.subtract_concretized_mean_from_bias()

        expected_weights = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[100.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[31.0, 50.0, 30.0]],
        ])  # TODO: these are bullshit weights
        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensors_almost_equal(trans_l, l)  # Ensure soundness
        self.assert_tensors_almost_equal(trans_u, u)  # Ensure soundness

    def test_last_non_zero_index(self):
        tt = torch.tensor
        self.assert_tensors_equal(get_last_nonzero_index_per_row(tt([[1, 2, 3]])), tt([[2]]))
        self.assert_tensors_equal(get_last_nonzero_index_per_row(tt([[100, 2, 0]])), tt([[1]]))
        self.assert_tensors_equal(get_last_nonzero_index_per_row(tt([[300, 0, 0]])), tt([[0]]))

        self.assert_tensors_equal(get_last_nonzero_index_per_row(tt([[300, 2, 0],
                                                                     [1, 0, 5000],
                                                                     [5000, 0, 0]])),
                                  tt([[1, 2, 0]]))

    def assert_bounds_correct(self, l: torch.Tensor, output: torch.Tensor, u: torch.Tensor, rel_eps=1e-7):
        error_lower_bound = ((l - output) / abs(output + 1e-12))
        error_upper_bound = ((output - u) / abs(output + 1e-12))
        assert (error_lower_bound < rel_eps).all(), "Max rel violation: %s    Output size for violations: %s    Lower bound: %s" \
                                          % (error_lower_bound.max().item(), output[output < l], l[output < l])
        assert (error_upper_bound < rel_eps).all(), "Max rel violation: %s    Output size for violations: %s    Upper bound: %s" \
                                          % (error_upper_bound.max().item(), output[output > u], u[output > u])

        # EPS = 1e-5
        # assert (l - EPS <= output).all(), "Max violation: %s    Output size for violations: %s" % ((l - output)[output < l], output[output < l])
        # assert (output <= u + EPS).all(), "Max violation: %s    Output size for violations: %s" % ((output - u)[output > u], output[output > u])

    def check_values_and_interesting_points(self, input_zonotopes: List[Zonotope], concrete_fn: Callable[[torch.Tensor], torch.Tensor],
                                            deduced_l_output: torch.Tensor, deduced_u_output: torch.Tensor,
                                            do_asserts=True, rel_eps=1e-7) -> Tuple[float, float, float, float]:
        l_s, u_s = list(zip(*[input_zonotope.concretize() for input_zonotope in input_zonotopes]))

        if input_zonotopes[0].zonotope_w.ndim == 3:
            centers = [input_zonotope.zonotope_w[0] for input_zonotope in input_zonotopes]
        else:
            centers = [input_zonotope.zonotope_w[:, 0] for input_zonotope in input_zonotopes]

        max_l_error, max_u_error = None, None
        max_l_error_rel, max_u_error_rel = None, None

        for i in range(500):
            # NOTE: we CANNOT pick l and u as potential sampled_input, since they MIGHT fall outside the zonotope
            # I fell into this trap once and spent quite some time puzzling over why my asserts failed, do not repeat this mistake
            # Instead, if I want to test some error cases, pick an error values vector of all -1 or all +1
            if i == 0 and input_zonotopes[0].p > 10:  # Can only test these extremes for the inf norm
                sampled_inputs = [
                    input_zonotope.compute_val_given_error_terms(
                        -torch.ones(input_zonotope.num_error_terms, device=input_zonotope.device)
                    )
                    for input_zonotope in input_zonotopes
                ]
            elif i == 1 and input_zonotopes[0].p > 10: # Can only test these extremes for the inf norm
                sampled_inputs = [
                    input_zonotope.compute_val_given_error_terms(
                        torch.ones(input_zonotope.num_error_terms, device=input_zonotope.device)
                    )
                    for input_zonotope in input_zonotopes
                ]
            elif i == 2:
                sampled_inputs = centers
            else:
                sampled_inputs, error_terms = unzip([input_zonotope.sample_point() for input_zonotope in input_zonotopes])

            # Check that the sampled inputs are within the bounds
            for i in range(len(sampled_inputs)):
                margin_eps = 1e-6 if i > 3 and (error_terms[i].abs() == 1).all().item() else 0
                assert (l_s[i] <= sampled_inputs[i] + margin_eps).all()
                assert (sampled_inputs[i] <= u_s[i] + margin_eps).all()

            output = concrete_fn(*sampled_inputs)
            if do_asserts:
                self.assert_bounds_correct(deduced_l_output, output, deduced_u_output, rel_eps)

            new_max_u_error_rel = -((deduced_u_output - output) / abs(output)).min()  # u = 4    output = 10   ->  6/10
            new_max_l_error_rel = ((deduced_l_output - output) / abs(output)).max()   # l = 6    output = 2    ->  4/2
            if max_u_error_rel is None or new_max_u_error_rel > max_u_error_rel: max_u_error_rel = new_max_u_error_rel
            if max_l_error_rel is None or new_max_l_error_rel > max_l_error_rel: max_l_error_rel = new_max_l_error_rel

            new_max_u_error = -(deduced_u_output - output).min()  # u = 4    output = 10   ->  6/10
            new_max_l_error = (deduced_l_output - output).max()  # l = 6    output = 2    ->  4/2
            if max_u_error is None or new_max_u_error > max_u_error: max_u_error = new_max_u_error
            if max_l_error is None or new_max_l_error > max_l_error: max_l_error = new_max_l_error

        return max_l_error.item(), max_u_error.item(), max_l_error_rel.item(), max_u_error_rel.item()

    def verify_sanity(self,
                      generate_zonotope_weights_fn: Callable[[], Union[torch.Tensor, Iterable[torch.Tensor]]],
                      concrete_point_fn: Callable[[torch.Tensor], torch.Tensor],
                      zonotope_fn: Callable[[Zonotope], Zonotope],
                      iterations=30, do_asserts=True, max_rel_excess_allowed=1e-7,
                      check_fn_on_output_zonotope: Callable = None,
                      check_fn_on_input_and_output_zonotope: Callable = None,
                      collect_stats_on_output_zonotopes: Callable = None,
                      make_zonotope_fn: Callable[[torch.Tensor], Zonotope] = None
                      ) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, Any]]:
        """
        Does 'iterations' iterations of the following process:
            - create the input zonotopes
            - get the output zonotope by applying zonotope_fn on the input zonotopes
            - it can optionally do checks on the output zonotope
            - it can optionally do checks on the output zonotope and input zonotopes
            - it then samples 500 points from the input zonotopes, applies the real operation on them
              and checks if they are within the bounds of the output zonotope
        """
        worst_l, worst_u = None, None
        worst_l_rel, worst_u_rel = None, None

        stats_state = None
        stats_output = None
        make_zonotope_fn = make_zonotope_fn or self.make_zonotope

        for i in tqdm(range(iterations)):
            weights = generate_zonotope_weights_fn()

            if type(weights) == tuple:
                input_zonotopes = [make_zonotope_fn(z_weights) for z_weights in weights]
            else:
                input_zonotopes = [make_zonotope_fn(weights)]

            output_zonotope = zonotope_fn(*input_zonotopes)

            if check_fn_on_output_zonotope is not None:
                check_fn_on_output_zonotope(output_zonotope)

            if check_fn_on_input_and_output_zonotope is not None:
                check_fn_on_input_and_output_zonotope(*input_zonotopes, output_zonotope)

            if collect_stats_on_output_zonotopes is not None:
                stats_state, stats_output = collect_stats_on_output_zonotopes(output_zonotope, stats_state)

            zonotope_l, zonotope_u = output_zonotope.concretize()
            max_l_error, max_u_error, max_l_error_rel, max_u_error_rel = self.check_values_and_interesting_points(input_zonotopes, concrete_point_fn, zonotope_l, zonotope_u, do_asserts, max_rel_excess_allowed)

            if worst_l is None or worst_l < max_l_error: worst_l = max_l_error
            if worst_u is None or worst_u < max_u_error: worst_u = max_u_error
            if worst_l_rel is None or worst_l_rel < max_l_error_rel: worst_l_rel = max_l_error_rel
            if worst_u_rel is None or worst_u_rel < max_u_error_rel: worst_u_rel = max_u_error_rel

        if collect_stats_on_output_zonotopes is None:
            return worst_l, worst_u, worst_l_rel, worst_u_rel
        else:
            return worst_l, worst_u, worst_l_rel, worst_u_rel, stats_output

    def test_compare_new_old_dotproduct_repeatedly(self):
        for i in tqdm(range(1000)):
            a = torch.randn(3, 30, 5, 10)

            b = torch.randn(3, 30, 3, 10)
            b[15:] = 0.0

            b_narrowed = b.clone()[:15]

            a_z, b_z, b_narrowed_z  = self.make_zonotope(a), self.make_zonotope(b), self.make_zonotope(b_narrowed)
            results_new = a_z.dot_product_precise(b_narrowed_z, zonotopes_can_have_different_number_noise_symbols=True)
            result_old = a_z.dot_product_precise(b_z, zonotopes_can_have_different_number_noise_symbols=False)

            self.assert_tensors_equal(results_new.zonotope_w, result_old.zonotope_w)

            # TODO: this should be true, but only for norm p = 1 and norm p = 2
            # for norm p = inf the old implementation should occasionally be a bit more precise
            # result_updated = a_z.dot_product_new(b_z)
            # l_updated, u_updated = result_updated.concretize()
            # l_new, u_new = results_new.concretize()
            # l_old, u_old = result_old.concretize()
            # self.assert_tensor_is_smaller(l_old, l_updated)
            # self.assert_tensor_is_smaller(l_new, l_updated)
            # self.assert_tensor_is_smaller(u_updated, u_new)
            # self.assert_tensor_is_smaller(u_updated, u_old)

    def test_box_reduction_repeatedly(self):
        def fun_concrete(x):
            return x

        def fun_zonotope(x: Zonotope):
            num_error_terms = x.num_error_terms
            random_smaller_amount = randint(1, num_error_terms - 1)
            return x.reduce_num_error_terms_box(random_smaller_amount)

        def make_zonotope():
            return torch.randn(200, 5, 10)

        def assert_good_rel_error(lower, higher, input_val, fp_rel_error_tolerance):
            """ Input val is either lower or higher """
            excess = lower - higher
            rel_excess = excess / abs(input_val)
            assert (rel_excess <= fp_rel_error_tolerance).all()

        def check_input_contained_within_output(input_zonotope: Zonotope, output_zonotope: Zonotope):
            l_input, u_input = input_zonotope.concretize()
            l_output, u_output = output_zonotope.concretize()

            eps = 1e-6
            assert_good_rel_error(l_output, l_input, input_val=l_input, fp_rel_error_tolerance=eps)
            assert_good_rel_error(l_input, u_input, input_val=l_input, fp_rel_error_tolerance=eps)
            assert_good_rel_error(u_input, u_output, input_val=u_input, fp_rel_error_tolerance=eps)
            assert_good_rel_error(l_output, u_output, input_val=l_output, fp_rel_error_tolerance=eps)

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=make_zonotope,
            concrete_point_fn=fun_concrete,
            zonotope_fn=fun_zonotope,
            iterations=50,
            do_asserts=True,
            max_rel_excess_allowed=1e-7,
            check_fn_on_input_and_output_zonotope=check_input_contained_within_output
        )

    def test_tanh_repeatedly(self):
        def fun_concrete(x: torch.Tensor):
            return x.tanh()

        def fun_zonotope(x: Zonotope):
            return x.tanh()

        def make_zonotopes():
            a = torch.randn(5, 5, 10)
            return a

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=make_zonotopes,
            concrete_point_fn=fun_concrete,
            zonotope_fn=fun_zonotope,
            iterations=50,
            do_asserts=False,
            max_rel_excess_allowed=1e-6
        )
        print(worst_l, worst_u, worst_l_rel, worst_u_rel )

    def test_dot_product_then_dense_repeatedly(self):
        dense_layer  = torch.randn(10, 7)

        def fun_concrete(x: torch.Tensor, y: torch.Tensor):
            result = x.bmm(y.transpose(-1, -2))   # 3, 5, 7
            return F.linear(result, dense_layer)

        def fun_zonotope(x: Zonotope, y: Zonotope):
            # return x.dot_product_new(y)
            # return x.dot_product_old(y, new_implementation=True)
            result = x.dot_product(y, new_implementation=False)
            return result.matmul(dense_layer)

        def make_zonotopes():
            a = torch.randn(3, 30, 5, 10)
            b = torch.randn(3, 30, 7, 10)
            return a, b

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=make_zonotopes,
            concrete_point_fn=fun_concrete,
            zonotope_fn=fun_zonotope,
            iterations=50,
            do_asserts=True,
            max_rel_excess_allowed=1e-7
        )

    def test_dot_product_then_dense_repeatedly_2_norm_and_inf_norm(self):
        dense_layer = torch.randn(10, 7)

        def fun_concrete(x: torch.Tensor, y: torch.Tensor):
            result = x.bmm(y.transpose(-1, -2))   # 3, 5, 7
            return F.linear(result, dense_layer)

        def fun_zonotope(x: Zonotope, y: Zonotope):
            # return x.dot_product_new(y)
            # return x.dot_product_old(y, new_implementation=True)
            result = x.dot_product(y, new_implementation=False)
            return result.matmul(dense_layer)

        def make_zonotopes_weights():
            a = torch.randn(3, 30, 5, 10)
            b = torch.randn(3, 30, 7, 10)
            return a, b

        def make_zonotope_fn(weights: torch.Tensor):
            # 10 first terms are 1-bounded
            args = Args(num_input_error_terms=30)
            zonotope = Zonotope(args=args, p=2.0, eps=1.0, perturbed_word_index=0, zonotope_w=weights)
            return zonotope

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=make_zonotopes_weights,
            make_zonotope_fn=make_zonotope_fn,
            concrete_point_fn=fun_concrete,
            zonotope_fn=fun_zonotope,
            iterations=50,
            do_asserts=True,
            max_rel_excess_allowed=1e-7
        )
        print(worst_l, worst_u, worst_l_rel, worst_u_rel)

    def test_reciprocal_repeatedly(self):
        def build_weights_of_positive_zonotope():
            weights = torch.randn(4, 1, 3)

            # Ensure the value is positive
            lower_bound_box = weights[0] - torch.norm(weights[1:], p=1.0, dim=0)
            weights[0, lower_bound_box <= 0] += -lower_bound_box[lower_bound_box <= 0] + torch.rand((lower_bound_box <= 0).sum()) + 0.00000001

            new_lower_bound = weights[0] - torch.norm(weights[1:], p=1.0, dim=0)
            assert (new_lower_bound > 0).all()

            return weights



        # original implmementation (has positivity constraint)
        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=build_weights_of_positive_zonotope,
            concrete_point_fn=lambda x: x.reciprocal(),
            zonotope_fn=lambda my_zonotope: my_zonotope.reciprocal(original_implementation=True),
            iterations=30,
            do_asserts=False,
            max_rel_excess_allowed=1e-6,
            check_fn_on_output_zonotope=check_zonotope_positivity
        )
        print(f"Old implementation:          worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}")

        # new implementation, with constraint
        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=build_weights_of_positive_zonotope,
            concrete_point_fn=lambda x: x.reciprocal(),
            zonotope_fn=lambda my_zonotope: my_zonotope.reciprocal(original_implementation=False, y_positive_constraint=True),
            iterations=30,
            do_asserts=False,
            max_rel_excess_allowed=1e-6,
            check_fn_on_output_zonotope=check_zonotope_positivity
        )
        print(f"New implementation, with positivity constraint:          worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}")

        # new implementation, without constraint
        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=build_weights_of_positive_zonotope,
            concrete_point_fn=lambda x: x.reciprocal(),
            zonotope_fn=lambda my_zonotope: my_zonotope.reciprocal(original_implementation=False, y_positive_constraint=False),
            iterations=30,
            do_asserts=False,
            max_rel_excess_allowed=1e-6,
            check_fn_on_output_zonotope=None
        )
        print(f"New implementation, without positivity constraint:          worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}")
        a = 5

    def test_exp_repeatedly(self):
        def build_zonotope():
            # weights = torch.tensor(
            #     [[ 3.0976, -2.3951, -1.8525, -3.6466]])
            # weights = weights.permute(1, 0).reshape(1, 4, 1, 1)
            # return weights

            return torch.randn(4, 1, 10)

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=lambda x: x.exp(),
            zonotope_fn=lambda my_zonotope: my_zonotope.exp(mark=True),
            iterations=30,
            do_asserts=True,
            max_rel_excess_allowed=5e-5,
            check_fn_on_output_zonotope=check_zonotope_positivity
        )

    def test_sum_exp_repeatedly(self):
        def fun_concrete(x):
            return x.exp().sum(dim=-1, keepdim=True)

        def fun_zonotope(x: Zonotope):
            y = x.exp(mark=True)
            y_summed = y.zonotope_w.sum(dim=-1, keepdim=True)
            return make_zonotope_new_weights_same_args(y_summed, y)

        def make_zonotope():
            # weights = [
            #     [2.0, 1.0],  # center
            #     [1.0, -0.5],  # error 1
            #     [0.0, 0.0]   # error 2
            # ]
            # result = torch.tensor(weights).unsqueeze(1)
            # assert result.shape == (3, 1, 2)
            # return result
            return torch.randn(20, 50, 10)

        worst_l, worst_u, worst_l_rel, worst_u_rel = self.verify_sanity(
            generate_zonotope_weights_fn=make_zonotope,
            concrete_point_fn=fun_concrete,
            zonotope_fn=fun_zonotope,
            iterations=30,
            do_asserts=True,
            max_rel_excess_allowed=1e-7
        )

        # Initially we have
        #   x1 = a0 + a1 e1 + a2 e2 + a3 e3
        #   x2 = b0 + b1 e1 + b2 e2 + b3 e3
        # we then apply the exp() to x1 and x2 and get
        #   y1 = c0 + c1 e1 + c2 e2 + c3 e3 + c4 e4
        #   y2 = d0 + d1 e1 + d2 e2 + d3 e3          + c5 e5
        # we then sum then and obtain
        #   w = (c0 + d0) + (c1 + d1) e1 + (c2 + d2) e2 + (c3 + d3) e3 + c4 e4 + c5 e5
        #
        # Since the exp passes the test pretty well (rel_error <= 1-e7) but the sum fails
        # there must be a problem with the terms 0-3. It can't be the term e4 and e5 since the sum doesn't affect hem
        # Therefore, there's something in the cancellation that leads to pretty big problems
        #
        # More specifically, the range can only be reduced if the c and d term have opposite signs

    def test_softmax_new(self):
        # weights = torch.tensor(
        #     [[[[1., 2., 3.]],
        #       [[0., 0., 0.]],
        #       [[0., 0., 0.]],
        #       [[0., 0., 0.]]]]
        # )
        # # weights = torch.tensor([[-0.3762, -0.9883, -1.0568,  0.7542,  1.0575],
        # #     [-1.2182,  0.7110,  0.8776, -0.3580,  2.0015],
        # #     [-0.3121,  0.4590,  0.0265,  0.4091, -0.4017]]).reshape(1, 3, 1, 5)
        # zonotope = self.make_zonotope(weights)
        #
        # ### CHECK predefined zonotope, WITHOUT constraint, check old and new softmax implementation
        # zonotope_softmax_new = zonotope.softmax(use_new_softmax=True, no_constraints=True)
        # zonotope_softmax = zonotope.softmax(use_new_softmax=False, no_constraints=True)
        #
        # l_new, u_new = zonotope_softmax_new.concretize()
        # l, u = zonotope_softmax.concretize()
        # self.check_values_and_interesting_points(zonotope, l, u)
        # self.check_values_and_interesting_points(zonotope, l_new, u_new)

        def intermediate_fn(x: torch.Tensor):
            assert x.ndim == 3
            A, num_rows, num_values = x.size(0), x.size(1), x.size(2)

            vals_w = x.unsqueeze(-1)  # (A, num_rows, num_values, 1)
            vals_w = vals_w.repeat(1, 1, 1, num_values)  # (A, num_rows, num_values, num_values)
            diffs_w = vals_w.transpose(2, 3) - vals_w  # TODO: check this trick for easy subtraction word

            # End shape: (A, 1 + num_error_terms, num_rows, num_softmax * num_softmax)?
            diffs_w_reshaped = diffs_w.reshape(A, num_rows, num_values * num_values)
            # return diffs_w_reshaped

            diffs_exp_w = diffs_w_reshaped.exp()
            # return diffs_exp_w

            diffs_exp_w_reshaped = diffs_w.exp()
            sum_diffs_w = diffs_exp_w_reshaped.sum(dim=-1)
            return sum_diffs_w

            softmax_x = sum_diffs_w.reciprocal()
            return softmax_x

        def build_zonotope():
            # return torch.tensor([[[[ 1.5956,  1.8460, -0.7677, -0.6367,  0.2691]],
            #                      [[ 2.4251,  0.6641,  0.2327, -0.6209, -0.4296]],
            #                      [[ 0.5281,  0.1622,  0.9291,  0.1453, -0.1330]],
            #                      [[ 0.2203,  0.4353, -1.3272, -1.0418, -0.9941]]]])

            # return torch.randn(1, 3, 1, 2)
            num_error_terms = 10
            return torch.randn(2, 1 + num_error_terms, 3, 5) / num_error_terms * 5.0

            # return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
            #                      [1.0, 0.0, 0.0, 0.0, 0.0],
            #                      [0.0, 1.0, 0.0, 0.0, 0.0]]).reshape(1, 3, 1, 5)


            # return torch.tensor([[-1.2140, 1.6013, 0.7947, -0.5784, 0.6385],
            #         [0.8106, 1.8376, 0.5781, 0.1218, 0.1735],
            #         [0.5213, 2.2342, -1.9506, -1.3038, -0.2559]]).reshape(1, 3, 1, 5)

        # worst_l_new, worst_u_new = self.verify_sanity(
        #     generate_zonotope_weights_fn=build_zonotope,
        #     concrete_point_fn=intermediate_fn,
        #     zonotope_fn=lambda zonotope: zonotope.softmax(use_new_softmax=True, no_constraints=True),
        #     iterations=30,
        #     do_asserts=True,
        #     max_rel_excess_allowed=1e-6
        # )

        # a = 10
        # return

        ### Test: random zonotopes, WITHOUT constraints, check old and new softmax implementation

        ITERATIONS = 50
        DO_ASSERTS = False
        NO_CONSTRAINTS = True
        softmax_fn = torch.nn.Softmax(dim=-1)

        # TODO: measure precision of the resulting zonotope
        # tighter is bettter
        # TODO: when I increase the number of error terms, I get negative exps... fix this

        def get_sum_softmax(output_zonotope: Zonotope, state: Tuple[float, float, float, float, float]):
            values_lower, values_upper = output_zonotope.concretize()
            value_min, value_max = values_lower.min().item(), values_upper.max().item()
            max_range = (values_upper - values_lower).max().item()

            sum_weights = output_zonotope.zonotope_w.sum(dim=-1, keepdim=True)
            sum_zonotope = make_zonotope_new_weights_same_args(sum_weights, output_zonotope)
            lower, upper = sum_zonotope.concretize()
            lower_min, upper_max = lower.min().item(), upper.max().item()

            if lower_min < 0.0 or upper_max > 1.5:
                a = 5

            if state is None:
                result = (lower_min, upper_max, value_min, value_max, max_range)
            else:
                prev_min, prev_max, prev_value_min, prev_value_max, prev_range_max = state
                result = (
                    min(prev_min, lower_min),
                    max(prev_max, upper_max),
                    min(prev_value_min, value_min),
                    max(prev_value_max, value_max),
                    max(prev_range_max, max_range)
                )

            if lower_min < 0.5 or upper_max > 0.5:
                a = 5

            return result, result


        # NEW SOFTMAX, WITH EQUALITY CONSTRAINT
        NO_CONSTRAINTS = False

        def print_stats(sum_stats):
            print(f"sum_min : {sum_stats[0]:.9f}     sum_max   : {sum_stats[1]:.9f}      "
                  f"value_min:  {sum_stats[2]:.9f}   value_max : {sum_stats[3]:.9f}      "
                  f"values_max_range_size:  {sum_stats[4]:.9f}\n")

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS,
                                                                use_new_reciprocal=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"New softmax implementation, with equality constraint, old reciprocal, automatic y > 0 constraint:")
        print(f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            # check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"New softmax implementation, with equality constraint, new reciprocal, without y > 0 constraints:")
        print(
            f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=True),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            # check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )

        print(f"New softmax implementation, with equality constraint, new reciprocal, with y > 0 constraints:")
        print(
            f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)


        # NEW SOFTMAX, WITHOUT EQUALITY CONSTRAINT
        NO_CONSTRAINTS = True

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS, use_new_reciprocal=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"New softmax implementation, old reciprocal, automatic y > 0 constraint:")
        print(f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            # check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"New softmax implementation, new reciprocal, without y > 0 constraints:")
        print(f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)

        worst_l_new, worst_u_new, worst_l_new_rel, worst_u_new_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=True, no_constraints=NO_CONSTRAINTS,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=True),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-4,
            # check_fn_on_output_zonotope=check_zonotope_in_0_1_range,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )

        print(f"New softmax implementation, new reciprocal, with y > 0 constraints:")
        print(f"worst_l: {worst_l_new:.9f}          worst_u: {worst_u_new:.9f}    worst_l_rel: {worst_l_new_rel:.9f}          worst_u_rel: {worst_u_new_rel:.9f}")
        print_stats(sum_stats)


        # OLD SOFTMAX, WITHOUT EQUALITY CONSTRAINT
        NO_CONSTRAINTS = True

        # There's no positivity guarantee for the old softmax, since there's a multiply() after the reciprocal
        worst_l, worst_u, worst_l_rel, worst_u_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=False, no_constraints=True, use_new_reciprocal=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-7,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"Old softmax implementation, old reciprocal:")
        print(f"worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}    worst_l_rel: {worst_l_rel:.9f}          worst_u_rel: {worst_u_rel:.9f}")
        print_stats(sum_stats)

        worst_l, worst_u, worst_l_rel, worst_u_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=False, no_constraints=True,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=False),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-7,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"Old softmax implementation, new reciprocal, without reciprocal's y > 0 constraints:")
        print(f"worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}    worst_l_rel: {worst_l_rel:.9f}          worst_u_rel: {worst_u_rel:.9f}")
        print_stats(sum_stats)

        worst_l, worst_u, worst_l_rel, worst_u_rel, sum_stats = self.verify_sanity(
            generate_zonotope_weights_fn=build_zonotope,
            concrete_point_fn=softmax_fn,
            zonotope_fn=lambda my_zonotope: my_zonotope.softmax(use_new_softmax=False, no_constraints=True,
                                                                use_new_reciprocal=True, add_value_positivity_constraint=True),
            iterations=ITERATIONS,
            do_asserts=DO_ASSERTS,
            max_rel_excess_allowed=1e-7,
            collect_stats_on_output_zonotopes=get_sum_softmax
        )
        print(f"Old softmax implementation, new reciprocal, with reciprocal's y > 0 constraints:")
        print(f"worst_l: {worst_l:.9f}          worst_u: {worst_u:.9f}    worst_l_rel: {worst_l_rel:.9f}          worst_u_rel: {worst_u_rel:.9f}")
        print_stats(sum_stats)




        print("\n\n")
        a = 5
        return
        # TODO
        ### Test: random zonotopes, WITH constraints, check old and new softmax implementation
        for i in range(30):
            weights = torch.randn(2, 3, 4, 5)
            zonotope = self.make_zonotope(weights)
            zonotope_softmax_new = zonotope.softmax(use_new_softmax=True, no_constraints=False)
            zonotope_softmax = zonotope.softmax(use_new_softmax=False, no_constraints=False)

            l_new, u_new = zonotope_softmax_new.concretize()
            l, u = zonotope_softmax.concretize()
            max_l_error, max_u_error = check_values_and_interesting_points(zonotope, l, u)
            max_l_error_new, max_u_error_new = check_values_and_interesting_points(zonotope, l_new, u_new)

    def test_softmax(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([[
            [[1.0, 5.0, 3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ]])

        # Ranges: [-1, 3], [1, 3], [-3, -3]

        scaling_factor = 10
        zonotope = self.make_zonotope(zonotope_w * scaling_factor)
        transformed_zonotope = zonotope.softmax()

        expected_weights = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[100.0, 1.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[100.0, 0.0, 0.0]],
            [[31.0, 50.0, 30.0]],
        ])  # TODO: these are bullshit weights
        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensors_almost_equal(trans_l, l)  # Ensure soundness
        self.assert_tensors_almost_equal(trans_u, u)  # Ensure soundness

    def test_tanh(self):
        # First word was perturbed
        # num_words = 1
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])
        # Ranges: [-1, 3], [1, 3], [-3, -3]

        zonotope = self.make_zonotope(zonotope_w)
        transformed_zonotope = zonotope.tanh()

        # Re-used the equation in the paper
        #   Bias: λ a0 + 0.5 * (tanh(u) + tanh(l) - λ(u + l))
        #   Error weights: λ w_i
        #   New error weights: 0.5 * (tanh(u) + tanh(l) - λ(u - l))
        # IMPORTANT: the DeepZ paper has an error, where they say that the equation for the new error
        # weights is 0.5 * tanh(u) + tanh(l) - λ(u - l). This equation is INCORRECT. The one here passes all the tests
        lamb1 = min(1 - tanh(-1) ** 2, 1 - tanh(3) ** 2)
        lamb2 = min(1 - tanh(1) ** 2, 1 - tanh(3) ** 2)
        expected_weights = torch.tensor([
            [[lamb1 * 1.0 + 0.5 * (tanh(3) + tanh(-1) - lamb1 * (3 + (-1))), lamb2 * 2.0 + 0.5 * (tanh(3) + tanh(1) - lamb2 * (3 + 1)), tanh(-3)]],
            [[lamb1 * 1.0                   , lamb2 * 1.0, 0.0]],
            [[lamb1 * 1.0                   , 0.0, 0.0]],
            [[0.0                           , 0.0, 0.0]],
            [[0.5 * (tanh(3) - tanh(-1) - lamb1 * (3 - (-1)))                 , 0.0, 0.0]],
            [[0.0                  , 0.5 * (tanh(3) - tanh(1) - lamb2 * (3 - 1)), 0.0]],
        ])
        self.assert_tensors_almost_equal(transformed_zonotope.zonotope_w, expected_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = transformed_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, torch.tanh(l))  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, torch.tanh(u))  # Ensure soundness

        # For the tanh, we should have f(bounds(z)) = bounds(f(z))
        self.assert_tensors_almost_equal(torch.tanh(l), trans_l)
        self.assert_tensors_almost_equal(torch.tanh(u), trans_u)

    def test_elementwise_multiplication_with_other_zonotope(self):
        zonotope_w = torch.tensor([
            [[1.0, -5.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 1.0, 5.0]],
            [[0.0, -3.0, 0.0]],
        ])

        zonotope_w2 = torch.tensor([
            [[2.0, 5.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[2.0, 3.0, 0.0]],
            [[0.0, 1.0, 2.0]],
        ])

        #                           e1 e2                       e1 e3                           e2 e3
        mixed_term_abs1 = abs(1.0 * 2.0 + 1.0 * 1.0) + abs(1.0 * 0.0 + 0.0 * 1.0) + abs(1.0 * 0.0 + 0.0 * 2.0)
        mixed_term_abs2 = abs(1.0 * 3.0 + 1.0 * 1.0) + abs(1.0 * 1.0 + 1.0 * (-3.0)) + abs(1.0 * 1.0 + 3.0 * (-3.0))
        mixed_term_abs3 = abs(0.0 * 0.0 + 5.0 * 0.0) + abs(0.0 * 2.0 + 0.0 * 0.0) + abs(5.0 * 2.0 + 0.0 * 0.0)

        half_term1 = 0.5 * (1.0 * 1.0 + 1.0 * 2.0 + 0.0 * 0.0)
        half_term2 = 0.5 * (1.0 * 1.0 + 1.0 * 3.0 - 3.0 * 1.0)
        half_term3 = 0.5 * (0.0 * 0.0 + 5.0 * 0.0 + 0.0 * 2.0)

        half_term1_abs = 0.5 * (1.0 * 1.0 + 1.0 * 2.0 + 0.0 * 0.0)
        half_term2_abs = 0.5 * (1.0 * 1.0 + 1.0 * 3.0 + 3.0 * 1.0)
        half_term3_abs = 0.5 * (0.0 * 0.0 + 5.0 * 0.0 + 0.0 * 2.0)

        zonotope1 = self.make_zonotope(zonotope_w)
        zonotope2 = self.make_zonotope(zonotope_w2)

        output_zonotope = zonotope1.multiply(zonotope2)

        expected_weights = torch.tensor([
            [[2.0 + half_term1, -25.0 + half_term2, 9.0 + half_term3]],
            [[2.0 * 1.0 + 1.0 * 1.0, 5.0 * 1.0 - 5.0 * 1.0, -3.0 * 0.0 - 3.0 * 0.0]],
            [[2.0 * 1.0 + 1.0 * 2.0, 5.0 * 1.0 - 5.0 * 3.0, -3.0 * 0.0 - 3.0 * 5.0]],
            [[2.0 * 0.0 + 1.0 * 0.0, 5.0 * (-3.0) - 5.0 * 1.0, -3.0 * 0.0 - 3.0 * 2.0]],
            [[half_term1_abs + mixed_term_abs1, 0.0, 0.0]],
            [[0.0, half_term2_abs + mixed_term_abs2, 0.0]],
            [[0.0, 0.0, half_term3_abs + mixed_term_abs3]],
        ])

        self.assert_tensors_equal(output_zonotope.zonotope_w, expected_weights)

        # The interval bounds we get using Zonotope should be at least as good as Box
        # TODO: currently this isn't the case, understand where the problem comes from...
        # l1, u1 = zonotope1.concretize()
        # l2, u2 = zonotope2.concretize()
        # interval_bounds_l = torch.stack([l1 * l2, l1 * u2, u1 * l2, u1 * u2]).min(dim=0).values
        # interval_bounds_u = torch.stack([l1 * l2, l1 * u2, u1 * l2, u1 * u2]).max(dim=0).values
        # trans_l, trans_u = output_zonotope.concretize()
        # self.assert_tensor_is_smaller(interval_bounds_l, trans_l)  # Ensure soundness
        # self.assert_tensor_is_bigger(interval_bounds_u, trans_u)  # Ensure soundness

        # We do not necessarily have that f(bounds(z)) = bounds(f(z)), therefore we don't ensure this property

    def test_elementwise_multiplication_with_other_zonotope_example_lecture_notes(self):
        zonotope_w = torch.tensor([
            [[1.0]],
            [[1.0]],
            [[1.0]],
        ])

        zonotope_w2 = torch.tensor([
            [[1.0]],
            [[-1.0]],
            [[0.0]],
        ])

        zonotope1 = self.make_zonotope(zonotope_w)
        zonotope2 = self.make_zonotope(zonotope_w2)

        output_zonotope = zonotope1.multiply(zonotope2)

        expected_weights = torch.tensor([
            [[0.5]],
            [[0.0]],
            [[1.0]],
            [[1.5]],
        ])

        # If I compute their example by hand I get
        # Bias: 1 * 1 + 0.5 * abs(1 * 1) = 1.5
        # Error 1: 1 * 1 + 1 * (-1) = 0
        # Error 2: 1 * 1  + 1 * 0   = 1
        # Error 3: 0.5 * abs(1 * 1) + abs(1 * 0 + -1 * 1) = 0.5 * 1 + abs(-1) = 1.5
        # Final form: 1.5 + e2 + 1.5e3
        # Min: 1.5 - 1 - 1.5 = -1
        # Max: 1.5 + 1 + 1.5 = 4
        # So there's a problem between the formulas and desired output
        # which is 0.5 + e2 + 1.5e3
        # The problem is 0.5
        # I think the abs in the i=j case is incorrect
        # a * b * e_i * e_i = a * b * [0, 1] =
        # 1) [0, a * b]  -> (a * b) / 2 +  (a * b) / 2 * error
        #    OR
        # 2) [a * b, 0]  -> (a * b) / 2 +  abs((a * b) / 2) * error
        # so the range will be

        self.assert_tensors_equal(output_zonotope.zonotope_w, expected_weights)

        real_l, real_u = torch.tensor([[-2.0]]), torch.tensor([[2.25]])
        trans_l, trans_u = output_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, real_l)  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, real_u)  # Ensure soundness

        # We do not necessarily have that f(bounds(z)) = bounds(f(z)), therefore we don't ensure this property

    def test_dot_product_keys_queries(self):
        # Keys and Queries shape: error terms x N words x embedding dim
        # Output: error terms x N words x N words
        # We will use 2 words, and embedding size 3
        zonotope_keys_w = torch.tensor([[
            [[1.0, -5.0, -3.0],
             [1.0, -3.0, 2.0]],
            [[1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[1.0, -1.0, 5.0],
             [1.0, -2.0, 5.0]],
        ]])

        zonotope_queries_w = torch.tensor([[
            [[2.0, 5.0, -3.0],
             [1.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0],
             [1.0, 1.0, 0.0]],
            [[2.0, 3.0, 0.0],
             [2.0, 3.0, 0.0]],
        ]])

        new_coeffs_weights1 = torch.tensor([
            [
                [(1.0 + -5.0 * 1.0),
                 (1.0 + -3.0 * 1.0)],
                [(1.0 + -5.0 * 1.0),
                 (1.0 + -3.0 * 1.0)]
            ],
            [
                [(2.0 + -5.0 * 3.0),
                 (2.0 + -3.0 * 3.0)],
                [(2.0 + -5.0 * 3.0),
                 (2.0 + -3.0 * 3.0)]
            ]
        ])

        new_coeffs_weights2 = torch.tensor([
            [
                [(2.0 * 1 + 5 * 1 - 3 * 0),
                 (2.0 * 1 + 5 * 1 - 3 * 0)],
                [(1.0 * 1 + 1.0 * 0),
                 (1.0 * 1 + 1.0 * 0)]
            ],
            [
                [(2.0 * 1 + 5 * (-1) - 3 * 5),
                 (2.0 * 1 + 5 * (-2) - 3 * 5)],

                [(1.0 * 1 + 1.0 * 5),
                 (1.0 * 1 + 1.0 * 5)]
            ]
        ])

        new_coeffs_weights = torch.tensor([
            [
                [(1.0 + -5.0 * 1.0) + (2.0 * 1 + 5 * 1 - 3 * 0),
                 (1.0 + -5.0 * 1.0) + (1.0 * 1 + 1.0 * 0)],
                [(1.0 + -3.0 * 1.0) + (2.0 * 1 + 5 * 1 - 3 * 0),
                 (1.0 + -3.0 * 1.0) + (1.0 * 1 + 1.0 * 0)]
            ],
            [
                [(2.0 + -5.0 * 3.0) + (2.0 * 1 + 5 * (-1) - 3 * 5),
                 (2.0 + -5.0 * 3.0) + (1.0 * 1 + 1.0 * 5)],

                [(2.0 + -3.0 * 3.0) + (2.0 * 1 + 5 * (-2) - 3 * 5),
                 (2.0 + -3.0 * 3.0) + (1.0 * 1 + 1.0 * 5)]
            ]
        ])

        half_term_bias = 0.5 * torch.tensor([
            [(1.0 + 1.0) + (2.0 - 3.0), (1.0 + 1.0) + (2.0 - 3.0)],
            [(1.0 + 1.0) + (2.0 - 6.0), (1.0 + 1.0) + (2.0 - 6.0)]
        ])
        half_term_weights = 0.5 * torch.tensor([
            [(1.0 + 1.0) + abs(2.0 - 3.0), (1.0 + 1.0) + abs(2.0 - 3.0)],
            [(1.0 + 1.0) + abs(2.0 - 6.0), (1.0 + 1.0) + abs(2.0 - 6.0)]
        ])

        new_bias = torch.tensor([
            [2.0 - 25.0 + 9.0, 1.0 + 0.0 - 3.0],
            [2.0 - 15.0 - 6.0, 1.0 + 0.0 + 2.0]
        ])

        cross_terms_weights = torch.tensor([
            [
                abs((1.0 * 2.0 + 1.0 * 3.0) + (1.0 * 1.0 - 1.0 * 1.0 + 5.0 * 0)),
                abs((1.0 * 2.0 + 1.0 * 3.0) + (1.0 * 1.0 - 1.0 * 1.0 + 5.0 * 0))
            ],
            [
                abs((1.0 * 2.0 + 1.0 * 3.0) + (1.0 * 1.0  + (-2.0) * 1.0 + 5.0 * 0)),
                abs((1.0 * 2.0 + 1.0 * 3.0) + (1.0 * 1.0  + (-2.0) * 1.0 + 5.0 * 0))
            ]
        ])

        new_error_weights = half_term_weights + cross_terms_weights
        new_error_weight_reshaped = torch.zeros(4, 2, 2)
        new_error_weight_reshaped[0, 0, 0] = new_error_weights[0, 0]
        new_error_weight_reshaped[1, 0, 1] = new_error_weights[0, 1]
        new_error_weight_reshaped[2, 1, 0] = new_error_weights[1, 0]
        new_error_weight_reshaped[3, 1, 1] = new_error_weights[1, 1]

        expected_weights = torch.cat([
            (new_bias + half_term_bias).unsqueeze(0),
            new_coeffs_weights,
            new_error_weight_reshaped
        ]).unsqueeze(0)  # The unsqueeze is there to create a extra attention head dim

        zonotope_keys = self.make_zonotope(zonotope_keys_w)
        zonotope_queries = self.make_zonotope(zonotope_queries_w)

        output_zonotope = zonotope_keys.dot_product(zonotope_queries)
        output_zonotope_einsum_non_chunked = zonotope_keys.dot_product_einsum(zonotope_queries, N=1)
        output_zonotope_einsum_chunked = zonotope_keys.dot_product_einsum(zonotope_queries, N=2)
        output_zonotope_einsum_automaticly_chunked = zonotope_keys.dot_product_einsum(zonotope_queries, N=None)

        self.assert_tensors_equal(output_zonotope.zonotope_w, expected_weights)
        self.assert_tensors_equal(output_zonotope.zonotope_w, output_zonotope_einsum_non_chunked.zonotope_w)
        self.assert_tensors_equal(output_zonotope.zonotope_w, output_zonotope_einsum_chunked.zonotope_w)
        self.assert_tensors_equal(output_zonotope.zonotope_w, output_zonotope_einsum_automaticly_chunked.zonotope_w)

    def test_transpose(self):
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])

        zonotope = self.make_zonotope(zonotope_w)
        output_zonotope = zonotope.t()

        output_weights = torch.tensor([
            [[1.0, 3.0],
             [2.0, 4.0],
             [3.0, 5.0]],
            [[1.0, 0.0],
             [1.0, 0.0],
             [1.0, 0.0]],
            [[1.0, 0.0],
             [1.0, 0.0],
             [0.0, 0.0]],
            [[1.0, 0.0],
             [0.0, 0.0],
             [0.0, 0.0]],
        ])

        self.assert_tensors_equal(output_zonotope.zonotope_w, output_weights)

        l, u = zonotope.concretize()
        trans_l, trans_u = output_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, l.t())  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, u.t())  # Ensure soundness

        # For the transpose, we should have f(bounds(z)) = bounds(f(z))
        self.assert_tensors_almost_equal(l.t(), trans_l)
        self.assert_tensors_almost_equal(u.t(), trans_u)

    def test_add_zonotope(self):
        zonotope_w = torch.tensor([
            [[1.0, -5.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])

        zonotope_w2 = torch.tensor([
            [[1.0, 5.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[8.0, 0.0, 0.0]],
            [[0.0, 0.0, 10.0]],
        ])

        zonotope1 = self.make_zonotope(zonotope_w)
        zonotope2 = self.make_zonotope(zonotope_w2)

        output_zonotope = zonotope1.add(zonotope2)
        self.assert_tensors_equal(output_zonotope.zonotope_w, zonotope_w + zonotope_w2)

    def test_matmul(self):
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])
        zonotope = self.make_zonotope(zonotope_w)

        # Take embedding of size 3 and transform it into an embedding of size 1
        w = torch.tensor([
            [3.0, -2.0, 1.0],
        ])
        output_zonotope = zonotope.matmul(w)

        # num_words = 2
        # embedding_size = 1
        # error_terms = 3
        # shape (1 + 3, 2, 1)
        expected_weights = torch.tensor([
            [[2.0],
             [6.0]],
            [[2.0],
             [0.0]],
            [[1.0],
             [0.0]],
            [[3.0],
             [0.0]],
        ])

        self.assert_tensors_equal(output_zonotope.zonotope_w, expected_weights)

    def test_multiplication_float(self):
        zonotope_w = torch.tensor([
            [[1.0, -5.0, -3.0]],
            [[1.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ])

        zonotope = self.make_zonotope(zonotope_w)
        output_zonotope = zonotope.multiply(5)
        self.assert_tensors_equal(output_zonotope.zonotope_w, zonotope_w * 5)

        l, u = zonotope.concretize()
        trans_l, trans_u = output_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, l * 5)  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, u * 5)  # Ensure soundness

        # For the tanh, we should have f(bounds(z)) = bounds(f(z))
        self.assert_tensors_almost_equal(l * 5, trans_l)
        self.assert_tensors_almost_equal(u * 5, trans_u)

    def test_multiplication_elementwise_matrix(self):
        # First word was perturbed
        # num_words = 2
        # embedding_size = 3
        # error_terms = 3
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])

        matrix = torch.tensor([
            [0.0, 2.0, -2.0],
            [1.0, 5.0, -4.0]
        ])

        zonotope = self.make_zonotope(zonotope_w)
        output_zonotope = zonotope.multiply(matrix)

        self.assert_tensors_equal(output_zonotope.zonotope_w, zonotope_w * matrix)

        l, u = zonotope.concretize()
        trans_l, trans_u = output_zonotope.concretize()
        self.assert_tensor_is_smaller(trans_l, l * matrix)  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, u * matrix)  # Ensure soundness

        self.assert_tensors_almost_equal(l * (matrix >= 0) * matrix + u * (matrix <= 0) * matrix, trans_l)
        self.assert_tensors_almost_equal(u * (matrix >= 0) * matrix + l * (matrix <= 0) * matrix, trans_u)

    def test_softmax_todo(self):
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])

        zonotope = self.make_zonotope(zonotope_w)
        output_zonotope = zonotope.softmax()

        # TODO: I haven't manually computed the weights because that's a lot of work, should do that later

        l, u = zonotope.concretize()
        trans_l, trans_u = output_zonotope.concretize()
        softmax = nn.Softmax(dim=-1)

        self.assert_tensor_is_smaller(trans_l, softmax(l))  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, softmax(u))  # Ensure soundness

        self.assert_tensors_almost_equal(softmax(l), trans_l)
        self.assert_tensors_almost_equal(softmax(u), trans_u)

    def test_layer_norm(self):
        zonotope_w = torch.tensor([
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0]],
            [[1.0, 1.0, 1.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0],
             [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0]],
        ])

        zonotope = self.make_zonotope(zonotope_w)

        # This linear layer does nothing
        linear = nn.Linear(in_features=3, out_features=3, bias=False)
        linear.weight = torch.eye(3)

        output_zonotope = zonotope.layer_norm(normalizer=linear, layer_norm="no_var")

        # TODO: I haven't manually computed the weights because that's a lot of work, should do that later

        l, u = zonotope.concretize()
        trans_l, trans_u = output_zonotope.concretize()
        softmax = nn.Softmax(dim=-1)

        self.assert_tensor_is_smaller(trans_l, softmax(l))  # Ensure soundness
        self.assert_tensor_is_bigger(trans_u, softmax(u))  # Ensure soundness

        self.assert_tensors_almost_equal(softmax(l), trans_l)
        self.assert_tensors_almost_equal(softmax(u), trans_u)


if __name__ == '__main__':
    unittest.main()
