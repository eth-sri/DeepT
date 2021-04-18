from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn as nn

from Verifiers import Bounds
from Verifiers.relaxation import bounds as LP_bounds, LB_split, UB_split, adjust_plane_down, adjust_plane_up


@dataclass
class ConvexCombinationData:
    # The lambdas get created at the beginning, so that we can use them as inputs to the jacobian
    lambdas_optimization: torch.Tensor  # Always on CPU, used in the optimizer
    lambdas_verification: torch.Tensor  # Can be either CPU or GPU, used in the verifier

    # The hyperplanes are created as-needed, so this is initially None and will be filled later
    # on once we actually have the coefficient values
    hyperplanes_coeffs: Optional[torch.Tensor]


def get_hyperplanes(x: Bounds, y: Bounds, operation_name: str):
    print("\nCreating hyperplanes")
    # print_bounds(x, "x")
    # print_bounds(y, "y")
    lx, ly, ux, uy = get_bounds_in_nice_format(*x.concretize(), *y.concretize())
    return get_hyperplanes_from_concrete_bounds(lx, ly, ux, uy, operation_name)


def get_hyperplanes_from_concrete_bounds(lx: np.ndarray, ly: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                                         operation_name: str) -> torch.Tensor:
    # (Al, Bl, Cl, Au, Bu, Cu) x num_elements x (5 planes)
    coeffs = torch.zeros(6, lx.shape[0], lx.shape[1], lx.shape[2], 5)
    # This way of specifying the split is bizarre, but it's what they used
    # so I'm keeping it the same to avoid introducing unnecessary changes
    for a in range(lx.shape[0]):
        for b in range(lx.shape[1]):
            for c in range(lx.shape[2]):
                for p in range(5):
                    if p == 0:
                        coeffs[0, a, b, c, p], coeffs[1, a, b, c, p], coeffs[2, a, b, c, p], \
                        coeffs[3, a, b, c, p], coeffs[4, a, b, c, p], coeffs[5, a, b, c, p] = \
                            LP_bounds(
                                lx=lx[a, b, c],
                                ux=ux[a, b, c],
                                ly=ly[a, b, c],
                                uy=uy[a, b, c],
                                name=operation_name
                            )
                    else:
                        coeffs[0, a, b, c, p], coeffs[1, a, b, c, p], coeffs[2, a, b, c, p], _ = LB_split(
                            lx=lx[a, b, c],
                            ux=ux[a, b, c],
                            ly=ly[a, b, c],
                            uy=uy[a, b, c],
                            name=operation_name,
                            split_type=split_type(p)
                        )
                        coeffs[3, a, b, c, p], coeffs[4, a, b, c, p], coeffs[5, a, b, c, p], _ = UB_split(
                            lx=lx[a, b, c],
                            ux=ux[a, b, c],
                            ly=ly[a, b, c],
                            uy=uy[a, b, c],
                            name=operation_name,
                            split_type=split_type(p)
                        )
    return coeffs


def get_hyperplanes_from_concrete_bounds_dim2(lx: np.ndarray, ly: np.ndarray, ux: np.ndarray, uy: np.ndarray,
                                              operation_name: str) -> torch.Tensor:
    # (Al, Bl, Cl, Au, Bu, Cu) x num_elements x (5 planes)
    coeffs = torch.zeros(6, lx.shape[0], lx.shape[1], 5)
    # coeffs = torch.randn(6, lx.shape[0], lx.shape[1], 5)
    # return coeffs
    # This way of specifying the split is bizarre, but it's what they used
    # so I'm keeping it the same to avoid introducing unnecessary changes
    for a in range(lx.shape[0]):
        for b in range(lx.shape[1]):
            for p in range(5):
                if p == 0:
                    coeffs[0, a, b, p], coeffs[1, a, b, p], coeffs[2, a, b, p], \
                    coeffs[3, a, b, p], coeffs[4, a, b, p], coeffs[5, a, b, p] = \
                        LP_bounds(
                            lx=lx[a, b],
                            ux=ux[a, b],
                            ly=ly[a, b],
                            uy=uy[a, b],
                            name=operation_name
                        )
                else:
                    coeffs[0, a, b, p], coeffs[1, a, b, p], coeffs[2, a, b, p], _ = LB_split(
                        lx=lx[a, b],
                        ux=ux[a, b],
                        ly=ly[a, b],
                        uy=uy[a, b],
                        name=operation_name,
                        split_type=split_type(p)
                    )
                    coeffs[3, a, b, p], coeffs[4, a, b, p], coeffs[5, a, b, p], _ = UB_split(
                        lx=lx[a, b],
                        ux=ux[a, b],
                        ly=ly[a, b],
                        uy=uy[a, b],
                        name=operation_name,
                        split_type=split_type(p)
                    )
    return coeffs


def get_hyperplanes_from_concrete_bounds_relu(l: np.ndarray, u: np.ndarray) -> torch.Tensor:
    # Only get hyperplanes for lower bound, so the shape is: num_elements x (5 planes)
    # The lower bound always has Cl = 0, so we don't need to store that
    coeffs = torch.zeros(l.shape[0], l.shape[1], 5)
    coeffs[:, :, 4] = 1
    coeffs[:, :, 3] = 0.75
    coeffs[:, :, 2] = 0.50
    coeffs[:, :, 1] = 0.25
    # coeffs[:, :, 0] = 0  <- Already the case since we created coeffs using torch.zeros
    return coeffs


def get_convexed_coeffs_relu(coefficients, lambdas):
    softmax = nn.Softmax(dim=-1)
    Al = torch.sum(coefficients * softmax(lambdas), -1)
    return Al


def get_initial_lambdas_relu(shape, device):
    # Can't put the requires_grad directly when creating the zeros tensor, because then the
    # uniform_ call creates an error "RuntimeError: a leaf Variable that requires grad has been used in an in-place operation."
    # Shape: shape (element) x 5 (one lambda per plane)
    # lambdas = lambdas.uniform_(-1, 1)
    vals = np.ones((*shape, 5), dtype=np.float32)
    lambdas = torch.from_numpy(vals)
    if device == 'cuda':
        lambdas = lambdas.cuda()
    lambdas.requires_grad = True
    return lambdas


def bound_with_convex_combination(x: Bounds, y: Bounds, coefficients, lambdas):
    Al, Au, Bl, Bu, Cl, Cu = get_convexed_coeffs(coefficients, lambdas)
    res = bound_operation_with_2_operands(x, y, Al, Bl, Cl, Au, Bu, Cu)
    # print_bounds(x, "x")
    # print_bounds(y, "y")
    # print_bounds(res, "res")
    return res


def get_convexed_coeffs(coefficients, lambdas):
    softmax = nn.Softmax(dim=-1)
    # lambdas[0]: shape x 5
    # coefficients[:3]: 3 x shape x 5
    # coefficients[:3] * softmax(lambdas[0]): 3 x shape x 5
    # torch.sum(coeff[0], -1): shape
    coeff = coefficients[:3] * softmax(lambdas[0])
    Al = torch.sum(coeff[0], -1)
    Bl = torch.sum(coeff[1], -1)
    Cl = torch.sum(coeff[2], -1)
    coeff = coefficients[3:] * softmax(lambdas[1])
    Au = torch.sum(coeff[0], -1)
    Bu = torch.sum(coeff[1], -1)
    Cu = torch.sum(coeff[2], -1)
    return Al, Au, Bl, Bu, Cl, Cu


def update_hyperplanes(coeffs: torch.Tensor, x: Bounds, y: Bounds, operation_name: str):
    print("Adjusting hyperplanes up (upper bound) and down (lower bound)")
    lx, ly, ux, uy = get_bounds_in_nice_format(*x.concretize(), *y.concretize())
    for a in range(lx.shape[0]):
        for b in range(lx.shape[1]):
            for c in range(lx.shape[2]):
                for p in range(5):
                    coeffs[0, a, b, c, p], coeffs[1, a, b, c, p], coeffs[2, a, b, c, p] = adjust_plane_down(
                        coeffs[0, a, b, c, p].item(), coeffs[1, a, b, c, p].item(), coeffs[2, a, b, c, p].item(),
                        lx[a, b, c], ux[a, b, c], ly[a, b, c], uy[a, b, c], operation_name
                    )
                    coeffs[3, a, b, c, p], coeffs[4, a, b, c, p], coeffs[5, a, b, c, p] = adjust_plane_up(
                        coeffs[3, a, b, c, p].item(), coeffs[4, a, b, c, p].item(), coeffs[5, a, b, c, p].item(),
                        lx[a, b, c], ux[a, b, c], ly[a, b, c], uy[a, b, c], operation_name
                    )


def get_initial_lambdas(shape, device):
    # Can't put the requires_grad directly when creating the zeros tensor, because then the
    # uniform_ call creates an error "RuntimeError: a leaf Variable that requires grad has been used in an in-place operation."
    # Shape: 2 (lower and upper plane) x shape (element) x 5 (one lambda per plane)
    # lambdas = lambdas.uniform_(-1, 1)
    vals = np.zeros((2, *shape, 5), dtype=np.float32)
    vals[..., 0] = 5
    vals[..., 1:] = -1
    lambdas = torch.from_numpy(vals)
    if device == 'cuda':
        lambdas = lambdas.cuda()
    lambdas.requires_grad = True
    return lambdas


def split_type(p):
    return 10 * (p // 3 + 1) + (2 - p % 2)


class Container:
    def __init__(self):
        self.elements = []
        self.finished_initializing = False
        self.current_index = 0

    def add_element(self, element):
        assert not self.finished_initializing, "Initialization of the elements is finished, can't add more elements now!"
        self.elements.append(element)

    def get_next_element(self):
        assert self.finished_initializing, "Can only start fetching once all elements have been initialized"
        assert self.current_index <= len(self.elements), \
            "Already fetched all elements for this round! Please do rueset() if you're starting a new round"
        values = self.elements[self.current_index]
        self.current_index += 1
        return values

    def get_all_elements(self):
        assert self.finished_initializing, "Shouldn't access the elements before we finished initializing them!"
        return self.elements

    def mark_initialization_over(self):
        assert not self.finished_initializing, "Already initialized!"
        assert self.current_index == 0, "Current index isn't 0 but it should be!"
        self.finished_initializing = True

    def reset_cursor(self):
        assert self.finished_initializing, "Should only reset the cursor over the lambdas once we finished initializing things"
        self.current_index = 0

    def is_empty(self):
        return len(self.elements) == 0


def get_bounds_in_nice_format(lx: torch.Tensor, ly: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lx, ux, ly, uy = lx.cpu(), ux.cpu(), ly.cpu(), uy.cpu()  # Have to bring to CPU for use in Gurobi
    lx, ux, ly, uy = lx.detach(), ux.detach(), ly.detach(), uy.detach()  # Need to detach for use in Gurobi
    lx, ux, ly, uy = lx.numpy(), ux.numpy(), ly.numpy(), uy.numpy()  # Need to convert to numpy for use in Gurobi
    return lx, ly, ux, uy


def bound_operation_with_2_operands(x: Bounds, y: Bounds, Al, Bl, Cl, Au, Bu, Cu):
    # Al x + Bl y + Cl <= x / y <= Au x + Bu y + Cu
    # And we have that
    # Vl i + Cxl <= x <= Vu i + Cxu    and   Wl i + Cyl <= y <= Wu i + Cyu
    # We begin by examining the upper bound. We can deduce that
    #    (Au+ Vl + Au- Vu) i + (Au+ Cxl + Au- Cxu) <= Au x <= (Au+ Vu + Au- Vl) i + (Au+ Cxu + Au- Cxl)
    #    (Bu+ Wl + Bu- Wu) i + (Bu+ Cyl + Bu- Cyu) <= Bu y <= (Bu+ Wu + Bu- Wl) i + (Bu+ Cyu + Bu- Cyl)
    # which allows us to get an upper bound for x / y.
    #   x / y <= Au x + Bu y + Cu
    #         <= (Au+ Vu + Au- Vl) i + (Au+ Cxu + Au- Cxl) + (Bu+ Wu + Bu- Wl) i + (Bu+ Cyu + Bu- Cyl) + Cu
    #          = ((Au+ Vu + Au- Vl) + (Bu+ Wu + Bu- Wl)) i + ((Au+ Cxu + Au- Cxl) + (Bu+ Cyu + Bu- Cyl) + Cu)
    # and so
    #        uw = (Au+ Vu + Au- Vl) + (Bu+ Wu + Bu- Wl)
    #        ub = (Au+ Cxu + Au- Cxl) + (Bu+ Cyu + Bu- Cyl) + Cu
    #
    # By a similar reasoning, we get that:
    # We begin by examining the upper bound. We can deduce that
    #    (Al+ Vl + Al- Vu) i + (Al+ Cxl + Al- Cxu) <= Al x <= (Al+ Vu + Al- Vl) i + (Al+ Cxu + Al- Cxl)
    #    (Bl+ Wl + Bl- Wu) i + (Bl+ Cyl + Bl- Cyu) <= Bl y <= (Bl+ Wu + Bl- Wl) i + (Bl+ Cyu + Bl- Cyl)
    # And so
    #   x / y >= Al x + Bl y + Cl
    #         >= (Al+ Vl + Al- Vu) i + (Al+ Cxl + Al- Cxu) + (Bl+ Wl + Bl- Wu) i + (Bl+ Cyl + Bl- Cyu) + Cl
    #          = ((Al+ Vl + Al- Vu) + (Bl+ Wl + Bl- Wu)) i + ((Al+ Cxl + Al- Cxu) + (Bl+ Cyl + Bl- Cyu) + Cl)
    #
    # and so
    #        lw = (Al+ Vl + Al- Vu) + (Bl+ Wl + Bl- Wu)
    #        lb = (Al+ Cxl + Al- Cxu) + (Bl+ Cyl + Bl- Cyu) + Cl
    Alp, Aln = get_pos_and_neg(Al)
    Blp, Bln = get_pos_and_neg(Bl)
    Aup, Aun = get_pos_and_neg(Au)
    Bup, Bun = get_pos_and_neg(Bu)

    Vu, Vl, Cxu, Cxl = x.uw, x.lw, x.ub, x.lb
    Wu, Wl, Cyu, Cyl = y.uw, y.lw, y.ub, y.lb

    # Alp: [4, 20, 20]
    # Blp: [4, 20, 20]
    # Vl: [4, 20, 128, 20]
    # Cxu: [4, 20, 20]
    # Wu: [4, 20, 128, 20]
    # Cyu: [4, 20, 20]

    # Alp : 4 attention heads x 20 words x 20 exp() (for the dot product)
    # Vl : 4 attention heads x 20 queries x 128 dim (embedding size) x 20 keys (for the dot product)

    # Alp @ Vl = [4, 20, 20] @ [4, 20, 128, 20] = ???
    # Vl @ Alp = [4, 20, 128, 20] @ [4, 20, 20] = [4, 20, 128, 20] @ [4, 20, 20] = [4, 20, 128]

    # Maybe it shouldn't be a matrix multiplication, but instead a simple weight multiplication (element wise)
    # So if the weight for xi is 10, and xi depends on the input using a 128-vector K, then we'd multiply K by 10
    # And so it would be
    #       VL.transpose(2, 3) x Alp
    #       [4, 20, 20, 128]   x [4, 20, 20]
    # Unfortunately, this doesn't work. So what we have is keep the original VL dimensions and instead repeat Alp 128 times
    # To do so, we can do
    # Alp.repeat(128, 1, 1, 1) has dims 128 x 4 x 20 x 20
    #       Vl                 x Alp.repeat(128, 1, 1, 1).permute(1, 2, 0, 3)
    #       [4, 20, 128, 20]   x [4, 20, 128, 20]
    #
    # To do the element wise multiplications for the biases, that's super simple, since the tensor already have the same dimensions
    # We can simply do
    #       Alp         x Cxl
    #       [4, 20, 20] x [4, 20, 20]
    # Adding Cl is fine since Cl also has the dimensions [4, 20, 20]

    # Here, the output of the division should be
    # Weights: 4 attention heads x 20 words x 128 dim (embedding size) x 20 attention probs
    # Bias: 4 attention heads x 20 words x 20 attention probs

    def mult(a, b):
        return a.repeat(x.dim_in, 1, 1, 1).permute(1, 2, 0, 3) * b

    return Bounds(
        x.args, x.p, x.eps,
        lw=(mult(Alp, Vl) + mult(Aln, Vu)) + (mult(Blp, Wl) + mult(Bln, Wu)),
        lb=(Al * Cxl + Aln * Cxu) + (Blp * Cyl + Bln * Cyu) + Cl,
        uw=(mult(Aup, Vu) + mult(Aun, Vl)) + (mult(Bup, Wu) + mult(Bun, Wl)),
        ub=(Aup * Cxu + Aun * Cxl) + (Bup * Cyu + Bun * Cyl) + Cu,
    )


def get_pos_and_neg(W: torch.Tensor):
    pos_mask = torch.gt(W, 0).to(torch.float32)
    W_pos = W * pos_mask
    W_neg = W - W_pos
    return W_pos, W_neg
