# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# (c) Anonymous Authours
# Licenced under the BSD 2-Clause License.
from typing import Callable, Any

import torch

from Verifiers.ConvexCombination import get_hyperplanes, bound_with_convex_combination, update_hyperplanes, get_initial_lambdas
from Verifiers.utils import dual_norm

epsilon = 1e-12


class Bounds:
    # W: actually transposed versions are stored
    def __init__(self, args, p, eps, w=None, b=None, lw=None, lb=None, uw=None, ub=None, clone=True):
        self.args = args
        self.use_ibp = args.method == "ibp"
        self.device = lw.device if lw is not None else w.device
        self.p = p

        self.q = dual_norm(self.p)
        # self.q = 1. / (1. - 1. / args.p) if args.p != 1 else float("inf")  # dual norm

        self.eps = eps
        self.perturbed_words = args.perturbed_words
        self.lw = lw if lw is not None else (w.clone() if clone else w)
        self.uw = uw if uw is not None else (w.clone() if clone else w)
        self.lb = lb if lb is not None else (b.clone() if clone else b)
        self.ub = ub if ub is not None else (b.clone() if clone else b)
        if self.use_ibp:
            self.lw = self.lw[:, :, :self.perturbed_words, :]
            self.uw = self.uw[:, :, :self.perturbed_words, :]
        self.update_shape()

    def to_device(self, val):
        return val if self.args.cpu else val.cuda()

    def update_shape(self):
        """ Extract the shape dimensions into internal variables (batch size, length, dim in, dim out) """
        self.batch_size = self.lw.shape[0]
        self.length = self.lw.shape[1]
        self.dim_in = self.lw.shape[2]
        self.dim_out = self.lw.shape[3]

    def print(self, message):
        """ Print a message and then statistics about the bounds and norms of lw and uw"""
        print(message)
        l, u = self.concretize()
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(l)), torch.mean(torch.abs(u))))
        print("diff %.5f %.5f %.5f" % (torch.min(u - l), torch.max(u - l), torch.mean(u - l)))
        print("lw norm", torch.mean(torch.norm(self.lw, dim=-2)))
        print("uw norm", torch.mean(torch.norm(self.uw, dim=-2)))
        print("uw - lw norm", torch.mean(torch.norm(self.uw - self.lw, dim=-2)))
        print("min", torch.min(l))
        print("max", torch.max(u))
        print()

    def concretize_l(self, lw=None):
        """ Computes the concretized scalar lower bound value given eps and the lower bound weights """
        if lw is None: lw = self.lw
        return -self.eps * torch.norm(lw, p=self.q, dim=-2)

    def concretize_u(self, uw=None):
        """ Computes the concretized scalar upper bound value given eps and the upper bound weights """
        if uw is None: uw = self.uw
        return self.eps * torch.norm(uw, p=self.q, dim=-2)

    def concretize(self):
        """ Computes the concretized scalar upper and lower bound values for each perturbed word
        and them sums the results, obtaining a total concretize lower and upper bound """
        dim = self.lw.shape[-2] // self.perturbed_words
        res_l = self.lb.clone()
        res_u = self.ub.clone()
        if self.args.attack_type == "synonym":
            for i in range(self.perturbed_words):
                # (batch len, len, dim in, dim out)
                res_l += torch.norm(self.lw[:, :, (dim * i): (dim * (i + 1)), :].transpose(2, 3) * self.args.embedding_radii[i], p=self.q, dim=-1)
                res_u += torch.norm(self.uw[:, :, (dim * i): (dim * (i + 1)), :].transpose(2, 3) * self.args.embedding_radii[i], p=self.q, dim=-1)
        else:
            for i in range(self.perturbed_words):
                res_l += self.concretize_l(self.lw[:, :, (dim * i): (dim * (i + 1)), :])
                res_u += self.concretize_u(self.uw[:, :, (dim * i): (dim * (i + 1)), :])

        return res_l, res_u

    def clone(self):
        return Bounds(
            self.args, self.p, self.eps,
            lw=self.lw.clone(), lb=self.lb.clone(),
            uw=self.uw.clone(), ub=self.ub.clone()
        )

    def t(self):
        """ Transposes the length and the dim_out dimensions of the bounds """
        return Bounds(
            self.args, self.p, self.eps,
            lw=self.lw.transpose(1, 3),
            uw=self.uw.transpose(1, 3),
            lb=self.lb.transpose(1, 2),
            ub=self.ub.transpose(1, 2)
        )

    def new(self):
        """ Utility method to create all the parameters of the current Bounds object"""
        l, u = self.concretize()

        mask_pos = torch.gt(l, 0).to(torch.float)
        mask_neg = torch.lt(u, 0).to(torch.float)
        mask_both = 1 - mask_pos - mask_neg

        lw = torch.zeros(self.lw.shape).to(self.device)
        lb = torch.zeros(self.lb.shape).to(self.device)
        uw = torch.zeros(self.uw.shape).to(self.device)
        ub = torch.zeros(self.ub.shape).to(self.device)

        return l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub

    def add_linear(self, mask, w_out, b_out, type, k, x0, y0, src=None):
        if mask is None:
            mask_w = mask_b = 1
        else:
            mask_w = mask.unsqueeze(2)
            mask_b = mask
        if src is None:  # It can be useful to use another bounds parameters, for example, the 2nd bounds of a multiply
            src = self
        if type == "lower":
            w_pos, b_pos = src.lw, src.lb
            w_neg, b_neg = src.uw, src.ub
        else:
            w_pos, b_pos = src.uw, src.ub
            w_neg, b_neg = src.lw, src.lb

        # Straightfoward application of the formulas they describe in the paper, with the 2 cases
        mask_pos = torch.gt(k, 0).to(torch.float)
        w_out += mask_w * mask_pos.unsqueeze(2) * w_pos * k.unsqueeze(2)
        b_out += mask_b * mask_pos * ((b_pos - x0) * k + y0)
        mask_neg = 1 - mask_pos
        w_out += mask_w * mask_neg.unsqueeze(2) * w_neg * k.unsqueeze(2)
        b_out += mask_b * mask_neg * ((b_neg - x0) * k + y0)

    def add(self, delta):
        if type(delta) == Bounds:
            return Bounds(
                self.args, self.p, self.eps,
                lw=self.lw + delta.lw, lb=self.lb + delta.lb,
                uw=self.uw + delta.uw, ub=self.ub + delta.ub
            )
        else:
            return Bounds(
                self.args, self.p, self.eps,
                lw=self.lw, lb=self.lb + delta,
                uw=self.uw, ub=self.ub + delta
            )

    def matmul(self, W):
        if type(W) == Bounds:
            raise NotImplementedError
        elif len(W.shape) == 2:
            W = W.t()

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            # Multiplication taking into account the sign, to ensure the bounds are correctly computed
            return Bounds(
                self.args, self.p, self.eps,
                lw=self.lw.matmul(W_pos) + self.uw.matmul(W_neg),
                lb=self.lb.matmul(W_pos) + self.ub.matmul(W_neg),
                uw=self.lw.matmul(W_neg) + self.uw.matmul(W_pos),
                ub=self.lb.matmul(W_neg) + self.ub.matmul(W_pos)
            )
        else:
            W = W.transpose(1, 2)

            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return Bounds(
                self.args, self.p, self.eps,
                lw=(self.lw.squeeze(0).bmm(W_pos) + self.uw.squeeze(0).bmm(W_neg)).unsqueeze(0),
                lb=(self.lb.transpose(0, 1).bmm(W_pos) + self.ub.transpose(0, 1).bmm(W_neg)).transpose(0, 1),
                uw=(self.lw.squeeze(0).bmm(W_neg) + self.uw.squeeze(0).bmm(W_pos)).unsqueeze(0),
                ub=(self.lb.transpose(0, 1).bmm(W_neg) + self.ub.transpose(0, 1).bmm(W_pos)).transpose(0, 1)
            )

    def multiply(self, W):
        """ Multiply by either a float, Bounds or a matrix. Takes into account the signs to
        ensure the compute lower and upper bounds weights and biases are correct. """
        if type(W) == float:
            if W > 0:
                return Bounds(
                    self.args, self.p, self.eps,
                    lw=self.lw * W, lb=self.lb * W,
                    uw=self.uw * W, ub=self.ub * W
                )
            else:
                return Bounds(
                    self.args, self.p, self.eps,
                    lw=self.uw * W, lb=self.ub * W,
                    uw=self.lw * W, ub=self.lb * W
                )
        elif type(W) == Bounds:
            assert (self.lw.shape == W.lw.shape)

            l_a, u_a = self.concretize()
            l_b, u_b = W.concretize()

            l1, u1, mask_pos_only, mask_neg_only, mask_both, lw, lb, uw, ub = self.new()

            alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = \
                self.get_bounds_xy(
                    l_a.reshape(-1),
                    u_a.reshape(-1),
                    l_b.reshape(-1),
                    u_b.reshape(-1)
                )

            alpha_l = alpha_l.reshape(l_a.shape)
            beta_l = beta_l.reshape(l_a.shape)
            gamma_l = gamma_l.reshape(l_a.shape)
            alpha_u = alpha_u.reshape(l_a.shape)
            beta_u = beta_u.reshape(l_a.shape)
            gamma_u = gamma_u.reshape(l_a.shape)

            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=alpha_l, x0=0, y0=gamma_l
            )
            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=beta_l, x0=0, y0=0, src=W
            )
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=alpha_u, x0=0, y0=gamma_u
            )
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=beta_u, x0=0, y0=0, src=W
            )

            return Bounds(
                self.args, self.p, self.eps,
                lw=lw, lb=lb, uw=uw, ub=ub
            )

        else:
            pos_mask = torch.gt(W, 0).to(torch.float32)
            W_pos = W * pos_mask
            W_neg = W - W_pos

            return Bounds(
                self.args, self.p, self.eps,
                lw=self.lw * W_pos + self.uw * W_neg,
                lb=self.lb * W_pos + self.ub * W_neg,
                uw=self.lw * W_neg + self.uw * W_pos,
                ub=self.lb * W_neg + self.ub * W_pos
            )

    def get_bounds_xy(self, l_x, u_x, l_y, u_y):
        if self.use_ibp:
            prod1 = l_x * l_y
            prod2 = l_x * u_y
            prod3 = u_x * l_y
            prod4 = u_x * u_y

            l = torch.min(prod1, torch.min(prod2, torch.min(prod3, prod4)))
            u = torch.max(prod1, torch.max(prod2, torch.max(prod3, prod4)))

            zeros = torch.zeros(l_x.shape).cuda()

            return zeros, zeros, l, zeros, zeros, u

        alpha_l = l_y
        beta_l = l_x
        gamma_l = -alpha_l * beta_l

        alpha_u = u_y
        beta_u = l_x
        gamma_u = -alpha_u * beta_u

        return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    """
    Dot product for multi-head self-attention (also used for obtaining context)

    A, B [b * h, l, in, out]

    For each one in the batch:

    d[i][j] \approx \sum_k a[i][k] * b[k][j]
            \approx \sum_k (\sum_m A[i][m][k] * x^r[m])(\sum_m B[j][m][k] * x^r[m])
        
    With a relaxation on b[k][j], so that b[k][j] \in [l[k][j], r[k][j]]:
        d[i][j] \approx \sum_k (\sum_m A[i][m][k] * x^r[m]) * b[j][k]
                = \sum_m (\sum_k A[i][m][k] * b[j][k]) * x^r[m]
        
        Consider the signs of A^L[i][m][k], A^U[i][m][k], b^L[j][k], b^U[j][k]
        Most coarse/loose first:
            D^u[i][j][m] = sum_k max(abs(A^L[i][m][k]), abs(A^U[i][m][k])) * \
                max(abs(b^L[j][k]), abs(b^U[j][k]))
            D^l[i][j][m] = -d^u[i][j]
    """
    def dot_product(self, other, debug=False, verbose=False, lower=True, upper=True):
        l_a, u_a = self.concretize()
        l_b, u_b = other.concretize()

        lw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        uw = torch.zeros(self.lw.shape[0], self.lw.shape[1], self.dim_in, other.lw.shape[1]).to(self.device)
        lb = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)
        ub = torch.zeros(self.lw.shape[0], self.lw.shape[1], other.lw.shape[1]).to(self.device)

        for t in range(self.batch_size):
            alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = \
                self.get_bounds_xy(
                    l_a[t].repeat(1, other.length).reshape(-1),
                    u_a[t].repeat(1, other.length).reshape(-1),
                    l_b[t].reshape(-1).repeat(self.length),
                    u_b[t].reshape(-1).repeat(self.length)
                )

            alpha_l = alpha_l.reshape(self.length, other.length, self.dim_out)
            beta_l = beta_l.reshape(self.length, other.length, self.dim_out)
            gamma_l = gamma_l.reshape(self.length, other.length, self.dim_out)
            alpha_u = alpha_u.reshape(self.length, other.length, self.dim_out)
            beta_u = beta_u.reshape(self.length, other.length, self.dim_out)
            gamma_u = gamma_u.reshape(self.length, other.length, self.dim_out)

            lb[t] += torch.sum(gamma_l, dim=-1)
            ub[t] += torch.sum(gamma_u, dim=-1)

            def add_w_alpha(new, old, weight, cmp):
                a = old[t].reshape(self.length, self.dim_in, 1, self.dim_out)
                b = (weight * cmp(weight, 0).to(torch.float)) \
                    .reshape(self.length, 1, other.length, self.dim_out) \
                    .transpose(2, 3)
                new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :]) \
                    .reshape(self.length, self.dim_in, other.length)

            def add_b_alpha(new, old, weight, cmp):
                new[t, :, :] += \
                    ((old[t].reshape(self.length, 1, self.dim_out)) \
                     .bmm((weight * cmp(weight, 0).to(torch.float)) \
                          .reshape(self.length, other.length, self.dim_out) \
                          .transpose(1, 2)) \
                     .reshape(self.length, other.length))

            def add_w_beta(new, old, weight, cmp):
                a = old[t].reshape(other.length, self.dim_in, 1, self.dim_out)
                b = (weight * cmp(weight, 0).to(torch.float)) \
                    .transpose(0, 1) \
                    .reshape(other.length, 1, self.length, self.dim_out) \
                    .transpose(2, 3)
                new[t, :, :, :] += a[:, :, :, :].matmul(b[:, :, :, :]) \
                    .reshape(other.length, self.dim_in, self.length).transpose(0, 2)

            def add_b_beta(new, old, weight, cmp):
                new[t, :, :] += \
                    ((old[t].reshape(other.length, 1, self.dim_out)) \
                     .bmm((weight * cmp(weight, 0).to(torch.float)) \
                          .transpose(0, 1) \
                          .reshape(other.length, self.length, self.dim_out) \
                          .transpose(1, 2)) \
                     .reshape(other.length, self.length)).transpose(0, 1)

            if lower:
                add_w_alpha(lw, self.lw, alpha_l, torch.gt)
                add_w_alpha(lw, self.uw, alpha_l, torch.lt)
                add_w_beta(lw, other.lw, beta_l, torch.gt)
                add_w_beta(lw, other.uw, beta_l, torch.lt)

                add_b_alpha(lb, self.lb, alpha_l, torch.gt)
                add_b_alpha(lb, self.ub, alpha_l, torch.lt)
                add_b_beta(lb, other.lb, beta_l, torch.gt)
                add_b_beta(lb, other.ub, beta_l, torch.lt)

            if upper:
                add_w_alpha(uw, self.uw, alpha_u, torch.gt)
                add_w_alpha(uw, self.lw, alpha_u, torch.lt)
                add_w_beta(uw, other.uw, beta_u, torch.gt)
                add_w_beta(uw, other.lw, beta_u, torch.lt)

                add_b_alpha(ub, self.ub, alpha_u, torch.gt)
                add_b_alpha(ub, self.lb, alpha_u, torch.lt)
                add_b_beta(ub, other.ub, beta_u, torch.gt)
                add_b_beta(ub, other.lb, beta_u, torch.lt)

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb, uw=uw, ub=ub
        )

    def divide(self, W):
        if type(W) == Bounds:
            self_l, self_u = self.concretize()
            w_l, w_u = W.concretize()
            # print("x_l: min - %s, max - %s" % (self_l.min(), self_l.max()))
            # print("x_u: min - %s, max - %s" % (self_u.min(), self_u.max()))
            # print("y_l: min - %s, max - %s" % (w_l.min(), w_l.max()))
            # print("y_u: min - %s, max - %s" % (w_u.min(), w_u.max()))
            W = W.reciprocal()
            w_l, w_u = W.concretize()
            # print("y_reciprocal_l: min - %s, max - %s" % (w_l.min(), w_l.max()))
            # print("y_reciprocal_u: min - %s, max - %s" % (w_u.min(), w_u.max()))
            
            res = self.multiply(W)
            res_l, res_u = res.concretize()
            # print("res_l: min - %s, max - %s" % (res_l.min(), res_l.max()))
            # print("res_u: min - %s, max - %s" % (res_u.min(), res_u.max()))

            return res
        else:
            raise NotImplementedError

    def context(self, value):
        context = self.dot_product(value.t())
        return context

    """
    U: (u+l) * (x-l) + l^2 = (u + l) x - u * l

    L: 2m (x - m) + m^2
    To make the lower bound never goes to negative:
        2m (l - m) + l^2 >= 0 => m (2l - m) >= 0
        2m (u - m) + u^2 >= 0 => m (2u - m) >= 0
    """

    def sqr(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.use_ibp:
            lb = torch.min(l * l, u * u)
            lb -= mask_both * lb  # lower bound is zero for this case
            ub = torch.max(l * l, u * u)
        else:
            # upper bound
            k = u + l
            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=k, x0=l, y0=l.pow(2)
            )

            # lower bound
            m = torch.max((l + u) / 2, 2 * u)
            self.add_linear(
                mask=mask_neg, w_out=lw, b_out=lb, type="lower",
                k=2 * m, x0=m, y0=m.pow(2)
            )
            m = torch.min((l + u) / 2, 2 * l)
            self.add_linear(
                mask=mask_pos, w_out=lw, b_out=lb, type="lower",
                k=2 * m, x0=m, y0=m.pow(2)
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb, uw=uw, ub=ub
        )

    def sqrt(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()
        assert (torch.min(l) > 0), "Min <= 0"

        if self.use_ibp:
            lb = torch.sqrt(l)
            ub = torch.sqrt(u)
        else:
            k = (torch.sqrt(u) - torch.sqrt(l)) / (u - l + epsilon)

            self.add_linear(
                mask=None, w_out=lw, b_out=lb, type="lower",
                k=k, x0=l, y0=torch.sqrt(l)
            )

            m = (l + u) / 2
            k = 0.5 / torch.sqrt(m)

            self.add_linear(
                mask=None, w_out=uw, b_out=ub, type="upper",
                k=k, x0=m, y0=torch.sqrt(m)
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb,
            uw=uw, ub=ub
        )

    def relu(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.use_ibp:
            lb = self.to_device(torch.max(l, torch.tensor(0.)))
            ub = self.to_device(torch.max(u, torch.tensor(0.)))
        else:
            # Negative (u < 0): 0 <= y <= 0  (i.e. y = 0)
            self.add_linear(
                mask=mask_neg, w_out=lw, b_out=lb, type="lower",
                k=torch.zeros(l.shape).to(self.device), x0=0, y0=0
            )
            self.add_linear(
                mask=mask_neg, w_out=uw, b_out=ub, type="upper",
                k=torch.zeros(l.shape).to(self.device), x0=0, y0=0
            )
            # Positive (l > 0): x <= y <= x  (i.e. y = x)
            self.add_linear(
                mask=mask_pos, w_out=lw, b_out=lb, type="lower",
                k=torch.ones(l.shape).to(self.device), x0=0, y0=0
            )
            self.add_linear(
                mask=mask_pos, w_out=uw, b_out=ub, type="upper",
                k=torch.ones(l.shape).to(self.device), x0=0, y0=0
            )
            # TODO: add lambdas in this case
            # Can be both positive or negative (l < 0 < x):  kx <= y <=  k (x - l)
            k = u / (u - l + epsilon)
            self.add_linear(
                mask=mask_both, w_out=uw, b_out=ub, type="upper",
                k=k, x0=l, y0=0
            )

            k = torch.gt(torch.abs(u), torch.abs(l)).to(torch.float)

            self.add_linear(
                mask=mask_both, w_out=lw, b_out=lb, type="lower",
                k=k, x0=0, y0=0
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb,
            uw=uw, ub=ub
        )

    """
    Relaxation for exp(x):
        L: y = e^((l + u) / 2) * (x - (l + u) / 2) + e ^ ((l + u) / 2)
        U: y = (e^u - e^l) / (u - l) * (x - l) + e^l
    """

    def exp(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.use_ibp:
            lb = torch.exp(l)
            ub = torch.exp(u)
        else:
            """
            To ensure that the global lower bound is always positive:
                e^alpha (l - alpha) + e^alpha > 0
                => alpha < l + 1
            """
            # TODO: add lambdas here
            m = torch.min((l + u) / 2, l + 0.99)  # Pick good point to get slope

            thres = torch.tensor(12.).to(self.device)

            def exp_with_trick(x):
                mask = torch.lt(x, thres).to(torch.float)  # x < 12    (avoid incredibly steep value for upper bound slope)
                return mask * torch.exp(torch.min(x, thres)) + (1 - mask) * (torch.exp(thres) * (x - thres + 1))

            kl = torch.exp(torch.min(m, thres))  # Get slope, but ensure the slope is <= exp(12)
            lw = self.lw * kl.unsqueeze(2)
            lb = kl * (self.lb - m + 1)

            # TODO: understand their logic

            # If u > 12 and l > 12, then the slope is (exp(12) * (u - l)) / (u - l + epsilon) = exp(12)
            # slope = exp(12)
            # equation: slope * w     +      slope * b - slope * l + slope * (l - 12 + 1) =
            # weights = w * exp(12)
            # new bias = exp(12) * (l - 12 + 1)  - slope * l    + prev_bias * slope

            # if u < 12 and l < 12, then
            # ku = slope = (e^u - e^l) / (u - l)
            # weights = prev_weights * slope
            # bias = exp(l) - slope * l + prev_bias * slope

            # if u > 12 and l < 12, then
            # ku = slope = (exp(12) * (u - 12 + 1) - e^l) / (u - l)
            #   and since l < 12, then (exp(12) * (u - 12)) / (u - l) < exp(12)
            # weights = prev_weights * slope
            # bias = exp(l) - slope * l + prev_bias * slope
            ku = (exp_with_trick(u) - exp_with_trick(l)) / (u - l + epsilon)
            uw = self.uw * ku.unsqueeze(2)
            ub = self.ub * ku - ku * l + exp_with_trick(l)

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb,
            uw=uw, ub=ub
        )

    def softmax(self, verbose=False):
        if self.args.use_new_softmax:
            # To encode the softmax, the following approach will be taken:
            # softmax_i(x1, ..., xn) = 1 / ((exp(x1) + ... exp(xn)) / exp(xi))
            #                        = 1 / ((exp(x1 - xi) + ... exp(xn - xi))
            # so we will have to have subtractions, exponentials and in the top formulation also divisions
            # We'll begin by using the bottom formulation

            # Step 1: compute all the xj - xi
            # The value xj - xi is stored at index i * dim + j
            # There's probably nicer ways to create this matrix
            dim = self.dim_out
            weights = self.to_device(torch.zeros(dim * dim, dim))
            for i in range(dim):
                for j in range(dim):
                    # Note: in the special case where i = j, we end up with 1 - 1 = 0 = xi - xi
                    weights[i * dim + j][j] += 1
                    weights[i * dim + j][i] += -1

            bounds_diff = self.matmul(weights)

            # Step 2: compute all the exp(xj - xi)
            bounds_exp_diff = bounds_diff.exp()

            # Step 3: sum all the exp(xj - xi) for each i
            # The values for a specific i are stored between i * dim and (i + 1) * dim - 1
            weights_sum_exp = self.to_device(torch.zeros(dim, dim * dim))
            for i in range(dim):
                weights_sum_exp[i, i * dim:(i + 1) * dim] = 1
            bounds_sum_exp_diff = bounds_exp_diff.matmul(weights_sum_exp)

            # Step 4: Compute the inverse for all of these sums, thus obtaining all the softmax values
            bounds_softmax = bounds_sum_exp_diff.reciprocal()

            return bounds_softmax
        else:
            bounds_exp = self.exp()
            bounds_sum = Bounds(
                self.args, self.p, self.eps,
                lw=torch.sum(bounds_exp.lw, dim=-1, keepdim=True).repeat(1, 1, 1, self.dim_out),
                uw=torch.sum(bounds_exp.uw, dim=-1, keepdim=True).repeat(1, 1, 1, self.dim_out),
                lb=torch.sum(bounds_exp.lb, dim=-1, keepdim=True).repeat(1, 1, self.dim_out),
                ub=torch.sum(bounds_exp.ub, dim=-1, keepdim=True).repeat(1, 1, self.dim_out),
            )
            return bounds_exp.divide(bounds_sum)

    def dense(self, dense):
        return self.matmul(dense.weight).add(dense.bias)

    def tanh(self):
        l, u, mask_pos, mask_neg, mask_both, lw, lb, uw, ub = self.new()

        if self.use_ibp:
            lb = torch.tanh(l)
            ub = torch.tanh(u)
        else:
            def dtanh(x):
                return 1. / torch.cosh(x).pow(2)

            # lower bound for negative
            m = (l + u) / 2
            k = dtanh(m)
            self.add_linear(
                mask=mask_neg, w_out=lw, b_out=lb, type="lower",
                k=k, x0=m, y0=torch.tanh(m)
            )
            # upper bound for positive
            self.add_linear(
                mask=mask_pos, w_out=uw, b_out=ub, type="upper",
                k=k, x0=m, y0=torch.tanh(m)
            )

            # upper bound for negative
            k = (torch.tanh(u) - torch.tanh(l)) / (u - l + epsilon)
            self.add_linear(
                mask=mask_neg, w_out=uw, b_out=ub, type="upper",
                k=k, x0=l, y0=torch.tanh(l)
            )
            # lower bound for positive
            self.add_linear(
                mask=mask_pos, w_out=lw, b_out=lb, type="lower",
                k=k, x0=l, y0=torch.tanh(l)
            )

            # bounds for both
            max_iter = 10

            # lower bound for both
            diff = lambda d: (torch.tanh(u) - torch.tanh(d)) / (u - d + epsilon) - dtanh(d)
            d = l / 2
            _l = l
            _u = torch.zeros(l.shape).to(self.device)
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * mask_p + _l * (1 - mask_p)
                _u = d * (1 - mask_p) + _u * mask_p
                d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
            k = (torch.tanh(d) - torch.tanh(u)) / (d - u + epsilon)
            self.add_linear(
                mask=mask_both, w_out=lw, b_out=lb, type="lower",
                k=k, x0=d, y0=torch.tanh(d)
            )

            # upper bound for both
            diff = lambda d: (torch.tanh(d) - torch.tanh(l)) / (d - l + epsilon) - dtanh(d)
            d = u / 2
            _l = torch.zeros(l.shape).to(self.device)
            _u = u
            for t in range(max_iter):
                v = diff(d)
                mask_p = torch.gt(v, 0).to(torch.float)
                _l = d * (1 - mask_p) + _l * mask_p
                _u = d * mask_p + _u * (1 - mask_p)
                d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
            k = (torch.tanh(d) - torch.tanh(l)) / (d - l + epsilon)
            self.add_linear(
                mask=mask_both, w_out=uw, b_out=ub, type="upper",
                k=k, x0=d, y0=torch.tanh(d)
            )

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb,
            uw=uw, ub=ub
        )

    def act(self, act_name):
        if act_name == "relu":
            return self.relu()
        else:
            raise NotImplementedError

    def layer_norm(self, normalizer, layer_norm):
        if layer_norm == "no":
            return self

        l_in, u_in = self.concretize()
        w_avg = torch.ones((self.dim_out, self.dim_out)).to(self.device) * (1. / self.dim_out)
        minus_mu = self.add(self.matmul(w_avg).multiply(-1.))

        l_minus_mu, u_minus_mu = minus_mu.concretize()
        dim = self.dim_out

        if layer_norm == "standard":
            variance = minus_mu.sqr().matmul(w_avg)
            normalized = minus_mu.divide(variance.add(epsilon).sqrt())
        else:
            assert (layer_norm == "no_var")
            normalized = minus_mu

        normalized = normalized.multiply(normalizer.weight).add(normalizer.bias)

        return normalized

    # """
    # Requirement: x should be guaranteed to be positive
    # """
    def reciprocal(self):
        l, u = self.concretize()

        if self.use_ibp:
            lw = self.lw * 0.
            uw = self.uw * 0.
            lb = 1. / u
            ub = 1. / l
        else:
            m = (l + u) / 2

            assert (torch.min(l) >= epsilon), "reciprocal: min(l) < epsilon"

            kl = -1 / m.pow(2)
            lw = self.uw * kl.unsqueeze(2)
            lb = self.ub * kl + 2 / m

            ku = -1. / (l * u)
            uw = self.lw * ku.unsqueeze(2)
            ub = self.lb * ku - ku * l + 1 / l

        return Bounds(
            self.args, self.p, self.eps,
            lw=lw, lb=lb,
            uw=uw, ub=ub
        )


class BestBoundsChachingOp:
    def __init__(self, use_best_bounds=True):
        # TODO: currently the place where we parametrize the caching is here
        # This isn't fantastic, since it's hardcoded in the code
        # It would be better if this was an argument instead
        self.best_lb = None
        self.best_ub = None
        self.use_best_bounds = use_best_bounds


    def reset_best_lb_and_best_ub(self):
        self.best_ub = None
        self.best_lb = None

    def compute_bounds(self, *args) -> Bounds:
        raise NotImplementedError("Implement in a sub class")

    def forward(self, input_bounds: Bounds, *args):
        if not self.use_best_bounds:
            return self.compute_bounds(*args)

        output_bounds = self.compute_bounds(input_bounds, *args)
        if self.best_lb is not None:
            self.best_lb = torch.max(self.best_lb, output_bounds.lb)
            self.best_ub = torch.min(self.best_ub, output_bounds.ub)
        else:
            self.best_lb, self.best_ub = output_bounds.lb, output_bounds.ub

        output_bounds.lb, output_bounds.ub = self.best_lb, self.best_ub

        # Detach to try to avoid keeping increasing memory forever
        self.best_lb, self.best_ub = self.best_lb.detach(), self.best_ub.detach()
        return output_bounds


class WrapperOp(BestBoundsChachingOp):
    def __init__(self, operator: Callable[[Bounds], Bounds]):
        """
        This class is used to wrap an operation in a class, so that we can re-use them
        The operator typically is something like  lambda bounds: bounds.exp()
        """
        super().__init__()
        self.operator = operator

    # noinspection PyMethodOverriding
    def compute_bounds(self, input_bounds: Bounds):
        return self.operator(input_bounds)


class Wrapper2Op(BestBoundsChachingOp):
    def __init__(self, operator: Callable[[Bounds, Bounds], Bounds]):
        """
        This class is used to wrap an operation in a class, so that we can re-use them
        The operator typically is something like  lambda bounds_pair: bounds_pair[0].dot_product(bounds_pair[1])
        """
        super().__init__()
        self.operator = operator

    # noinspection PyMethodOverriding
    def compute_bounds(self, input_bounds: Bounds, input_bounds2: Bounds):
        return self.operator(input_bounds, input_bounds2)


class WrapperGenericOp(BestBoundsChachingOp):
    def __init__(self, operator: Callable[[Bounds, Any], Bounds]):
        """
        This class is used to wrap an operation in a class, so that we can re-use them
        The operator typically is something like  lambda bounds_pair: bounds_pair[0].dot_product(bounds_pair[1])
        """
        super().__init__()
        self.operator = operator

    def compute_bounds(self, input_bounds: Bounds, *args):
        return self.operator(input_bounds, *args)


class AddOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.add(bounds2))


class MatmulOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.matmul(bounds2))


class MultiplyOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.multiply(bounds2))


class DotProductOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.dot_product(bounds2))


class ContextOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.context(bounds2))


class OldDivideOp(Wrapper2Op):
    def __init__(self):
        super().__init__(lambda bounds1, bounds2: bounds1.divide(bounds2))


class SqrOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.sqr())


class SqrtOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.sqrt())


class ActOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.relu())


class ExpOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.exp())


class TanhOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.tanh())


class ReciprocalOp(WrapperOp):
    def __init__(self):
        super().__init__(lambda bounds: bounds.reciprocal())


class LayerNormOp(WrapperGenericOp):
    def __init__(self):
        super().__init__(lambda bounds, normalizer, layer_norm: bounds.layer_norm(normalizer, layer_norm))


class DenseOp(WrapperGenericOp):
    def __init__(self):
        super().__init__(lambda bounds, dense_param: bounds.dense(dense_param))









#### TODO

# TODO: implement this in a different class (both for the old divide and the new divide)
class DivideOp:
    pass

# TODO: some layer such as LayerNormOp have suboperations, but those aren't wrapped in an Op yet
# TODO: I have to deal with that, otherwise there's going to be no caching for that
# TODO: the softmaxOp is a bit of an outside class (not using the same code as above), and maybe it shouldn't

####




class SoftmaxOp:
    def __init__(self, bounds: Bounds, args):
        self.args = args
        self.num_elements = bounds.lb.shape
        bounds_exp = bounds.exp()
        bounds_sum = Bounds(
            bounds.args, bounds.p, bounds.eps,
            lw=torch.sum(bounds_exp.lw, dim=-1, keepdim=True).repeat(1, 1, 1, bounds.dim_out),
            uw=torch.sum(bounds_exp.uw, dim=-1, keepdim=True).repeat(1, 1, 1, bounds.dim_out),
            lb=torch.sum(bounds_exp.lb, dim=-1, keepdim=True).repeat(1, 1, bounds.dim_out),
            ub=torch.sum(bounds_exp.ub, dim=-1, keepdim=True).repeat(1, 1, bounds.dim_out),
        )
        self.lambdas = get_initial_lambdas(self.num_elements, device='cpu' if self.args.cpu else 'cuda')
        self.coeffs = self.to_device(get_hyperplanes(bounds_exp, bounds_sum, "divide"))
        self.need_to_adjust_hyperplanes = False
        self.best_lb = None
        self.best_ub = None

    def reset_best_lb_and_best_ub(self):
        self.best_ub = None
        self.best_lb = None

    def reset_lambdas(self):
        self.lambdas = get_initial_lambdas(self.num_elements, device='cpu' if self.args.cpu else 'cuda')

    def to_device(self, val):
        return val if self.args.cpu else val.cuda()

    def needs_to_adjust_hyperplanes(self):
        self.need_to_adjust_hyperplanes = True

    def forward(self, bounds: Bounds):
        bounds_exp = bounds.exp()
        bounds_sum = Bounds(
            bounds.args, bounds.p, bounds.eps,
            lw=torch.sum(bounds_exp.lw, dim=-1, keepdim=True).repeat(1, 1, 1, bounds.dim_out),
            uw=torch.sum(bounds_exp.uw, dim=-1, keepdim=True).repeat(1, 1, 1, bounds.dim_out),
            lb=torch.sum(bounds_exp.lb, dim=-1, keepdim=True).repeat(1, 1, bounds.dim_out),
            ub=torch.sum(bounds_exp.ub, dim=-1, keepdim=True).repeat(1, 1, bounds.dim_out),
        )

        if self.need_to_adjust_hyperplanes:
            update_hyperplanes(self.coeffs, bounds_exp, bounds_sum, "divide")

        return self.divide(bounds_exp, bounds_sum)

    def divide(self, num: Bounds, denom: Bounds) -> Bounds:
        computed_bounds = bound_with_convex_combination(num, denom, self.coeffs, self.lambdas)
        if self.best_lb is not None:
            self.best_lb = torch.max(self.best_lb, computed_bounds.lb)
            self.best_ub = torch.min(self.best_ub, computed_bounds.ub)
        else:
            self.best_lb, self.best_ub = computed_bounds.lb, computed_bounds.ub

        computed_bounds.lb, computed_bounds.ub = self.best_lb, self.best_ub

        # Detach to try to avoid keeping increasing memory forever
        self.best_lb, self.best_ub = self.best_lb.detach(), self.best_ub.detach()
        return computed_bounds


def print_bounds(val, name):
    low, up = val.concretize()
    print("%s_l:  min - %s, max - %s" % (name, low.min(), low.max()))
    print("%s_u:  min - %s, max - %s" % (name, up.min(), up.max()))




