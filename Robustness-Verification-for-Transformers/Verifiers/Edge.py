# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
import math
from typing import Union, List, Optional

import torch

from Verifiers.BacksubstitutionComputer import BacksubstitutionComputer
from Verifiers.ConvexCombination import get_initial_lambdas, get_convexed_coeffs, get_hyperplanes_from_concrete_bounds_dim2, \
    get_bounds_in_nice_format, get_initial_lambdas_relu, get_hyperplanes_from_concrete_bounds_relu, get_convexed_coeffs_relu
from Verifiers.Layer import Layer
from Verifiers.autograd import compute_but_replace_grad_fn

epsilon = 1e-12


# IMPORTANT: if change here, also change in Edge.py
INPLACE = False  # True


def get_bounds_xy(l_x, u_x, l_y, u_y):
    alpha_l = l_y
    beta_l = l_x
    gamma_l = -alpha_l * beta_l

    alpha_u = u_y
    beta_u = l_x
    gamma_u = -alpha_u * beta_u

    return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u


def to_dev(val: torch.Tensor, args) -> torch.Tensor:
    if args.cpu:
        return val
    else:
        return val.cuda()


class Edge:
    def __init__(self, args, controller: BacksubstitutionComputer):
        self.args = args
        self.controller = controller
        self.use_forward = ("baf" in args.method)
        self.empty_cache = args.empty_cache
        self.lambdas_index = None

    def get_parents(self) -> List[Layer]:
        optional_par1 = getattr(self, 'par', None)
        optional_par2 = getattr(self, 'par1', None)
        optional_par3 = getattr(self, 'par2', None)
        parents = [optional_par1, optional_par2, optional_par3]
        parents = [parent for parent in parents if parent is not None]
        assert 0 <= len(parents) <= 2, "Layer has more than 2 parents!"
        return parents

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        raise NotImplementedError

    def to_dev(self, val: torch.Tensor) -> torch.Tensor:
        return to_dev(val, self.args)

    def do_op(self, f, *args):
        return compute_but_replace_grad_fn(f, *args, with_gpu=(not self.args.cpu))

    def build_copy_from_parents(self, *new_parents: Layer) -> "Edge":
        raise NotImplementedError("Subclass should implement this")

    def has_lambdas(self):
        return False

    def params_depend_on_input_bounds(self):
        return False

    def get_lambdas_index(self):
        assert self.has_lambdas(), "Requesting lambdas_index for a layer with no lambdas"
        assert self.lambdas_index is not None, "Requesting lambdas_index for a layer with no lambdas"
        return self.lambdas_index

    def set_lambdas_index(self, lambdas_index: int):
        assert self.has_lambdas(), "Setting index to an Edge even though the layer doesn't have lambdas"
        assert self.lambdas_index is None, "Setting index to an Edge that already has an index"
        self.lambdas_index = lambdas_index


class EdgeComplex(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer):
        super(EdgeComplex, self).__init__(args, controller)
        self.res: Optional[Layer] = None  # Should be filled by created by the subclasses

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        """ Add the bounds of this results edge (which may be the result of multiply operations)
        to the weights matrix of the result. For example, in a divide, the result is a complex thing
        with a reciprocal and a multiply """
        self.res.backward_buffer(lw, uw)


class EdgeDirect(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeDirect, self).__init__(args, controller)
        self.par = par

    def backward(self, lw, uw):
        """ Add the bounds of this linear edge to the weights matrix of the parent """
        self.par.backward_buffer(lw, uw)

    def build_copy_from_parents(self, par: Layer):
        return EdgeDirect(self.args, self.controller, par)


class EdgeInput(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, embeddings: torch.Tensor, index: Union[int, List[int]]):
        super(EdgeInput, self).__init__(args, controller)
        self.embeddings = embeddings
        self.index = index
        self.perturbed_words = args.perturbed_words

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        if self.use_forward:
            if self.args.attack_type == "synonym":
                # lw:         (input/output len, output dim, input dim)
                length = lw.shape[0]
                dim = lw.shape[2]
                self.controller.final_lw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], length * dim))
                self.controller.final_uw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], length * dim))

                for i in range(self.perturbed_words):
                    # Pertubed Word 1 (len, output dim, total relevant input dim)
                    self.controller.final_lw[i, :, i*dim:(i+1)*dim] = lw[i, :, :]
                    self.controller.final_uw[i, :, i*dim:(i+1)*dim] = uw[i, :, :]

                # They multiply the embedding by the weight coefficient and then sum everything up
                # to get the bias term
                # _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                # _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)

                # lw: [28, 2, 128]
                # embeddings: [28, 128]
                # result: [28, 2]
                # [28, 2, 128] x [28, 128, 1]
                _lb = torch.matmul(lw, self.embeddings.unsqueeze(2)).squeeze()
                _ub = torch.matmul(uw, self.embeddings.unsqueeze(2)).squeeze()

                # _lb = torch.matmul(self.embeddings.unsqueeze(1), lw).squeeze()
                # _ub = torch.matmul(self.embeddings.unsqueeze(1), uw).squeeze()
            elif self.perturbed_words == 2:
                assert (type(self.index) == list), "2 perturbed words and EdgeInput.index is not a list!"
                dim = lw.shape[2]
                self.controller.final_lw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], dim * 2))
                self.controller.final_uw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], dim * 2))
                # Pertubed Word 1 (len, output dim, total relevant input dim)
                # TODO: there was a bug here! They had lw repeated 4 times
                self.controller.final_lw[self.index[0], :, :dim] = lw[self.index[0], :, :]
                self.controller.final_uw[self.index[0], :, :dim] = uw[self.index[0], :, :]
                # Perturbed Word 2
                self.controller.final_lw[self.index[1], :, dim:] = lw[self.index[1], :, :]
                self.controller.final_uw[self.index[1], :, dim:] = uw[self.index[1], :, :]
                # They multiply the embedding by the weight coefficient and then sum everything up
                # to get the bias term

                # embeddings: (input/output len, input dim)
                # lw:         (input/output len, output dim, input dim)

                # This line is not correct!
                # _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                # _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)

                _lb = torch.matmul(self.embeddings.unsqueeze(1), lw).squeeze()
                _ub = torch.matmul(self.embeddings.unsqueeze(1), uw).squeeze()
            elif self.perturbed_words == 1:
                assert (type(self.index) == int), "1 perturbed word and EdgeInput.index is not an int!"

                if INPLACE:
                    self.controller.final_lw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]))
                    self.controller.final_uw = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2]))
                    self.controller.final_lw[self.index, :, :] = lw[self.index, :, :]
                    self.controller.final_uw[self.index, :, :] = uw[self.index, :, :]
                else:
                    # Make a copy of lw, uw where everything is 0 except the values for the word we're perturbing (with position 'index')
                    with torch.no_grad():
                        mask = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], lw.shape[2], requires_grad=False))
                        mask.data[self.index, :, :] = 1
                    self.controller.final_lw = mask * lw
                    self.controller.final_uw = mask * uw

                _lb = torch.sum(self.embeddings.unsqueeze(1) * lw, dim=-1)
                _ub = torch.sum(self.embeddings.unsqueeze(1) * uw, dim=-1)
            else:
                raise NotImplementedError
        else:
            assert (type(self.index) == int), "1 perturbed word and EdgeInput.index is not an int!"
            # lw/uw: (output length, output dim, embedding length, embedding dim)

            if self.args.num_perturbed_words == 1:
                self.controller.final_lw = lw[:, :, self.index, :]
                self.controller.final_uw = uw[:, :, self.index, :]
            else:
                self.controller.final_lw = lw.reshape(lw.size(0), lw.size(1), -1)
                self.controller.final_uw = uw.reshape(uw.size(0), uw.size(1), -1)

            _lb = torch.sum(lw * self.embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            _ub = torch.sum(uw * self.embeddings.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

        # They add the computed lower/upper bounds biases to the controller's biases
        if INPLACE:
            self.controller.lb += _lb
            self.controller.ub += _ub
        else:
            self.controller.lb = self.controller.lb + _lb
            self.controller.ub = self.controller.ub + _ub

        self.controller.final_lb = self.controller.lb.reshape(_lb.shape).clone()
        self.controller.final_ub = self.controller.ub.reshape(_lb.shape).clone()

        # They concretize the concroller's bounds (without the bias) and then add that to the biases of the controller
        # This defines the bounds after the backsubstituion
        l, u = self.controller.concretize(self.controller.final_lw, self.controller.final_uw)
        l = l.reshape(_lb.shape)
        u = u.reshape(_lb.shape)
        # print("l shape: ", l.shape)

        if INPLACE:
            self.controller.lb += l
            self.controller.ub += u
        else:
            self.controller.lb = self.controller.lb + l
            self.controller.ub = self.controller.ub + u

        if self.empty_cache:
            torch.cuda.empty_cache()

    def build_copy_from_parents(self):
        return EdgeInput(self.args, self.controller, self.embeddings, self.index)


# TODO: has lambdas (think about it)
class EdgeSoftmax(EdgeComplex):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer, num_attention_heads: int,
                 lambdas: torch.Tensor = None, coeffs: Optional[torch.Tensor] = None):
        super(EdgeSoftmax, self).__init__(args, controller)
        self.length = par.length
        self.num_attention_heads = num_attention_heads
        self.par = par

        self.update_exp_and_sum()

        # self.divide_op = EdgeDivideHyperplanes(self.args, self.controller, self.exp, self.sum, lambdas, coeffs)
        # self.res = self.exp.next(self.divide_op)
        self.res = self.exp.next(EdgeDivide(self.args, self.controller, self.exp, self.sum))

    def build_copy_from_parents(self, par: Layer):
        # TODO
        raise Exception("TODO: implement EdgeSoftmax from parents (special case, since lambdas are used here)")

    def params_depend_on_input_bounds(self):
        return True

    @property
    def lambdas(self):
        return self.divide_op.lambdas

    def set_lambdas(self, current_lambdas):
        self.divide_op.lambdas = current_lambdas

    def update_exp_and_sum(self):
        num_attention_heads = self.num_attention_heads
        self.exp = self.par.next(EdgeExp(self.args, self.controller, self.par))
        if self.use_forward:
            raise NotImplementedError
        self.sum = self.exp.next(EdgeSum.build_sum_edge(self.args, self.controller, self.exp, num_attention_heads))


class EdgeSum:
    @staticmethod
    def build_sum_edge(args, controller: BacksubstitutionComputer, par: Layer, num_attention_heads: int):
        length = par.length
        ones = to_dev(torch.ones(1, length, length), args)
        zeros = to_dev(torch.zeros(num_attention_heads, length, length), args)
        # This creates a matrix of the form (for length = 3 and num_attention_heads = 4)
        # [[1 1 1 0 0 0 0 0 0 0 0 0],
        #  [1 1 1 0 0 0 0 0 0 0 0 0],
        #  [1 1 1 0 0 0 0 0 0 0 0 0],
        #  [0 0 0 1 1 1 0 0 0 0 0 0],
        #  [0 0 0 1 1 1 0 0 0 0 0 0],
        #  [0 0 0 1 1 1 0 0 0 0 0 0],
        #  [0 0 0 0 0 0 1 1 1 0 0 0],
        #  [0 0 0 0 0 0 1 1 1 0 0 0],
        #  [0 0 0 0 0 0 1 1 1 0 0 0],
        #  [0 0 0 0 0 0 0 0 0 1 1 1],
        #  [0 0 0 0 0 0 0 0 0 1 1 1],
        #  [0 0 0 0 0 0 0 0 0 1 1 1]]
        w = torch.cat([
            ones,
            torch.cat([zeros, ones], dim=0).repeat(num_attention_heads - 1, 1, 1)
        ], dim=0) \
            .reshape(num_attention_heads, num_attention_heads, length, length) \
            .permute(0, 2, 1, 3) \
            .reshape(num_attention_heads * length, num_attention_heads * length)
        return EdgeDense(args, controller, par, w=w, b=0.)




class EdgeJoiner(Edge):
    """
    This Edge is used in the softmax, where we split the tensor in several blocks and process it one at the time.
    This Edge then receives all of the chunks one by one and transmits them to the parent layer in the right position
    It allows other edges to be unaware the splitting has happened.
    """

    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer, num_splits: int, total_dim_size: int, join_dim: int):
        super(EdgeJoiner, self).__init__(args, controller)
        self.par = par
        self.num_splits = num_splits
        self.num_children_processed = 0
        self.total_dim_size = total_dim_size
        self.join_dim = join_dim

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        """ Add the bounds of this linear edge to the weights matrix of the parent """
        if self.par.lw is None:  # Means we're calling this for the first time / resetting
            self.num_children_processed = 0

            # We need to ensure this has the right shape
            # It's a bit complicated by the fact that at different points it receives tensors of different sizes:
            # 1) When we just added the EdgeJoiner and do backwards, it's going to receive a tensor where the join
            #    dimension has the whole size last_size
            # 2) When we're at a later layer and do backwards, we only receive part of the input and so the join
            #    dimension will be last_size / num_childs
            desired_shape = list(lw.shape)
            desired_shape[self.join_dim] = self.total_dim_size

            self.par.setup_buffers(desired_shape)

        assert self.num_children_processed < self.num_splits, \
            "EdgeJoiner: self.num_children_processed (%d) >= self.num_splits (%d) - dim %s" % \
            (self.num_children_processed, self.num_splits, self.join_dim)

        self.par.backward_buffer_at_pos(lw, uw, pos=self.num_children_processed, dim=self.join_dim)
        self.num_children_processed += 1

        # This is needed when two EdgeJoiner are in a row
        if self.num_children_processed == self.num_splits:  # Reset for the next backwards pass
            self.num_children_processed = 0

    def build_copy_from_parents(self, par: Layer):
        return EdgeJoiner(self.args, self.controller, par, self.num_splits, self.total_dim_size, self.join_dim)


class EdgeSoftmaxNew(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer, num_attention_heads: int):
        super(EdgeSoftmaxNew, self).__init__(args, controller)

        self.length = par.length
        self.num_attention_heads = num_attention_heads

        self.par = par

        # To encode the softmax, the following approach will be taken:
        # softmax_i(x1, ..., xn) = 1 / ((exp(x1) + ... exp(xn)) / exp(xi))
        #                        = 1 / ((exp(x1 - xi) + ... exp(xn - xi))
        # so we will have to have subtractions, exponentials and in the top formulation also divisions
        # We'll begin by using the bottom formulation

        # Dimension = Length * Length because there's Length attention score per query and there's Length queries
        # Problem: doing everything at once requires squaring the number of outputs (each pair of differences)
        # This leads to huge tensors (60GB) which aren't runnable on GPU and are probably very slow
        # Instead, it's better to split things up.
        # If there are 30 words and 4 attention heads, then for each word there are going to be 4
        # queries and we'll have to do a dot product between each query and all 30 keys. In the naive version,
        # we have thus 30 x 4 x 30 dot products = 30 x 120 dot products.
        # In the naive version, we compute the difference between each dot product, and so obtain a 30 x 120 x 120
        # outputs which is a lot. The first and obvious approach is to treat the softmaxes for each word one at
        # the time, e.g. split the 30 x 120 x 120 matrix into 30 tensors of size 1 x 120 x 120 and treat them
        # one by one. Since the dot products between different heads are not related, we could improve
        # efficiency further by splitting the 120 into 4 x 30, which would be much better (4 x 30 x 30) is
        # better than (120 x 120), we save a factor of 4
        # The only hard part here is that we have to split thing and then reunite them
        self.dim = dim = self.length

        # Weights and bias for step 1
        self.weights_diff = self.to_dev(torch.zeros(self.dim * self.dim, self.dim))
        CONSTANT_IN_EXP = -par.l.min()
        self.bias_diff = self.to_dev(torch.ones(self.dim * self.dim)) * CONSTANT_IN_EXP
        for i in range(self.dim):
            # The values for the i-th term are store in rows i*length:(i + 1)*length
            for j in range(self.dim):
                # The value xj - xi is stored at index i * dim + j
                # Note: in the special case where i = j, we end up with 1 - 1 = 0 = xi - xi
                self.weights_diff[i * self.dim + j][j] += 1
                self.weights_diff[i * self.dim + j][i] += -1

        # Weights and bias for step 3
        self.weights_sum_exp = self.to_dev(torch.zeros(self.dim, self.dim * self.dim))
        self.bias_sum_exp = self.to_dev(torch.zeros(self.dim))
        for i in range(self.dim):  # The i-th row has all the exp(xj - xi) terms
            self.weights_sum_exp[i, i * self.dim:(i + 1) * self.dim] = 1

        # Create the layer for the attention scores of each word one at the time
        # They will have size length x num_attention_heads
        self.attention_head_joiners = []
        self.diffs = [[] for _ in range(self.length)]
        self.exp_diffs = [[] for _ in range(self.length)]
        self.sum_exp_diffs = [[] for _ in range(self.length)]
        self.results = [[] for _ in range(self.length)]
        self.word_joiner = self.par.next(
            EdgeJoiner(self.args, self.controller, self.par,
                       num_splits=self.length, total_dim_size=self.length, join_dim=-2),
            length=self.length,
            dim=self.length * self.num_attention_heads
        )

        for word_num in range(self.length):
            attention_head_joiner = self.word_joiner.next(
                EdgeJoiner(self.args, self.controller, self.word_joiner, num_splits=self.num_attention_heads,
                           total_dim_size=self.length * self.num_attention_heads, join_dim=-1),
                length=1,
                dim=self.length * self.num_attention_heads
            )
            self.attention_head_joiners.append(attention_head_joiner)

            for attention_head in range(self.num_attention_heads):
                # Step 1: compute all the xj - xi
                # We treat only the current word and the current attention, so the new lw shape is A x B x 1 x (30 * 30)
                diff = self.attention_head_joiners[word_num].next(
                    EdgeDense(self.args, self.controller, self.attention_head_joiners[word_num],
                              w=self.weights_diff, b=self.bias_diff),
                    length=1,
                    dim=dim * dim
                )
                self.diffs[word_num].append(diff)
                # print("lb(min):", diff.final_lb.min())
                # print("lb(max):", diff.final_lb.max())
                # print("ub(min):", diff.final_ub.min())
                # print("ub(max):", diff.final_ub.max())

                # Step 2: compute all the exp(xj - xi)
                exp_diff = diff.next(EdgeExp(self.args, self.controller, diff), length=1, dim=dim * dim)
                self.exp_diffs[word_num].append(exp_diff)

                assert (torch.min(exp_diff.l) >= 0), "Exp: some values of the concrete lower bounds are negative %s" % exp_diff.l.min()

                # Step 3: sum all the exp(xj - xi) for each i
                sum_exp_diff = exp_diff.next(
                    EdgeDense(self.args, self.controller, exp_diff, w=self.weights_sum_exp, b=self.bias_sum_exp),
                    length=1,
                    dim=dim
                )
                self.sum_exp_diffs[word_num].append(sum_exp_diff)

                assert (torch.min(sum_exp_diff.l) >= 0), "SumExp: some values of the concrete lower bounds are negative"

                # Step 4: To compensate the +C in the exps, we need to multiply exp(-C) to cancel the effect.
                if CONSTANT_IN_EXP == 0:
                    adjusted_sum_exp_diffs = sum_exp_diff
                else:
                    adjusted_sum_exp_diffs = sum_exp_diff.next(
                        EdgeLinear(self.args, self.controller, sum_exp_diff, w=math.exp(-CONSTANT_IN_EXP), b=0.),
                        length=1,
                        dim=dim
                    )

                # Step 5: Compute the inverse for all of these sums, thus obtaining all the softmax values
                res = adjusted_sum_exp_diffs.next(
                    EdgeReciprocal(self.args, self.controller, adjusted_sum_exp_diffs),
                    length=1,
                    dim=dim)
                self.results[word_num].append(res)

    def backward(self, lw, uw):
        """ Add the bounds of this results edge (which may be the result of multiply operations)
        to the weights matrix of the result. For example, in a divide, the result is a complex thing
        with a reciprocal and a multiply """
        for word_num in range(self.length):
            for n_head in range(self.num_attention_heads):
                # We need to do a backwards pass
                # lw and uw have the shape: A x B x length x (n_attention_heads * length)
                # So if length = 30 and n_attention_heads = 4, then it's A x B x 30 x 120
                # This is true because the lw matrices have shape Output x Input
                # We treat the current word only, so lw for the word will have shape A x B x 1 x 120
                # Additionally, we want to only treat the current attention head for that word
                # so the lw will actually be A x B x 1 x 30
                # however, we need to pick the relevant part
                # Implementation note: we use word_num:word_num+1 instead of word_num to preserve that dimension
                self.results[word_num][n_head].backward_buffer(
                    lw[..., word_num:word_num + 1, n_head * self.length:(n_head + 1) * self.length],
                    uw[..., word_num:word_num + 1, n_head * self.length:(n_head + 1) * self.length]
                )

    def build_copy_from_parents(self, par: Layer):
        return EdgeSoftmaxNew(self.args, self.controller, par, self.num_attention_heads)


class EdgePooling(Edge):
    def __init__(self, args, controller, par):
        super(EdgePooling, self).__init__(args, controller)

        self.par = par
        self.length = par.length

    def backward(self, lw, uw):
        # TODO: not sure why they add all these zero's (and what the sizes correspond to in the
        # forward and backwards case)
        if self.use_forward:
            dim = 0
            zeros = self.to_dev(torch.zeros(self.length - 1, lw.shape[1], lw.shape[2]))
        else:
            dim = 2
            zeros = self.to_dev(torch.zeros(lw.shape[0], lw.shape[1], self.length - 1, lw.shape[3]))
        lw = torch.cat([lw, zeros], dim=dim)
        uw = torch.cat([uw, zeros], dim=dim)
        self.par.backward_buffer(lw, uw)

    def build_copy_from_parents(self, par: Layer):
        return EdgePooling(self.args, self.controller, par)


class EdgeDense(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer,
                 w: Union[float, torch.Tensor] = 0.0, b: Union[float, torch.Tensor] = 0.0, dense=None):
        super(EdgeDense, self).__init__(args, controller)
        self.par = par
        if dense is not None:
            w = dense.weight
            b = dense.bias
        self.w = w
        if type(b) == float:
            self.b = self.to_dev(torch.ones(w.shape[-1])) * b
        else:
            self.b = b

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        # 1) Multiply the bounds weights by the matrix weights
        # 2) Add those bounds to the parent weights (not the bias)
        # 3) Add the bias terms created in this layer to the biases in the controller
        # TODO: it's surprising that here they don't do the positive / negative thing. Understand why
        # and check if it's a bug / problem
        if self.use_forward:
            if INPLACE:
                self.controller.lb += torch.sum(lw * self.b, dim=-1)
                self.controller.ub += torch.sum(uw * self.b, dim=-1)
            else:
                self.controller.lb = self.controller.lb + torch.sum(lw * self.b, dim=-1)
                self.controller.ub = self.controller.ub + torch.sum(uw * self.b, dim=-1)
        else:
            if INPLACE:
                self.controller.lb += torch.sum(lw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2])
                self.controller.ub += torch.sum(uw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2])
            else:
                self.controller.lb = self.controller.lb + torch.sum(lw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2])
                self.controller.ub = self.controller.ub + torch.sum(uw * self.b.reshape(1, 1, 1, -1), dim=[-1, -2])

        # (N, Output Dim, Input Dim) @ (Input Dim, Input2 Dim) = (N, Output Dim, Input2 Dim)
        _lw = torch.matmul(lw, self.w)
        # print("lw shape", lw.shape, "self.w shape", self.w.shape, "_lw shape", _lw.shape)
        _uw = torch.matmul(uw, self.w)
        return self.par.backward_buffer(_lw, _uw)

    def build_copy_from_parents(self, par: Layer):
        return EdgeDense(self.args, self.controller, par, self.w, self.b)


class EdgeActivation(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer, par2: Optional[Layer] = None):
        super(EdgeActivation, self).__init__(args, controller)
        self.par = par
        self.par2 = par2
        self.init_linear()

    def init_linear(self):
        self.mask_pos = torch.gt(self.par.l, 0).to(torch.float)
        self.mask_neg = torch.lt(self.par.u, 0).to(torch.float)
        self.mask_both = 1 - self.mask_pos - self.mask_neg

        # element-wise for now
        shape = (self.par.length, self.par.dim)
        self.lw = self.to_dev(torch.zeros(shape))
        self.lb = self.to_dev(torch.zeros(shape))
        self.uw = self.to_dev(torch.zeros(shape))
        self.ub = self.to_dev(torch.zeros(shape))

        if self.par2 is not None:
            shape = (self.par2.length, self.par2.dim)
            self.lw2 = self.to_dev(torch.zeros(shape))
            self.lb2 = self.to_dev(torch.zeros(shape))
            self.uw2 = self.to_dev(torch.zeros(shape))
            self.ub2 = self.to_dev(torch.zeros(shape))

    def params_depend_on_input_bounds(self):
        return True

    def add_linear(self, mask: Optional[torch.Tensor], type: str,
                   k: Union[int, float, torch.Tensor],
                   x0: Union[int, float, torch.Tensor],
                   y0: Union[int, float, torch.Tensor],
                   second=False):
        """ This is a line: k is the slope, x0 is the first point, y0 is the image at x0
        Therefore y = k (x - x0) + y0 = kx + (-x0 * k + y0).
        IMPORTANT: this method doesn't return anything, instead it multiplies the weights
        and bias matrices of the layer! """
        if mask is None:
            mask = 1
        if INPLACE:
            if type == "lower":
                if second:
                    w_out, b_out = self.lw2, self.lb2
                else:
                    w_out, b_out = self.lw, self.lb
            else:
                if second:
                    w_out, b_out = self.uw2, self.ub2
                else:
                    w_out, b_out = self.uw, self.ub
            w_out += mask * k
            b_out += mask * (-x0 * k + y0)
        else:
            if type == "lower":
                if second:
                    self.lw2 = self.lw2 + (mask * k)
                    self.lb2 = self.lb2 + mask * (-x0 * k + y0)
                else:
                    self.lw = self.lw + (mask * k)
                    self.lb = self.lb + mask * (-x0 * k + y0)
            else:
                if second:
                    self.uw2 = self.uw2 + (mask * k)
                    self.ub2 = self.ub2 + mask * (-x0 * k + y0)
                else:
                    self.uw = self.uw + (mask * k)
                    self.ub = self.ub + mask * (-x0 * k + y0)

    def backward_par(self, lw: torch.Tensor, uw: torch.Tensor,
                     self_lw: torch.Tensor, self_lb: torch.Tensor,
                     self_uw: torch.Tensor, self_ub: torch.Tensor, par: Layer):
        mask_l = torch.gt(lw, 0.).to(torch.float)
        mask_u = torch.gt(uw, 0.).to(torch.float)

        # The expression is a conditional one depending if lw/uw > 0 or not
        # They use masks (e.g. boolean matrices) to deal with that
        # They compute the biases and weights of the lower / upper bounds
        # and add the weights to the parent and the biases to the controller

        if self.use_forward:
            _lw = mask_l * lw * self_lw.unsqueeze(1) + \
                  (1 - mask_l) * lw * self_uw.unsqueeze(1)
            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(1) + \
                            (1 - mask_l) * lw * self_ub.unsqueeze(1), dim=-1)
            _uw = mask_u * uw * self_uw.unsqueeze(1) + \
                  (1 - mask_u) * uw * self_lw.unsqueeze(1)
            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(1) + \
                            (1 - mask_u) * uw * self_lb.unsqueeze(1), dim=-1)
        else:
            _lw = mask_l * lw * self_lw.unsqueeze(0).unsqueeze(0) + \
                  (1 - mask_l) * lw * self_uw.unsqueeze(0).unsqueeze(0)
            _lb = torch.sum(mask_l * lw * self_lb.unsqueeze(0).unsqueeze(0) + \
                            (1 - mask_l) * lw * self_ub.unsqueeze(0).unsqueeze(0), dim=[-1, -2])
            _uw = mask_u * uw * self_uw.unsqueeze(0).unsqueeze(0) + \
                  (1 - mask_u) * uw * self_lw.unsqueeze(0).unsqueeze(0)
            _ub = torch.sum(mask_u * uw * self_ub.unsqueeze(0).unsqueeze(0) + \
                            (1 - mask_u) * uw * self_lb.unsqueeze(0).unsqueeze(0), dim=[-1, -2])

        if INPLACE:  # PUTTING False here is enough to say it doesn't depend on the lambdas...
            self.controller.lb += _lb
            self.controller.ub += _ub
        else:
            self.controller.lb = self.controller.lb + _lb
            self.controller.ub = self.controller.ub + _ub

        par.backward_buffer(_lw, _uw)

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        self.backward_par(lw, uw, self.lw, self.lb, self.uw, self.ub, self.par)
        if self.par2 is not None:
            self.backward_par(lw, uw, self.lw2, self.lb2, self.uw2, self.ub2, self.par2)

        # cannot be combined with the forward framework


class EdgeDotProduct(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, a: Layer, b: Layer, num_attention_heads: int):
        super(EdgeDotProduct, self).__init__(args, controller)

        assert "baf" not in args.method, "EdgeDotProduct doesn't support BAF"

        # Given the bounds of a and b, compute the linear approximation weights (and biases) for each multiplication
        # and stores them

        # DO NOT REMOVE THE PAR1 and PAR2 ASSIGNMENT! THESE ARE USED IN THE GET_PARENTS() FUNCTION!
        self.par1 = self.a = a
        self.par2 = self.b = b
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = a.l.shape[-1] // num_attention_heads

        l_a = a.l.reshape(a.length, num_attention_heads, self.attention_head_size).repeat(1, 1, b.length).reshape(-1)
        u_a = a.u.reshape(a.length, num_attention_heads, self.attention_head_size).repeat(1, 1, b.length).reshape(-1)

        l_b = b.l.reshape(b.length, num_attention_heads, self.attention_head_size).transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)
        u_b = b.u.reshape(b.length, num_attention_heads, self.attention_head_size).transpose(0, 1).repeat(a.length, 1, 1).reshape(-1)

        self.alpha_l, self.beta_l, self.gamma_l, self.alpha_u, self.beta_u, self.gamma_u = get_bounds_xy(l_a, u_a, l_b, u_b)

        # batch_size, length, h, h_size*length
        self.alpha_l = self.alpha_l.reshape(a.length, num_attention_heads, b.length, self.attention_head_size)
        self.alpha_u = self.alpha_u.reshape(a.length, num_attention_heads, b.length, self.attention_head_size)

        self.beta_l = self.beta_l.reshape(a.length, num_attention_heads, b.length, self.attention_head_size)  # .transpose(0, 2)
        self.beta_u = self.beta_u.reshape(a.length, num_attention_heads, b.length, self.attention_head_size)  # .transpose(0, 2)

        # batch_size, length, h, length*h_size
        self.gamma_l = self.gamma_l.reshape(a.length, num_attention_heads, b.length, self.attention_head_size).sum(dim=-1)
        self.gamma_u = self.gamma_u.reshape(a.length, num_attention_heads, b.length, self.attention_head_size).sum(dim=-1)

    def params_depend_on_input_bounds(self):
        return True

    def build_copy_from_parents(self, par1: Layer, par2: Layer):
        return EdgeDotProduct(self.args, self.controller, par1, par2, self.num_attention_heads)

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        # [length, 1, h, length, r]
        alpha_l = self.alpha_l.unsqueeze(0).unsqueeze(0)
        alpha_u = self.alpha_u.unsqueeze(0).unsqueeze(0)
        beta_l = self.beta_l.unsqueeze(0).unsqueeze(0)
        beta_u = self.beta_u.unsqueeze(0).unsqueeze(0)
        gamma_l = self.gamma_l.reshape(1, 1, self.a.length, -1)
        gamma_u = self.gamma_u.reshape(1, 1, self.a.length, -1)

        mask = torch.gt(lw, 0.).to(torch.float)
        _lb = torch.sum(lw * (mask * gamma_l + (1 - mask) * gamma_u), dim=[-1, -2])
        self.controller.delete_tensor(mask)

        mask = torch.gt(uw, 0.).to(torch.float)
        _ub = torch.sum(uw * (mask * gamma_u + (1 - mask) * gamma_l), dim=[-1, -2])
        self.controller.delete_tensor(mask)
        self.controller.delete_tensor(gamma_l)
        self.controller.delete_tensor(gamma_u)

        if self.empty_cache:
            torch.cuda.empty_cache()

        if INPLACE:
            self.controller.lb += _lb
            self.controller.ub += _ub
        else:
            self.controller.lb = self.controller.lb + _lb
            self.controller.ub = self.controller.ub + _ub

        # [length, h * length (o), h, length, 1]
        _lw = lw.reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_lw, 0.).to(torch.float)
        _lw = torch.sum(mask * _lw * alpha_l + (1 - mask) * _lw * alpha_u, dim=-2).reshape(lw.shape[0], lw.shape[1],
                                                                                           lw.shape[2], -1)

        # [length, h * length (o), h, length, 1]
        _uw = uw.reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_uw, 0.).to(torch.float)
        _uw = torch.sum(mask * _uw * alpha_u + (1 - mask) * _uw * alpha_l, dim=-2).reshape(uw.shape[0], uw.shape[1],
                                                                                           uw.shape[2], -1)

        self.controller.delete_tensor(mask)
        if self.empty_cache:
            torch.cuda.empty_cache()

        self.a.backward_buffer(_lw, _uw)

        self.controller.delete_tensor(_lw)
        self.controller.delete_tensor(_uw)
        if self.empty_cache:
            torch.cuda.empty_cache()

        _lw2 = lw.reshape(lw.shape[0], lw.shape[1], lw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_lw2, 0.).to(torch.float)
        _lw2 = torch.sum(mask * _lw2 * beta_l + (1 - mask) * _lw2 * beta_u, dim=-4).transpose(2, 3)
        _lw2 = _lw2.reshape(_lw2.shape[0], _lw2.shape[1], _lw2.shape[2], -1)

        _uw2 = uw.reshape(uw.shape[0], uw.shape[1], uw.shape[2], self.num_attention_heads, self.b.length, 1)
        mask = torch.gt(_uw2, 0.).to(torch.float)
        _uw2 = torch.sum(mask * _uw2 * beta_u + (1 - mask) * _uw2 * beta_l, dim=-4).transpose(2, 3)
        _uw2 = _uw2.reshape(_uw2.shape[0], _uw2.shape[1], _uw2.shape[2], -1)

        self.controller.delete_tensor(mask)
        if self.empty_cache:
            torch.cuda.empty_cache()

        self.b.backward_buffer(_lw2, _uw2)


class EdgeTranspose(Edge):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer, num_attention_heads: int):
        super(EdgeTranspose, self).__init__(args, controller)

        assert ("baf" not in args.method), "EdgeTranspose doesn't support BAF"

        self.par = par
        self.num_attention_heads = num_attention_heads

    def build_copy_from_parents(self, par: Layer):
        return EdgeTranspose(self.args, self.controller, par, self.num_attention_heads)

    def transpose(self, w: torch.Tensor):
        w = w.reshape(
            w.shape[0], w.shape[1], w.shape[2],
            self.num_attention_heads, -1
        ).transpose(2, 4)
        w = w.reshape(w.shape[0], w.shape[1], w.shape[2], -1)
        return w

    def backward(self, lw: torch.Tensor, uw: torch.Tensor):
        lw = self.transpose(lw)
        uw = self.transpose(uw)

        self.par.backward_buffer(lw, uw)


class EdgeMultiply(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, a: Layer, b: Layer):
        super(EdgeMultiply, self).__init__(args, controller, par=a, par2=b)

        alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u = get_bounds_xy(
            a.l.reshape(-1),
            a.u.reshape(-1),
            b.l.reshape(-1),
            b.u.reshape(-1)
        )
        alpha_l = alpha_l.reshape(a.l.shape)
        beta_l = beta_l.reshape(a.l.shape)
        gamma_l = gamma_l.reshape(a.l.shape)
        alpha_u = alpha_u.reshape(a.l.shape)
        beta_u = beta_u.reshape(a.l.shape)
        gamma_u = gamma_u.reshape(a.l.shape)

        self.add_linear(mask=None, type="lower", k=alpha_l, x0=0, y0=gamma_l)
        self.add_linear(mask=None, type="lower", k=beta_l, x0=0, y0=0, second=True)
        self.add_linear(mask=None, type="upper", k=alpha_u, x0=0, y0=gamma_u)
        self.add_linear(mask=None, type="upper", k=beta_u, x0=0, y0=0, second=True)

    def build_copy_from_parents(self, par1: Layer, par2: Layer):
        return EdgeMultiply(self.args, self.controller, par1, par2)


class EdgeSqrt(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeSqrt, self).__init__(args, controller, par)

        assert (torch.min(self.par.l) >= 0), "EdgeSqrt: some values of the parent lower bounds are negative"
        k = (torch.sqrt(self.par.u) - torch.sqrt(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=None, type="lower", k=k, x0=self.par.l, y0=torch.sqrt(self.par.l) + epsilon)
        m = (self.par.l + self.par.u) / 2
        k = 0.5 / torch.sqrt(m)
        self.add_linear(mask=None, type="upper", k=k, x0=m, y0=torch.sqrt(m) + epsilon)

    def build_copy_from_parents(self, par: Layer):
        return EdgeSqrt(self.args, self.controller, par)


class EdgeReciprocal(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeReciprocal, self).__init__(args, controller, par)

        assert (torch.min(self.par.l)), "EdgeSqrt: some values of the parent lower bounds are 0"
        # Same formulas as mine (https://www.desmos.com/calculator/f5clkmmyuh), but written differently
        m = (self.par.l + self.par.u) / 2
        kl = -1 / m.pow(2)
        self.add_linear(mask=None, type="lower", k=kl, x0=m, y0=1. / m)

        ku = -1. / (self.par.l * self.par.u)
        self.add_linear(mask=None, type="upper", k=ku, x0=self.par.l, y0=1. / self.par.l)

    def build_copy_from_parents(self, par: Layer):
        return EdgeReciprocal(self.args, self.controller, par)


class EdgeLinear(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer,
                 w: Union[float, torch.Tensor], b: Union[float, torch.Tensor]):
        super(EdgeLinear, self).__init__(args, controller, par)
        self.w = w
        self.b = b
        self.add_linear(mask=None, type="lower", k=w, x0=0., y0=b)
        self.add_linear(mask=None, type="upper", k=w, x0=0., y0=b)

    def build_copy_from_parents(self, par: Layer):
        return EdgeLinear(self.args, self.controller, par, self.w, self.b)


class EdgeSqr(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeSqr, self).__init__(args, controller, par)

        k = self.par.u + self.par.l
        self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=self.par.l.pow(2))
        m = torch.max((self.par.l + self.par.u) / 2, 2 * self.par.u)
        self.add_linear(mask=self.mask_neg, type="lower", k=2 * m, x0=m, y0=m.pow(2))
        m = torch.min((self.par.l + self.par.u) / 2, 2 * self.par.l)
        self.add_linear(mask=self.mask_pos, type="lower", k=2 * m, x0=m, y0=m.pow(2))

    def build_copy_from_parents(self, par: Layer):
        return EdgeSqr(self.args, self.controller, par)


class EdgeExp(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeExp, self).__init__(args, controller, par)

        if not self.args.use_new_exp:
            m = torch.min((self.par.l + self.par.u) / 2, self.par.l + 0.99)
            k = torch.exp(m)
            self.add_linear(mask=None, type="lower", k=k, x0=m, y0=torch.exp(m))

            k = (torch.exp(self.par.u) - torch.exp(self.par.l)) / (self.par.u - self.par.l + epsilon)
            self.add_linear(mask=None, type="upper", k=k, x0=self.par.l, y0=torch.exp(self.par.l))
        else:
            # New version: we want the lower bound to always be positive (this will be useful for the reciprocal)
            # and below e^x in the range [l, u]. One simple way to achieve this is make the line start from (l, 0)
            # and the grow upwards. The line will therefore be of the form y = a * (x -l) where a is the slope.
            #
            # We therefore want
            #   y = a (x - l) <= e^x, l <= x <= u
            # and so
            #   a (x - l) <= e^x, l <= x <= u
            #   a <= g(x) = e^x / (x - l),  l <= x <= u
            # We neee the minimum value of g(x) in the range [l, u]. The derivative of g is 0 at point l + 1. Therefore,
            # if that point is in the range, we can use g(l + 1) = e^(l + 1) as the slope. Otherwise, if the distance
            # between u and l is smaller than 1, then have to check both sides of the range.
            # g(l) = e^l / (l - l) = +- infinity   <- no good
            # g(u) = e^u / (u - l)                 <- good
            # So we'll pick g(u) as the slope in that case. This slope corresponds to the line that goes from (l, 0)
            # to (u, e^u). There's a final special case, which is when the u = l. This corresponds to exp(constant).
            # In that case, we can simply use a slope of 0 and value of exp(l).
            # So, in summary, the equations is:
            #    1) slope: 0 and y0 = exp(v), if u = l
            #    2) slope: e^(l + 1)  and  y0 = 0, if u - l >= 1
            #    3) slope: e^u / (u - l) and y0 = 0, if u - l <= 1 and u != l

            if False:
                equality = (self.par.u == self.par.l).float()
                distance_gt_1 = ((self.par.u - self.par.l) >= 1).float()

                y_case_1 = self.par.l.exp()
                slope_case_2 = (self.par.l + 1.0).exp()
                # Note: we don't want to divide by 0 and get nans, so we add an epsilon in the denominator when u = l
                slope_case_3 = self.par.u.exp() / (self.par.u - self.par.l + epsilon * equality)

                # w = (1 - equality) * distance_gt_1 * slope_case_2 + (1 - equality) * (1 - distance_gt_1) * slope_case_3
                # bias = y_case_1 * equality

                self.add_linear(mask=equality, type="lower", k=0., x0=0., y0=y_case_1)
                self.add_linear(mask=distance_gt_1, type="lower", k=slope_case_2, x0=self.par.l, y0=0.) #epsilon)  # TODO: added epsilon, adjust slope
                self.add_linear(mask=(1 - equality) * (1 - distance_gt_1), type="lower", k=slope_case_3, x0=self.par.l, y0=0.) # epsilon)  # TODO: same here

                k = (torch.exp(self.par.u) - torch.exp(self.par.l)) / (self.par.u - self.par.l + epsilon)
                self.add_linear(mask=equality, type="upper", k=0., x0=0., y0=y_case_1)
                self.add_linear(mask=1-equality, type="upper", k=k, x0=self.par.l, y0=torch.exp(self.par.l))
            else:
                equality = (self.par.u == self.par.l).float()
                y_case_1 = self.par.l.exp()

                self.add_linear(mask=None, type="lower", k=0., x0=0., y0=y_case_1)

                k = (torch.exp(self.par.u) - torch.exp(self.par.l)) / (self.par.u - self.par.l + epsilon)
                self.add_linear(mask=equality, type="upper", k=0., x0=0., y0=y_case_1)
                self.add_linear(mask=1 - equality, type="upper", k=k, x0=self.par.l, y0=torch.exp(self.par.l))

    def build_copy_from_parents(self, par: Layer):
        return EdgeExp(self.args, self.controller, par)


class EdgeDivide(EdgeComplex):
    def __init__(self, args, controller: BacksubstitutionComputer, a: Layer, b: Layer):
        super(EdgeDivide, self).__init__(args, controller)
        self.par1 = a
        self.par2 = b

        b_reciprocal = b.next(EdgeReciprocal(args, controller, b))
        self.res = a.next(EdgeMultiply(args, controller, a, b_reciprocal))

    def params_depend_on_input_bounds(self):
        return True

    def build_copy_from_parents(self, par1: Layer, par2: Layer):
        return EdgeDivide(self.args, self.controller, par1, par2)


class EdgeDivideHyperplanes(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, num: Layer, denom: Layer,
                 lambdas: torch.Tensor, coeffs: Optional[torch.Tensor], lambdas_index: int = None):
        super(EdgeDivideHyperplanes, self).__init__(args, controller, num, denom)
        if lambdas is None:
            lambdas = get_initial_lambdas(num.l.shape, 'cpu' if args.cpu else 'cuda')
        self.lambdas = lambdas
        self.set_lambdas_index(lambdas_index)

        if coeffs is None:
            print("Divide Hyperplanes - Creating hyperplane coefficients")
            lx, ly, ux, uy = get_bounds_in_nice_format(num.l, denom.l, num.u, denom.u)
            self.coeffs = self.to_dev(get_hyperplanes_from_concrete_bounds_dim2(lx, ly, ux, uy, operation_name="divide"))
        else:
            print("Divide Hyperplanes - Re-using hyperplanes coefficients from storage")
            self.coeffs = coeffs

        Al, Au, Bl, Bu, Cl, Cu = get_convexed_coeffs(self.coeffs, self.lambdas)
        self.add_linear(mask=None, type="lower", k=Al, x0=0, y0=Cl)
        self.add_linear(mask=None, type="lower", k=Bl, x0=0, y0=0, second=True)
        self.add_linear(mask=None, type="upper", k=Au, x0=0, y0=Cu)
        self.add_linear(mask=None, type="upper", k=Bu, x0=0, y0=0, second=True)

    def has_lambdas(self):
        return True

    def build_copy_from_parents(self, par1: Layer, par2: Layer, lambdas: torch.Tensor):
        return EdgeDivideHyperplanes(self.args, self.controller, par1, par2, lambdas, self.coeffs, self.lambdas_index)


def setup_relu_lambdas(lambdas: torch.Tensor, parent: Layer):
    lambdas.fill_(0.0)
    k = torch.gt(torch.abs(parent.u), torch.abs(parent.l)).to(torch.float)  # k = 1 -> slope = 1     ||   k = 0 -> slope = 0
    lambdas[:, :, 0] = 5 * (1 - k)   # slope = 0
    lambdas[:, :, 4] = 5 * k         # slope = 1


class EdgeReluHyperplanes(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer,
                 lambdas: torch.Tensor, coeffs: Optional[torch.Tensor], lambdas_index: int = None):
        super(EdgeReluHyperplanes, self).__init__(args, controller, par)

        assert lambdas is not None, "lambdas must already be initialized"
        self.lambdas = lambdas
        self.set_lambdas_index(lambdas_index)

        if coeffs is None:
            coeffs = self.to_dev(get_hyperplanes_from_concrete_bounds_relu(par.l, par.u))
        self.coeffs = coeffs

        self.add_linear(mask=self.mask_neg, type="lower", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_neg, type="upper", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="lower", k=1., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="upper", k=1., x0=0, y0=0)

        k = self.par.u / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=self.par.l, y0=0)

        # The difference compared to a normal ReLU is here
        k = get_convexed_coeffs_relu(self.coeffs, self.lambdas)
        print("Lower bound EdgeHyperplanes: nonzero - %s, nonzero used - %s" % (k.nonzero().nelement(), (k * self.mask_both).nonzero().nelement()))
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=0, y0=0)

    def has_lambdas(self):
        return True

    def build_copy_from_parents(self, par1: Layer, lambdas: torch.Tensor):
        return EdgeReluHyperplanes(self.args, self.controller, par1, lambdas, self.coeffs, self.lambdas_index)


class EdgeRelu(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeRelu, self).__init__(args, controller, par)

        self.add_linear(mask=self.mask_neg, type="lower", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_neg, type="upper", k=0., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="lower", k=1., x0=0, y0=0)
        self.add_linear(mask=self.mask_pos, type="upper", k=1., x0=0, y0=0)

        k = self.par.u / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=self.par.l, y0=0)

        k = torch.gt(torch.abs(self.par.u), torch.abs(self.par.l)).to(torch.float)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=0, y0=0)

    def build_copy_from_parents(self, par: Layer):
        return EdgeRelu(self.args, self.controller, par)


class EdgeTanh(EdgeActivation):
    def __init__(self, args, controller: BacksubstitutionComputer, par: Layer):
        super(EdgeTanh, self).__init__(args, controller, par)

        def dtanh(x):
            return 1. / torch.cosh(x).pow(2)

        # lower bound for negative
        m = (self.par.l + self.par.u) / 2
        k = dtanh(m)
        self.add_linear(mask=self.mask_neg, type="lower", k=k, x0=m, y0=torch.tanh(m))
        # upper bound for positive
        self.add_linear(mask=self.mask_pos, type="upper", k=k, x0=m, y0=torch.tanh(m))

        # upper bound for negative
        k = (torch.tanh(self.par.u) - torch.tanh(self.par.l)) / (self.par.u - self.par.l + epsilon)
        self.add_linear(mask=self.mask_neg, type="upper", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))
        # lower bound for positive
        self.add_linear(mask=self.mask_pos, type="lower", k=k, x0=self.par.l, y0=torch.tanh(self.par.l))

        # bounds for both
        max_iter = 10

        # lower bound for both
        diff = lambda d: (torch.tanh(self.par.u) - torch.tanh(d)) / (self.par.u - d + epsilon) - dtanh(d)
        d = self.par.l / 2
        _l = self.par.l
        _u = self.to_dev(torch.zeros(self.par.l.shape))
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * mask_p + _l * (1 - mask_p)
            _u = d * (1 - mask_p) + _u * mask_p
            d = (d + _u) / 2 * mask_p + (d + _l) / 2 * (1 - mask_p)
        k = (torch.tanh(d) - torch.tanh(self.par.u)) / (d - self.par.u + epsilon)
        self.add_linear(mask=self.mask_both, type="lower", k=k, x0=d, y0=torch.tanh(d))

        # upper bound for both
        diff = lambda d: (torch.tanh(d) - torch.tanh(self.par.l)) / (d - self.par.l + epsilon) - dtanh(d)
        d = self.par.u / 2
        _l = self.to_dev(torch.zeros(self.par.l.shape))
        _u = self.par.u
        for t in range(max_iter):
            v = diff(d)
            mask_p = torch.gt(v, 0).to(torch.float)
            _l = d * (1 - mask_p) + _l * mask_p
            _u = d * mask_p + _u * (1 - mask_p)
            d = (d + _u) / 2 * (1 - mask_p) + (d + _l) / 2 * mask_p
        k = (torch.tanh(d) - torch.tanh(self.par.l)) / (d - self.par.l + epsilon)
        self.add_linear(mask=self.mask_both, type="upper", k=k, x0=d, y0=torch.tanh(d))

    def build_copy_from_parents(self, par: Layer):
        return EdgeTanh(self.args, self.controller, par)


