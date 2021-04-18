# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.
from typing import List, Union, Tuple, Any

import torch

from Verifiers.BacksubstitutionComputer import BacksubstitutionComputer
from Verifiers.Bounds import Bounds
from Verifiers.EdgeWithGrad import EdgeWithGrad

epsilon = 1e-12

# IMPORTANT: if change here, also change in Edge.py
INPLACE = False

class Layer:
    def __init__(self, args, backsubstitution_computer: BacksubstitutionComputer, length: int, dim, bounds=None, layer_pos=None):
        self.args = args
        self.backsubstitution_computer = backsubstitution_computer
        self.length = length
        self.dim = dim
        self.use_forward = ("baf" in args.method)
        self.parents = []
        self.l = self.u = None
        # for back propagation
        self.lw = self.uw = None
        # bounds of the layer 
        self.final_lw = self.final_uw = None
        self.final_lb = self.final_ub = None
        self.empty_cache = args.empty_cache
        self.layer_pos = layer_pos  # But this append_layer() call will immediately set a new ID if the received ID was None
        self.backsubstitution_computer.append_or_update_layer(self)

        # bounds obtained from the forward framework
        if bounds is not None:
            self.back = False
            self.bounds = bounds

            self.l, self.u = bounds.concretize()
            self.final_lw, self.final_lb = bounds.lw.transpose(-1, -2), bounds.lb
            self.final_uw, self.final_ub = bounds.uw.transpose(-1, -2), bounds.ub

            # incompatible format (batch)
            self.l = self.l[0]
            self.u = self.u[0]
            self.final_lw = self.final_lw[0]
            self.final_lb = self.final_lb[0]
            self.final_uw = self.final_uw[0]
            self.final_ub = self.final_ub[0]
        else:
            self.back = True

    def has_pos(self):
        return self.layer_pos is not None

    def get_pos(self):
        assert self.has_pos(), "Layer has no id! (Self = %s)" % self
        return self.layer_pos

    def set_pos(self, layer_pos):
        assert not self.has_pos(), "Layer already has id '%s', can't change it!" % self.layer_pos
        self.layer_pos = layer_pos

    def to_dev(self, val: torch.Tensor) -> torch.Tensor:
        if self.args.cpu:
            return val
        else:
            return val.cuda()

    def print(self, message):
        print(message)
        print("shape (%d, %d)" % (self.length, self.dim))
        print("mean abs %.5f %.5f" % (torch.mean(torch.abs(self.l)), torch.mean(torch.abs(self.u))))
        print("diff %.5f %.5f %.5f" % (torch.min(self.u - self.l), torch.max(self.u - self.l), torch.mean(self.u - self.l)))
        print("min", torch.min(self.l))
        print("max", torch.max(self.u))
        print()

    def add_edge(self, edge):
        self.parents.append(edge)

    def next(self, edge: 'Edge', length=None, dim=None, layer_gradient_manager=None) -> Union['Layer', Tuple['Layer', Any]]:
        """ Creates a layer, adds the edge and the computes its value (by doing a backwards() call) """
        if length is None:
            length = self.length
        if dim is None:
            dim = self.dim
        layer = Layer(self.args, self.backsubstitution_computer, length, dim)
        layer.add_edge(edge)
        layer.compute(layer_gradient_manager)
        return layer

    def compute(self, layer_gradient_manager=None):
        """ This creates its own lower and upper bounds matrices and then calls the controller's compute()
        method and stores the bounds in the layer."""
        if self.use_forward:
            self.lw = self.to_dev(torch.eye(self.dim)).reshape(1, self.dim, self.dim).repeat(self.length, 1, 1)
        else:
            self.lw = self.to_dev(torch.eye(self.length * self.dim)).reshape(self.length, self.dim, self.length, self.dim)

        self.uw = self.lw.clone()
        self.backsubstitution_computer.compute_backsubstitution(self.length, self.dim, layer_gradient_manager)
        # L, U are the concretized lower and upper bounds
        self.l, self.u = self.backsubstitution_computer.lb, self.backsubstitution_computer.ub

        if not self.args.discard_final_dp:
            # LW, LB are the weights and biases of the lower bounds equations
            self.final_lw, self.final_uw = self.backsubstitution_computer.final_lw, self.backsubstitution_computer.final_uw
            # UW, UB are the weights and biases of the upper bounds equations
            self.final_lb, self.final_ub = self.backsubstitution_computer.final_lb, self.backsubstitution_computer.final_ub

    def backward_buffer(self, lw: torch.Tensor, uw: torch.Tensor):
        """ Adds the lower and upper bound weights to the ones it already has """
        if self.lw is None:
            self.lw, self.uw = lw, uw
        else:
            if INPLACE:
                self.lw += lw
                self.uw += uw
            else:
                self.lw = self.lw + lw
                self.uw = self.uw + uw

    def backward_buffer_at_pos(self, lw: torch.Tensor, uw: torch.Tensor, pos: int, dim: int):
        """ Adds the lower and upper bound weights to the ones at the specific position (we advance matrix size
        by matrix size. If lw has size height 5, then pos = 0 corresponds to index 0 and pos = 1 corresponds to index 5)
        """
        assert False, "This works but isn't supposed to be used right now, since I don't use the new softmax in my current tests"
        assert self.lw is not None, "backward_buffer_at_pos: lw is None"
        assert self.uw is not None, "backward_buffer_at_pos: us is None"

        # TODO: this is an inplace operation, might fail to compute the gradient appropriately
        if dim == -1:
            last_coord = lw.shape[-1]
            self.lw[..., pos*last_coord:(pos + 1)*last_coord] += lw
            self.uw[..., pos*last_coord:(pos + 1)*last_coord] += uw
        elif dim == -2:
            last_coord = lw.shape[-2]
            self.lw[..., pos * last_coord:(pos + 1) * last_coord, :] += lw
            self.uw[..., pos * last_coord:(pos + 1) * last_coord, :] += uw
        else:
            raise Exception("backward_buffer_at_pos only supported for the last 2 dimensions")

    def setup_buffers(self, shape: List[int]):
        """ Create the expression buffer with the given shape """
        assert self.lw is None, "setup_buffers: self.lw is None"
        assert self.uw is None, "setup_buffers: self.uw is None"
        self.lw = self.to_dev(torch.zeros(shape))
        self.uw = self.to_dev(torch.zeros(shape))

    def backward(self, layer_gradient_manager=None):
        """ If backwards, call the backwards() method of all the parent edges
            If forwards, multiplies the bounds of the variables with the upper/lower weights and add the biases,
            stores tha resulting bound and concretizes them to also store the scalar bound value"""
        if self.back:
            for edge in self.parents:
                if layer_gradient_manager is not None:  # Meaning we want to track gradients
                    # The EdgeWithGrad context manager updates the edge's lw,uw,lb,ub (and others) so that
                    # the gradient between them and lambda is injected
                    with EdgeWithGrad(edge, layer_gradient_manager) as (edge_with_grad, needs_grad):
                        # print("self.lw.shape: ", self.lw.shape)
                        if needs_grad:
                            edge_with_grad.backward(self.lw, self.uw)
                        else:
                            with torch.no_grad():
                                edge_with_grad.backward(self.lw, self.uw)
                else:
                    edge.backward(self.lw, self.uw)
        else:
            # 1) self.bounds: needs gradient injection
            # 2) self.lw/uw: doesn't need injected gradients, because it's the result of things with injected gradient (e.g.
            # the gradient injection has already been done)
            # 3) self.backsubstitution_computer.lb/ub: doesn't need gradient injection either, because the only place
            # where it's modified is in Edges, and those already have gradient injection.
            if layer_gradient_manager is not None:
                input_bounds = layer_gradient_manager.create_bounds_obj_with_grad(self.bounds, self, self.get_pos())
                print("Creating bounds with added gradient in Layer with self.back=False")
            else:
                input_bounds = self.bounds

            # Here we still have the gradient, so it gets turned off later on
            bounds_l = input_bounds.matmul(self.lw).add(self.backsubstitution_computer.lb.unsqueeze(0))
            bounds_u = input_bounds.matmul(self.uw).add(self.backsubstitution_computer.ub.unsqueeze(0))
            bounds = Bounds(
                bounds_l.args, bounds_l.p, bounds_l.eps,
                lw=bounds_l.lw, lb=bounds_l.lb,
                uw=bounds_u.uw, ub=bounds_u.ub
            )
            self.backsubstitution_computer.final_lw = bounds.lw[0].transpose(1, 2)
            self.backsubstitution_computer.final_uw = bounds.uw[0].transpose(1, 2)
            self.backsubstitution_computer.final_lb = bounds.lb[0]
            self.backsubstitution_computer.final_ub = bounds.ub[0]
            self.backsubstitution_computer.lb, self.backsubstitution_computer.ub = bounds.concretize()
            self.backsubstitution_computer.lb = self.backsubstitution_computer.lb[0]
            self.backsubstitution_computer.ub = self.backsubstitution_computer.ub[0]
            
        if self.empty_cache:
            torch.cuda.empty_cache()
        self.lw, self.uw = None, None
