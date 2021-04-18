# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# (c) Anonymous Authours
# Licenced under the BSD 2-Clause License.

import torch

from Verifiers.utils import dual_norm


class BacksubstitutionComputer:
    """ This BacksubstitutionComputer is the one that receives all the layers one by one and
    then does the whole backsubstitution for a layer. To do so, it goes through each layer
    one by one, in reverse chronological order (later layers first). Because each layer can have multiple
    children layers (such as a Key in the a transformer) we need to do this in reverse order, so that
    the key accumulates all the terms linked to it and only when it receives them all it can start
    backpropagating. After a full backsubstituion is done, everything is reset (the intermediate expressions are
    reset and set to None) and only the final result is kept. """

    def __init__(self, args, eps: float):
        self.args = args
        self.layers = []
        self.p = args.p

        self.q = dual_norm(self.p)
        # self.q = 1. / (1. - 1. / args.p) if args.p != 1 else float("inf")  # dual norm
        self.eps = eps
        self.perturbed_words = args.perturbed_words
        self.total_deleted_memory = 0

    def delete_tensor(self, x: torch.Tensor):
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # memory_before = torch.cuda.memory_allocated()
        # n_elem = x.nelement()
        del x
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # time.sleep(5)
        # memory_after = torch.cuda.memory_allocated()
        # assert memory_after < memory_before, \
        #    "Deleting the tensor (%d elements) didn't reduce the memory (before: %d, after: %d)" % (n_elem, memory_before, memory_after)
        # self.total_deleted_memory += (memory_before - memory_after)

    def append_or_update_layer(self, layer: 'Layer'):
        """ Add a newly created layer to the end of the layers managed by this controller """
        if layer.has_pos():
            self.layers[layer.get_pos()] = layer
        else:
            self.layers.append(layer)
            layer.set_pos(self.get_num_layers() - 1)

    def get_num_layers(self):
        return len(self.layers)

    def compute_backsubstitution(self, length: int, dim: int, layer_gradient_manager=None):
        """ It creates the zero matrices to store the lower and upper biases and the weights
         and then goes layer by layer (backwards, from end to start) and calls the backwards() method
         so that in the end these contain the final expression."""
        if self.args.cpu:
            self.lb = torch.zeros(length, dim)
        else:
            self.lb = torch.zeros(length, dim).cuda()
        self.ub = self.lb.clone()
        self.final_lw = self.final_uw = None
        self.final_lb = self.final_ub = None

        original_layers = self.layers[:]  # We keep a copy, because layers might get modified during the backsubstitution with jacobian
        for layer in self.layers[::-1]:
            if layer.lw is not None:
                layer.backward(layer_gradient_manager)
                # print("Controller has grad: ", self.lb.requires_grad)
        self.layers = original_layers  # We restore the original layers

    def concretize_l(self, lw: torch.Tensor):
        """ Computes the concretized scalar lower bound value given eps and the lower bound weights """
        return -self.eps * torch.norm(lw, p=self.q, dim=-1)

    def concretize_u(self, uw: torch.Tensor):
        """ Computes the concretized scalar upper bound value given eps and the upper bound weights """
        return self.eps * torch.norm(uw, p=self.q, dim=-1)

    def concretize(self, lw: torch.Tensor, uw: torch.Tensor):
        """ Computes the concretized scalar upper and lower bound values"""
        if self.args.attack_type == "synonym":
            # lw:         (input/output len, output dim, len * input dim)
            # lw = lw.transpose(0, 1)  # (output dim, input/output len, len * input dim)
            # uw = uw.transpose(0, 1)  # (output dim, input/output len, len * input dim)

            dim = lw.size(2) // self.perturbed_words
            concretized_l = torch.zeros(lw.size(0), lw.size(1), device=lw.device)
            concretized_u = torch.zeros(uw.size(0), uw.size(1), device=uw.device)
            for i in range(self.perturbed_words):
                concretized_l -= torch.norm(lw[:, :, i*dim:(i+1)*dim] * self.args.embedding_radii[i], p=self.q, dim=-1) #.t()
                concretized_u += torch.norm(uw[:, :, i*dim:(i+1)*dim] * self.args.embedding_radii[i], p=self.q, dim=-1) #.t()

            return concretized_l, concretized_u
        elif self.perturbed_words == 2:
            assert (len(lw.shape) == 3), "Lw does not have 3 dimensions! It has %d dimensions" % len(lw.shape)
            half = lw.shape[-1] // 2
            concretized_l = self.concretize_l(lw[:, :, :half]) + self.concretize_l(lw[:, :, half:])
            concretized_u = self.concretize_u(uw[:, :, :half]) + self.concretize_u(uw[:, :, half:])
            return concretized_l, concretized_u
        elif self.perturbed_words == 1:
            return self.concretize_l(lw), self.concretize_u(uw)
        else:
            raise NotImplementedError

    def cleanup(self):
        for layer in self.layers:
            try:
                del layer.lw
                del layer.uw
                del layer.final_lw
                del layer.final_uw
                del layer.final_lb
                del layer.final_ub
            except AttributeError:
                pass
