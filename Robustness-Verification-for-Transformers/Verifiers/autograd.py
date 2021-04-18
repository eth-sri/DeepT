from typing import Iterable, List, Tuple, Dict, Union, Sequence
from collections import defaultdict

import torch
from torch.autograd.functional import jacobian, _as_tuple, _grad_preprocess, _check_requires_grad, _autograd_grad, _grad_postprocess, \
    _tuple_postprocess

from Verifiers import Bounds
from Verifiers.Layer import Layer


def jacobian_and_output(func, inputs, create_graph=False, strict=False):
    r"""Variant of the Pytorch jacobian function that also returns the outputs"""

    is_inputs_tuple, inputs = _as_tuple(inputs, "inputs", "jacobian")
    inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)

    outputs = func(*inputs)
    is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
    _check_requires_grad(outputs, "outputs", strict=strict)

    jacobian = tuple()
    for i, out in enumerate(outputs):

        jac_i = tuple([] for _ in range(len(inputs)))
        for j in range(out.nelement()):
            # print("Start: Computing gradient relative to %d-th element of the output" % j)
            # start = time()
            vj = _autograd_grad((out.reshape(-1)[j],), inputs, retain_graph=True, create_graph=create_graph)
            # end = time()
            # print("  End: Computing gradient relative to the %d-th element of the output (time: %f)" % (j, end - start))

            for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                if vj_el is not None:
                    if strict and create_graph and not vj_el.requires_grad:
                        raise RuntimeError("The jacobian of the user-provided function is independent of "
                                           "input {}. This is not allowed in strict mode when create_graph=True.".format(i))
                    jac_i_el.append(vj_el)
                else:
                    if strict:
                        raise RuntimeError("Output {} of the user-provided function is independent of "
                                           "input {}. This is not allowed in strict mode.".format(i, el_idx))
                    jac_i_el.append(torch.zeros_like(inp_el))

        jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()
                                                             + inputs[el_idx].size()) for (el_idx, jac_i_el) in enumerate(jac_i)),)

    jacobian = _grad_postprocess(jacobian, create_graph)

    return _tuple_postprocess(jacobian, (is_outputs_tuple, is_inputs_tuple)), outputs


def to_cpu(x, delete_from_gpu=False):
    if torch.is_tensor(x):
        x_cpu = x.cpu()
        if delete_from_gpu and x.is_cuda:
            del x
        return x_cpu
    elif type(x) == tuple:
        return tuple([to_cpu(v) for v in x])
    else:
        raise Exception("Not supported")


def to_cuda(x):
    if torch.is_tensor(x):
        return x.cuda()
    elif type(x) == tuple:
        return tuple([to_cuda(v) for v in x])
    else:
        raise Exception("Not supported")


def delete_all_from_memory(x):
    if torch.is_tensor(x):
        del x
    elif type(x) == tuple:
        for v in x:
            delete_all_from_memory(v)
    else:
        raise Exception("Not supported")


def multiply_gradients(jacobians, grad_outputs):
    # f has N inputs, M outputs and the final thing has M gradient vectors (dfinal / doutput)
    # Input:
    #   jacobians: doutputs / dinputs  (M tuples each with N jacobians)
    #   grad_outputs: dfinal_value / doutputs (M vectors)
    # Returns:
    #   Composition of the jacobians doutputs / dinputs and the gradient to the end dfinal_values / doutputs
    #   eg. dfinal_values / dinputs of size (N gradient vector)
    #
    # Algorithm
    #   dfinal_value / dinput_j = sum((dfinal_value / doutput_i) * (doutput_i / dinput_j)) for all outputs_i
    #   This looks a lot like a matrix multiplication
    M = len(jacobians)
    N = len(jacobians[0])
    assert len(grad_outputs) == M, "Number of gradient vectors (%d) != number of function outputs (%d)" % (len(grad_outputs), M)

    grads = []
    for j in range(N):
        grad_j = None
        for i in range(M):
            # Jacobian has size O x I, grad_output has size O, and we want an output of size I
            # Therefore, we have I = (I x O) x O, so we need to transpose the jacobian dimensions
            jacobian = jacobians[i][j]
            num_o = len(grad_outputs[i].shape)
            num_d = len(jacobian.shape)
            jacobian = jacobian.permute(*range(num_o, num_d), *range(num_o))
            if num_o == 1:
                value = (jacobian * grad_outputs[i]).sum(-1)
            elif num_o == 2:
                value = (jacobian * grad_outputs[i]).sum((-1, -2))
            else:
                raise Exception("Not supported! Output values have 3 dims")

            if grad_j is None:
                grad_j = value
            else:
                grad_j += value

        grads.append(grad_j)

    return tuple(grads)


# noinspection PyMethodOverriding
class GradientProvider(torch.autograd.Function):
    @staticmethod
    def forward(context, result: torch.Tensor, jacobians: torch.Tensor, *args):
        context.jacobians = jacobians
        return result

    @staticmethod
    def backward(context, *grad_outputs):
        jacobians = context.jacobians
        return tuple([None, None, *multiply_gradients(jacobians, grad_outputs)])


# noinspection PyMethodOverriding
class GradientProviderGPU(torch.autograd.Function):
    @staticmethod
    def forward(context, output: torch.Tensor, jacobians: torch.Tensor, *inputs):
        # If we use the GPU, we store the jacobian in main memory while they're not in use
        # We'll later move them back to the GPU in the backwards call
        # This is needed because we can't fit all Jacobians in memory
        # context.jacobians = to_cpu(jacobians, delete_from_gpu=True)
        assert jacobians[0][0].is_cuda, "Jacobians are not CUDA!"
        context.jacobians = to_cuda(jacobians)
        return output

    @staticmethod
    def backward(context, *grad_outputs):
        # start = time()
        # jacobians = to_cuda(context.jacobians)
        jacobians = context.jacobians
        # end = time()
        # size = sum([sum([x.nelement() for x in y]) for y in jacobians])
        # print("Moved jacobians to GPU (time: %f) - size bytes: %d" % (end - start, size * 4))

        # start = time()
        grad_inputs = tuple([None, None, *multiply_gradients(jacobians, grad_outputs)])
        # end = time()
        # print("Multiplying the jacobian by the vector-shaped grad_output (time: %f)" % (end - start))

        # delete_all_from_cuda(jacobians)
        return grad_inputs


def get_val_and_jacobians(f, *args, post_f=lambda x: x):
    old_style = False
    if old_style:
        # No need to use the gradient just to get the value
        with torch.no_grad():
            value = f(*[arg.detach() for arg in args])

        # Need to ensure the gradient is enabled when computing the jacobian
        with torch.enable_grad():
            jacobians: Tuple[Tuple[torch.Tensor, ...], ...] = jacobian((lambda *inp: post_f(f(*inp))), args, strict=True)
    else:
        with torch.enable_grad():
            jacobians, value = jacobian_and_output((lambda *inp: post_f(f(*inp))), args, strict=True) # Put back to False)  # TODO: see if I can put true again

    return value, jacobians


def replace_grad_fn(inputs: Iterable[torch.Tensor], output: Iterable[torch.Tensor],
                    jacobians: Iterable[Iterable[torch.Tensor]], with_gpu=False) -> Tuple[torch.Tensor, ...]:
    if with_gpu:
        output = GradientProviderGPU.apply(output, jacobians, *inputs)
    else:
        output = GradientProvider.apply(output, jacobians, *inputs)

    return output


def compute_but_replace_grad_fn(f, *args: torch.Tensor, with_gpu=True, post_f=lambda x: x) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    assert all([type(arg) == torch.Tensor for arg in args]), "Not all arguments are torch Tensors!"

    value, jacobians = get_val_and_jacobians(f, *args, post_f=post_f)
    value = replace_grad_fn(inputs=args, output=value, jacobians=jacobians, with_gpu=with_gpu)

    if type(value) == tuple and len(value) == 1:
        return value[0]
    else:
        return value


# I want to have a edge.params <-> lambdas relationship with gradients, so that the gradient computation works correctly.
# We can create this relationship using two facts:
#   (1) edge.params = f(parent_layer)
#   (2) we stored the gradient parent_layer <-> lambdas
# Combining these two, we can get the gradient for edge.params <-> lambda correctly. This requires doing 2 things (in order):
#   (1) doing a parent_layer = magic(lambdas) that does nothing except add those gradients
#   (2) re-compute the edge params from that parent_layer, so that there's the link between edge.params <-> parent_layer <-> lambdas
#
# Step (2) is required. I have to explictly do the recomputation of the edge params in terms of the parent layer,
# because otherwise there will be no link between the edge and the lambdas, and so all the gradients will be 0,
# and this is not what we want.
#
# To make this all of this machinery work, I created the EdgeWithGrad context manager. EdgeWithGrad takes the edge,
# does step (1) to the parents if it hasn't been done already (thanks to another edge processed previously, in the case
# where a Layer is used several times) and then (2) recompute the Edge's parameters from the its parent layers'
# newly fake-computed concrete bounds.
#
# I also need a component that stopres the Gradients between lambdas and layers, and which also track for each layer, on which
# lambdas does it depend. This component is the LambdaBoundsGradientTracker. To make it work, it required adding the notion of an ID
# to a Layer. I added a set_id() and get_id() function to the layer, and the id is set for the first time when we add it to the
# BacksubstitutionComputer, which ensure it has an unique ID. When we create a copy of the layer with added gradients, we re-use the same
# id for that layer, since they are the same layer, just with added gradient for the newly created one.
#
# I think I have to be careful about something! When I create a Layer, it adds itself to the backsubstitution computer :(
# So the cloned layer will actually be appended in the end instead of replacing the layer. Hmmm. I need to think about this.


class LambdaBoundsGradientTracker:
    def __init__(self, use_cuda=False, handle_one_lambda_at_the_time=False):
        self.lambda_bounds_jacobians_dict: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.lambdas_relevant_for_layer: Dict[int, List[int]] = defaultdict(list)
        self.lambdas_list: List[torch.Tensor] = []
        self.keys_in_order = []
        self.use_cuda = use_cuda

        self.iteration_num = -1
        self.on_initial_iteration = False
        self.current_relevant_keys = []
        self.handle_one_lambda_at_the_time = handle_one_lambda_at_the_time

    def add_lambdas(self, lambdas: torch.Tensor):
        self.lambdas_list.append(lambdas)

    def add_bounds_lambda_grad(self, lambda_index: int, layer_pos: int, lambda_layer_bounds_jacobians: Tuple[torch.Tensor, torch.Tensor]):
        with torch.no_grad():
            assert not torch.isnan(lambda_layer_bounds_jacobians[0]).any(), "Jacobian[0] has NaNs"
            assert not torch.isnan(lambda_layer_bounds_jacobians[1]).any(), "Jacobian[1] has NaNs"

            self.lambdas_relevant_for_layer[layer_pos].append(lambda_index)
            self.lambda_bounds_jacobians_dict[(layer_pos, lambda_index)] = to_cpu(lambda_layer_bounds_jacobians, delete_from_gpu=True)
            self.keys_in_order.append((layer_pos, lambda_index))

    def get_lambda_jacobians(self, layer_pos: int, lambda_index: int) -> Tuple[torch.Tensor, ...]:
        return self.lambda_bounds_jacobians_dict[(layer_pos, lambda_index)]

    def get_jacobians_for_lambdas(self, layer_pos, lambdas_indices):
        # Example output when there's 3 lambdas and 2 outputs (l, u): tuple(
        #   (grad1_l, grad2_l, grad3_l),
        #   (grad1_u, grad2_u, grad3_u)
        # )
        val = self.get_lambda_jacobians(layer_pos, lambdas_indices[0])
        num_ouputs = len(val)

        gradients = [[] for _ in range(num_ouputs)]
        for lambda_index in lambdas_indices:
            jacobians = self.get_lambda_jacobians(layer_pos, lambda_index)
            for i, jacobian_tensor in enumerate(jacobians):
                gradients[i].append(jacobian_tensor)

        return tuple([tuple(x) for x in gradients])

    def delete_all(self):
        for key in self.keys_in_order:
            delete_all_from_memory(self.lambda_bounds_jacobians_dict[key])

        for i in range(len(self.lambdas_list)):
            del self.lambdas_list[i]

        del self.lambdas_list
        del self.lambda_bounds_jacobians_dict

    ########################
    # Lifecycle management #
    ########################

    def start_new_iteration(self):
        # len(self.keys_in_order) - 1 because a new key might have been added in between
        #assert self.iteration_num == -1 or self.iteration_num >= len(self.keys_in_order) - 1, \
        #    "Starting new iteration before using all gradients (iteration num = %d)" % self.iteration_num
        self.iteration_num = 0
        self.on_initial_iteration = True

        # The ordering is important, don't re-arrange it
        if self.use_cuda:
            self.move_current_jacobians_to_device('cpu')
        self.current_relevant_keys = []

    def mark_end_iteration(self):
        self.on_initial_iteration = False
        self.iteration_num += 1

    def have_injected_all_gradients(self) -> bool:
        if self.handle_one_lambda_at_the_time:
            return not self.on_initial_iteration and self.iteration_num >= len(self.keys_in_order)
        else:
            return not self.on_initial_iteration

    def move_current_jacobians_to_device(self, device: str):
        for key in self.current_relevant_keys:
            lower, upper = self.lambda_bounds_jacobians_dict[key]
            self.lambda_bounds_jacobians_dict[key] = (lower.to(device), upper.to(device))
            del lower
            del upper

    def prepare_next_gradients_to_be_injected(self):
        if self.use_cuda:
            self.move_current_jacobians_to_device(device='cpu')  # Move previous jacobians back to CPU

        if self.handle_one_lambda_at_the_time:
            # When adding the first lambdas, there's no keys but we still want to get jacobians
            # We deal with that exception here
            if self.on_initial_iteration and len(self.keys_in_order) == 0:
                self.current_relevant_keys = []
            else:
                self.current_relevant_keys = [self.keys_in_order[self.iteration_num]]
        else:
            self.current_relevant_keys = self.keys_in_order

        if self.use_cuda:
            self.move_current_jacobians_to_device(device='cuda')  # Move new jacobians to GPU for speed

    def should_inject_grad_on_direct_lambda_layers(self) -> bool:
        # We allow use of the lambdas in the softmax in the 1st iteration, to get those grads only once
        return self.on_initial_iteration

    def need_to_inject_gradient(self, layer_pos: int, lambda_index: int):
        return (layer_pos, lambda_index) in self.current_relevant_keys

    def need_to_inject_gradient_for_layer(self, layer: Layer) -> bool:
        return len(self.get_currently_applicable_lambdas_indices(layer.get_pos())) > 0

    def get_currently_applicable_lambdas_indices(self, layer_pos: int) -> List[int]:
        return [lambda_index
                for lambda_index in self.lambdas_relevant_for_layer[layer_pos]
                if self.need_to_inject_gradient(layer_pos, lambda_index)]


class LayerGradientManager:
    def __init__(self, lambdas_list: Sequence[torch.Tensor], gradient_tracker: LambdaBoundsGradientTracker):
        self.lambdas_list = lambdas_list
        self.gradient_tracker = gradient_tracker
        self.original_values = {}
        self.replaced_layers_cache = {}

    def get_layer_with_grad(self, layer: Layer) -> Layer:
        assert self.gradient_tracker.need_to_inject_gradient_for_layer(layer), \
            "Requesting adding grad to layer but layer doesn't need grad now (layer pos: %d, parents: %s)" % (layer.get_pos(), layer.parents)

        # Check in the cache first
        layer_pos = layer.get_pos()
        if layer_pos in self.replaced_layers_cache:
            return self.replaced_layers_cache[layer_pos]

        assert layer_pos not in self.original_values, \
            "Logical problem: added grad but didn't keep gradless value for layer with ID %s" % layer_pos
        self.original_values[layer_pos] = (layer.l, layer.u)
        self.replaced_layers_cache[layer_pos] = self.add_grad_to_layer(layer, layer_pos)
        return self.replaced_layers_cache[layer_pos]

    def add_grad_to_layer(self, layer: Layer, layer_pos: int) -> Layer:
        # We get the indices from the tracker, but use the lambdas received during the construction
        # to ensure the jacobian computation is actually correct (we need to use the lambdas the jacobian received as params)
        currently_relevant_lambdas_indices = self.gradient_tracker.get_currently_applicable_lambdas_indices(layer_pos)

        assert len(currently_relevant_lambdas_indices) > 0, \
            "Requesting adding grad to layer but layer doesn't need grad now (layer pos: %d, parents: %s)" % (layer.get_pos(), layer.parents)

        currently_relevant_lambdas = [self.lambdas_list[i] for i in currently_relevant_lambdas_indices]
        currently_relevant_jacobians = self.gradient_tracker.get_jacobians_for_lambdas(layer_pos, currently_relevant_lambdas_indices)

        bounds_values = layer.l, layer.u
        bounds_tensors = replace_grad_fn(
            inputs=currently_relevant_lambdas,
            output=bounds_values,
            jacobians=currently_relevant_jacobians,
            with_gpu=not layer.args.cpu
        )
        layer_below_with_grad = Layer(layer.args, layer.backsubstitution_computer, layer.length, layer.dim, layer_pos=layer.get_pos())
        layer_below_with_grad.l = bounds_tensors[0]
        layer_below_with_grad.u = bounds_tensors[1]
        return layer

    def create_bounds_obj_with_grad(self, bounds: Bounds, layer: Layer, layer_pos: int) -> Bounds:
        # We get the indices from the tracker, but use the lambdas received during the construction
        # to ensure the jacobian computation is actually correct (we need to use the lambdas the jacobian received as params)
        currently_relevant_lambdas_indices = self.gradient_tracker.get_currently_applicable_lambdas_indices(layer_pos)

        # There are no lambdas before (or in) the first forward block, and so for those bounds we don't need to inject any jacobians
        if len(currently_relevant_lambdas_indices) == 0:
            return bounds

        currently_relevant_lambdas = [self.lambdas_list[i] for i in currently_relevant_lambdas_indices]
        currently_relevant_jacobians = self.gradient_tracker.get_jacobians_for_lambdas(layer_pos, currently_relevant_lambdas_indices)

        bounds_values = bounds.lw, bounds.lb, bounds.uw, bounds.ub
        lw_with_grad, lb_with_grad, uw_with_grad, ub_with_grad = replace_grad_fn(
            inputs=currently_relevant_lambdas,
            output=bounds_values,
            jacobians=currently_relevant_jacobians,
            with_gpu=not bounds.args.cpu
        )

        return Bounds(
            bounds.args, bounds.p, bounds.eps,
            lw=lw_with_grad,
            lb=lb_with_grad,
            uw=uw_with_grad,
            ub=ub_with_grad
        )

    def restore_original_bounds(self):
        assert self.original_values.keys() == self.replaced_layers_cache.keys(), \
            "Difference between the layers with preserved value (keys: %s) and the layer where we replaced the values (%s)" % \
            (self.original_values.keys(), self.replaced_layers_cache.keys())

        for layer_pos, (lower_bound, upper_bound) in self.original_values.items():
            layer = self.replaced_layers_cache[layer_pos]
            # Delete the bounds gotten from the grad-injected operation
            del layer.l
            del layer.u
            # Restore the original bounds
            layer.l = lower_bound
            layer.u = upper_bound

    def get_lambdas_of_edge(self, edge):
        return self.lambdas_list[edge.get_lambdas_index()]

    def need_to_inject_gradient_for_any_layer(self, layers):
        return any((self.need_to_inject_gradient_for_layer(layer) for layer in layers))

    def need_to_inject_gradient_for_layer(self, layer: Layer) -> bool:
        return self.gradient_tracker.need_to_inject_gradient_for_layer(layer)

    def should_inject_grad_on_direct_lambda_layers(self) -> bool:
        return self.gradient_tracker.should_inject_grad_on_direct_lambda_layers()


# This is the desired code:
#
# gradient_tracker = LambdaBoundsGradientTracker()
# layer_gradient_manager = LayerGradientManager(lambdas_list, gradient_tracker)
# for layer in layers:
#     for edge in layer.edges:
#         # The EdgeWithGrad context maanger updates the edge's lw,uw,lb,ub (and others) so that they correctly rely
#         # on the lambdas, so that the grads for the lambdas are correctly transmitted
#         with EdgeWithGrad(edge, layer_gradient_manager) as edge_with_grad:
#             edge_with_grad.backward(layer.lw, layer.uw)
