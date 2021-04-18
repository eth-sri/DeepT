

class EdgeWithGrad:
    def __init__(self, edge, layer_gradient_manager):
        self.edge = edge
        self.edge_built_on_grad = None
        self.layer_gradient_manager = layer_gradient_manager
        self.gradient_is_added = None

    def __enter__(self):
        old_parents = self.edge.get_parents()

        self.gradient_is_added = (self.edge.has_lambdas() and
                                  self.layer_gradient_manager.should_inject_grad_on_direct_lambda_layers()) or \
                                 (self.edge.params_depend_on_input_bounds() and
                                  self.layer_gradient_manager.need_to_inject_gradient_for_any_layer(old_parents))

        # TODO: temporary change
        self.gradient_is_added = True

        edge_class = type(self.edge).__name__
        if not self.gradient_is_added:
            reason = ""
            if self.edge.has_lambdas():
                reason += "Has lambdas; but don't treat them in the current iteration;"
            else:
                reason += "No lambdas;"

            if not self.edge.params_depend_on_input_bounds():
                reason += "edge params don't depend on input layers"
            else:
                reason += "edge params depend on input layers but parent layers don't have grad"

            print("\tBacksubstitution: edge %s with no added gradient - reason '%s'" % (edge_class, reason))
            return self.edge, self.gradient_is_added

        if self.edge.has_lambdas():
            print("\tBacksubstitution: edge %s with added gradient - reason 'Layer has lambdas and currently treating them'" % edge_class)
        else:
            print("\tBacksubstitution: edge %s with added gradient - reason 'No lambdas but depends on input layers which have grad'" % edge_class)

        # Step (1) Inject gradient to the parent layers (if desired and appropriate in the current iteration)
        new_parents = []
        for parent in old_parents:
            if self.layer_gradient_manager.need_to_inject_gradient_for_layer(parent):
                new_parents.append(self.layer_gradient_manager.get_layer_with_grad(parent))
            else:
                new_parents.append(parent)

        # Step (2) Recompute the edge's parameters using the parents layer with gradients
        if self.edge.has_lambdas() and self.layer_gradient_manager.should_inject_grad_on_direct_lambda_layers():
            print("Creating Edge with lamdbdas from lambdas given by the GradientManager")
            lambdas_created_at_edge = self.layer_gradient_manager.get_lambdas_of_edge(self.edge)
            self.edge_built_on_grad = self.edge.build_copy_from_parents(*new_parents, lambdas_created_at_edge)
        elif self.edge.has_lambdas():  # New parents, but don't inject gradients for direct lambdas
            # By using the existing lambdas instead of the lambdas in the gradient manager, we avoid adding the jacobian
            # relative to those lambdas. It's a bit weird but the jacobians counts if we use the lambdas we received
            # as parameters in the jacobian() call that started the whole process
            self.edge_built_on_grad = self.edge.build_copy_from_parents(*new_parents, self.edge.lambdas)
        else:  # New parents, no direct lambdas
            self.edge_built_on_grad = self.edge.build_copy_from_parents(*new_parents)

        return self.edge_built_on_grad, self.gradient_is_added

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.gradient_is_added:
            assert self.edge_built_on_grad is not None, "Gradient_is_added = True but no edge was created"
            del self.edge_built_on_grad
            self.edge_built_on_grad = None
