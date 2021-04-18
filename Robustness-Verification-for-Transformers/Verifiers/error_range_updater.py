import multiprocessing
from typing import List, Tuple, Any

import torch
# from gurobipy import *


#############################################################
#                    CODE FROM ERAN                         #
#############################################################


class GlobalCache:
    model = None
    index_first_obj_var: int = None
    lower_bounds: List[float] = None
    upper_bounds: List[float] = None


def solver_call(objective_var_index: int) -> Tuple[float, float, bool, float]:
    model = GlobalCache.model.copy()
    runtime = 0

    obj = LinExpr()
    obj += model.getVars()[GlobalCache.index_first_obj_var + objective_var_index]  # TODO: Why create a LinExpr to get a single variable?
    # print (f"{ind} {model.getVars()[Cache.output_counter+ind].VarName}")

    model.setObjective(obj, GRB.MINIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    sol_l = GlobalCache.lower_bounds[objective_var_index] if model.SolCount == 0 else model.objbound
    # print (f"{ind} {model.status} lb ({Cache.lbi[ind]}, {soll}) {model.RunTime}s")
    sys.stdout.flush()

    model.setObjective(obj, GRB.MAXIMIZE)
    model.reset()
    model.optimize()
    runtime += model.RunTime
    sol_u = GlobalCache.upper_bounds[objective_var_index] if model.SolCount == 0 else model.objbound
    # print (f"{ind} {model.status} ub ({Cache.ubi[ind]}, {solu}) {model.RunTime}s")
    sys.stdout.flush()

    refined_the_original_bounds = (sol_l > GlobalCache.lower_bounds[objective_var_index]) or \
                                  (sol_u < GlobalCache.upper_bounds[objective_var_index])

    # Why max? If we had that x >= -1 and the constraint implies that x >= -2, then we stay at x >= -1
    sol_l = max(sol_l, GlobalCache.lower_bounds[objective_var_index])
    # Why min? If we had that x <= 1 and the constraint implies that x <= 2, then we stay at x <= 1
    sol_u = min(sol_u, GlobalCache.upper_bounds[objective_var_index])

    return sol_l, sol_u, refined_the_original_bounds, runtime


#############################################################
#                       MY CODE                             #
#############################################################


IntList = List[int]
FloatList = List[float]


def get_updated_error_ranges_using_LP(linear_expressions_less_or_equal_to_zero: torch.Tensor,
                                      original_lower_bounds: torch.Tensor,
                                      original_upper_bounds: torch.Tensor,
                                      timeout: int,
                                      num_processes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes new error ranges given the set of linear equations

    Parameters
    ----------
    linear_expressions_less_or_equal_to_zero: torch of the shape [num_equations, 1 + num_error_terms]. The bias must be the first value in the
                                              row and the coefficients the remaining elements of the row after it.
    original_lower_bounds
    original_upper_bounds
    timeout: how long Gurobi can run
    num_processes: number of processes processing in parallel

    Returns
    -------
    The new lower and upper bounds of the error terms, which will be equal or tighter than before
    """

    original_lower_bounds_list = original_lower_bounds.tolist()
    original_upper_bounds_list = original_upper_bounds.tolist()

    model, error_term_var_list = create_model(linear_expressions_less_or_equal_to_zero, original_lower_bounds, original_upper_bounds)

    num_error_terms = len(original_lower_bounds_list)
    result_l = [-1.0] * num_error_terms
    result_u = [1.0] * num_error_terms
    refined_vars_indices = []
    var_was_processed = [False] * num_error_terms

    model.setParam(GRB.Param.TimeLimit, timeout)
    model.setParam(GRB.Param.Threads, 2)

    model.update()
    model.reset()

    # Static attributes!
    GlobalCache.model = model
    GlobalCache.lower_bounds = original_lower_bounds_list
    GlobalCache.upper_bounds = original_upper_bounds_list

    # Which vars should we optimize
    GlobalCache.index_first_obj_var = 0  # start at 0

    widths = (original_upper_bounds - original_lower_bounds).tolist()
    candidate_var_indices = list(range(len(error_term_var_list)))  # Find the values of the lower and upper bounds vars
    candidate_var_indices = sorted(candidate_var_indices, key=lambda k: widths[k])  # Sort by width
    num_candidates = len(candidate_var_indices) # Process all candidates!
    var_indices = candidate_var_indices[:num_candidates]  # In practice, this line does nothing

    ### Process all vars once

    ## Set those variables as refined
    for v in var_indices:
        var_was_processed[v] = True

    ## Solve the model in different processes, in parallel, for different variables
    with multiprocessing.Pool(num_processes) as pool:
        solver_result = pool.map(solver_call, var_indices)

    ## Copy solver results into the results lists
    solve_time = 0
    for (l, u, add_to_indices, runtime), ind in zip(solver_result, var_indices):
        result_l[ind] = l
        result_u[ind] = u
        if add_to_indices:
            refined_vars_indices.append(ind)
        solve_time += runtime

    ### Among the remaining unprocess variables, process the top 50 vars
    # In our case, the code below will do NOTHING, since we process all variables in the first call

    ## Reset the model and update the time limit
    avg_solve_time = (solve_time + 1) / (2 * num_candidates + 1)
    model.setParam('TimeLimit', avg_solve_time / 2)
    model.update()
    model.reset()
    print("Average solve time: %f s" % avg_solve_time)

    ## Pick variables to be solved
    var_indices = candidate_var_indices[num_candidates:]
    if len(var_indices) >= 50:
        var_indices = var_indices[:50]

    assert len(var_indices) == 0, "Unexpected results"

    ## Set those variables as refined
    for v in var_indices:
        var_was_processed[v] = True

    ## Solve the model in different processes, in parallel, for different variables
    with multiprocessing.Pool(num_processes) as pool:
        solver_result = pool.map(solver_call, var_indices)

    ## Copy solver results into the results lists
    solve_time = 0
    for (l, u, add_to_indices, runtime), ind in zip(solver_result, var_indices):
        result_l[ind] = l
        result_u[ind] = u
        if add_to_indices:
            refined_vars_indices.append(ind)
        solve_time += runtime

    ## Copy bounds of the variables that were not processed this round
    for i, is_refined in enumerate(var_was_processed):
        if not is_refined:
            result_l[i] = original_lower_bounds_list[i]
            result_u[i] = original_upper_bounds_list[i]

    # avg_solvetime = solve_time / (2 * len(var_indices)) if len(var_indices) else 0.0

    ## Unsoundness checks
    # for i in range(abs_layer_count):
    #     for j in range(len(lower_bounds_per_layer[i])):
    #         if lower_bounds_per_layer[i][j] > upper_bounds_per_layer[i][j]:
    #             print("fp unsoundness detected ", lower_bounds_per_layer[i][j], upper_bounds_per_layer[i][j], i, j)

    ## If the solver lead to unsound bounds for some vars, revert them to the original bounds and print it out
    for j in range(len(result_l)):
        if result_l[j] > result_u[j]:
            print(f"unsound variable: index {j}")
            result_l[j], result_u[j] = original_lower_bounds_list[j], original_upper_bounds_list[j]

    return torch.tensor(result_l), torch.tensor(result_u)  #, sorted(refined_vars_indices)


def create_model(linear_expressions_less_or_equal_to_zero: torch.Tensor,
                 current_lower_bounds: torch.Tensor, current_upper_bounds: torch.Tensor) -> Tuple[Any, List]:
    """

    Parameters
    ----------
    linear_expressions_less_or_equal_to_zero: torch of the shape [num_equations, 1 + num_error_terms]. The bias must be the first value in the
                                       row and the coefficients the remaining elements of the row after it.
    current_lower_bounds
    current_upper_bounds

    Returns
    -------
    Returns the Gurobi Model as well as the list of variables that were created
    """
    # the linear program for expr1 = 2 + 3e1 + 2e2 and expr2 = 3 - e1 + 3e2
    # where e1 is in range [-1, 1] and e2 is in range [0.5, 1] would be

    # CONSTRAINTS:
    # -1 <= e1 <= 1
    # 0.5 <= e2 <= 1
    # (2 - 3e1 + 2e2) - (3 - e1 + 3 e2) < 0

    # OBJ1: minimize(e1) -> LB1
    # OBJ2: maximize(e1) -> UB1
    # OBJ3: minimize(e2) -> LB2
    # OBJ4: maximize(e2) -> UB2

    model = Model("ErrorRangeModel")
    model.setParam("OutputFlag", 0)
    model.setParam(GRB.Param.FeasibilityTol, 1e-5)

    number_of_error_terms = current_lower_bounds.nelement()

    # indices = list(range(number_of_error_terms))
    # error_term_vars_dict = model.addVars(indices, lb=current_lower_bounds.tolist(), ub=current_upper_bounds.tolist(), name="E", vtype=GRB.CONTINUOUS)
    # error_term_vars = [error_term_vars_dict[index] for index in indices]  # In the same order

    error_term_vars = []
    for error_term_num in range(number_of_error_terms):
        # 0.5 <= e1 <= 1
        error_term_var = model.addVar(vtype=GRB.CONTINUOUS, name="E" + str(error_term_num),
                                      lb=current_lower_bounds[error_term_num], ub=current_upper_bounds[error_term_num])

        # Store new variables
        error_term_vars.append(error_term_var)

    for equation_num in range(linear_expressions_less_or_equal_to_zero.size(0)):
        equation = linear_expressions_less_or_equal_to_zero[equation_num]
        linear_exp_coefficients = equation[1:]

        # Denser alternative: linear_expression = LinExpr(equation.tolist(), error_term_vars)
        linear_expression = LinExpr([(c, e) for (c, e) in zip(linear_exp_coefficients.tolist(), error_term_vars) if c != 0])
        linear_expression.addConstant(equation[0].item())

        # TODO: <= or < ?
        model.addConstr(linear_expression, GRB.LESS_EQUAL, 0)

    # all_variables = lower_bounds_vars + upper_bound_vars + error_term_vars
    all_variables = error_term_vars
    return model, all_variables
