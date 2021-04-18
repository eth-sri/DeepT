# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch


def check(name, bounds=None, l=None, u=None, std=None, verbose=False):
    """ Checks that the bounds l - eps <= std and u + eps >= std and same for the bounds
    std is a tensor containing the real values that were obtained in the BERT model.
    Here eps = 0.0001, so l <= real value + 0.0001 and u >= real value - 0.0001
    In other words, l and u can't be too far off from the real values (l can't be above real value
    by more than 0.0001 and u can't be below the real value by more than 0.0001)
    """
    if verbose:
        print("Check ", name)
    eps = 2e-3  # 1e-4

    # Compute concrete upper and lower bound
    if bounds is not None:
        l, u = bounds.concretize()
    if len(l.shape) == 3:
        l, u, std = l[0], u[0], std[0]

    # Compute for which terms we have std < (l - eps) or (u + eps) < std
    c = torch.gt(l - eps, std).to(torch.float) + torch.lt(u + eps, std).to(torch.float)

    if bounds is not None:
        c += torch.gt(bounds.lb[0] - eps, std).to(torch.float) + torch.lt(bounds.ub[0] + eps, std).to(torch.float)

    # Compute how many terms aren't good
    errors = torch.sum(c)
    score = float(torch.mean(u - l))

    # Print those terms
    if verbose:
        print("%d errors, %.5f average range" % (errors, score))
        if errors > 0:
            print("the maximum l violation is: ", (l - eps - std).max())
            print("the maximum u violation is: ", (std - eps - u).max())
            cnt = 0
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    if c[i][j] > 0:
                        print(i, j)
                        print(l[i][j], u[i][j], std[i][j])
                        cnt += 1
                        if cnt >= 10:
                            assert False
    assert (errors == 0), "max violation l = %s, max violation u = %s" % ((l - eps - std).max(), (std - eps - u).max())


def check_zonotope(name, zonotope=None, l=None, u=None, actual_values=None, verbose=False):
    """ Checks that the bounds l - eps <= std and u + eps >= std and same for the bounds
    std is a tensor containing the real values that were obtained in the BERT model.
    Here eps = 0.0001, so l <= real value + 0.0001 and u >= real value - 0.0001
    In other words, l and u can't be too far off from the real values (l can't be above real value
    by more than 0.0001 and u can't be below the real value by more than 0.0001)
    """
    if verbose:
        print("Check ", name)
    eps = 2e-3  # 1e-4

    # Compute concrete upper and lower bound
    if zonotope is not None:
        l, u = zonotope.concretize()

    if len(l.squeeze().shape) != len(actual_values.squeeze().shape):
        print("Actual values has shape '%s' but bounds have shape '%s'" % (actual_values.shape, l.shape))
        assert False

    l = l.squeeze()
    u = u.squeeze()
    actual_values = actual_values.squeeze()

    if len(l.shape) == 3:
        l, u, actual_values = l[0], u[0], actual_values[0]

    # Compute for which terms we have std < (l - eps) or (u + eps) < std
    c = torch.gt(l - eps, actual_values).to(torch.float) + torch.lt(u + eps, actual_values).to(torch.float)

    # if bounds is not None:
    #     c += torch.gt(bounds.lb[0] - eps, std).to(torch.float) + torch.lt(bounds.ub[0] + eps, std).to(torch.float)

    # Compute how many terms aren't good
    errors = torch.sum(c).item()
    score = float(torch.mean(u - l))

    # Print those terms
    if verbose:
        print("%d errors, %.5f average range" % (errors, score))
        if errors > 0:
            print("the maximum l violation is: ", (l - eps - actual_values).max())
            print("the maximum u violation is: ", (actual_values - eps - u).max())
            cnt = 0
            for i in range(c.shape[0]):
                for j in range(c.shape[1]):
                    if c[i][j] > 0:
                        print(i, j)
                        print(l[i][j], u[i][j], actual_values[i][j])
                        cnt += 1
                        if cnt >= 10:
                            assert False
    assert (errors == 0), "Check '%s' - %d errors: max violation l = %s, max violation u = %s" % (name, errors, ((l - eps) - actual_values).max(), (actual_values - (u + eps)).max())


INFINITY = float('inf')


def dual_norm(p: float) -> float:
    if p == 1:
        return INFINITY
    elif p == 2:
        return 2.0
    elif p > 10:  # represents the infinity norm
        return 1.0
    else:
        raise NotImplementedError("dual_norm: Dual norm only supported for 1-norm (p = 1), 2-norm (p = 2) or inf-norm (p > 10)")


DUAL_INFINITY = dual_norm(INFINITY)