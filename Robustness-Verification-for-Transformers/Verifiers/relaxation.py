#from gurobipy import *
import numpy as np
import torch


def f(x, y, name="tanh"):
    # Calculates sigmoid(x) * tanh(y) or sigmoid(x) * y or x * y.
    if name == "tanh":
        return np.tanh(y) / (1 + np.exp(-x))
    elif name == "mult":
        return x * y
    elif name == "sigmoid":
        return y / (1 + np.exp(-x))
    elif name == "divide":
        return x / y
    else:
        raise Exception("Function with name %s is not supported" % name)


def sigma(x):
    # Calculates sigmoid(x).
    return 1 / (1 + np.exp(-x))


def ibp_bounds(lx, ux, ly, uy, name="tanh"):
    # Calculates IBP bounds of [lx,ux] X [ly,uy].

    # They use four corners to find the lower and upper bounds.
    # This DOES NOT WORK IN GENERAL, it only works because the 2 considered functions (sigmoid(x) * tanh(y))
    # and (sigmoid(x) * y) are monotonically increasing wrt both their arguments x and y

    if type(lx) is torch.Tensor: lx = lx.item()
    if type(ux) is torch.Tensor: ux = ux.item()
    if type(ly) is torch.Tensor: ly = ly.item()
    if type(uy) is torch.Tensor: uy = uy.item()
    candidates = torch.tensor([f(lx, ly, name), f(lx, uy, name),
                               f(ux, ly, name), f(ux, uy, name)])
    return 0, 0, torch.min(candidates), 0, 0, torch.max(candidates)


def proper_roots(equ, lx, ux, ly, uy, var='x', fn=(lambda v: v)):
    # Equation solver and filter the roots satisfying the proper conditions.
    Xs = np.zeros([0])
    Ys = np.zeros([0])
    roots = np.roots(equ)
    if var == 'x':
        for sx in roots:
            if np.iscomplex(sx) or sx >= 1 or sx <= 0:  # The value of the sigmoid must be in (0, 1)
                continue
            sx = np.real(sx)
            x = -np.log((1 - sx) / sx)  # The solution is the value of the sigmoid(x) and here we compute x
            y = fn(sx)
            if lx <= x <= ux and ly <= y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    elif var == 'y':
        for ty in roots:
            if np.iscomplex(ty) or ty >= 1 or ty <= -1:  # The value of the tanh must be in (-1, 1)
                continue
            ty = np.real(ty)
            y = np.arctan(ty)
            x = fn(ty)
            if lx <= x <= ux and ly <= y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    return Xs, Ys


# The functions with delta in the name computes the maximum violation

def get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, name):
    # Cl calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if name == "sigmoid":
        # Case 2 only
        Xs1, Ys1 = proper_roots([1, -1, Al / ly], lx, ux, ly, uy, var='x', fn=(lambda v: ly))
        Xs2, Ys2 = proper_roots([1, -1, Al / uy], lx, ux, ly, uy, var='x', fn=(lambda v: uy))
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    elif name == "tanh":
        # Case 1
        Xs1, Ys1 = proper_roots([1, 0, -1, Bl / sigma(lx)], lx, ux, ly, uy, var='y', fn=(lambda v: lx))
        Xs2, Ys2 = proper_roots([1, 0, -1, Bl / sigma(ux)], lx, ux, ly, uy, var='y', fn=(lambda v: ux))
        # Case 2
        Xs3, Ys3 = proper_roots([1, -1, Al / np.tanh(ly)], lx, ux, ly, uy, var='x', fn=(lambda v: ly))
        Xs4, Ys4 = proper_roots([1, -1, Al / np.tanh(uy)], lx, ux, ly, uy, var='x', fn=(lambda v: uy))
        # Case 3
        Xs5, Ys5 = proper_roots([1, -2 - Bl, 1 + 2 * Bl, -Bl, -Al * Al],
                                lx, ux, ly, uy, var='x', fn=(lambda v: Al / v / (1 - v)))
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])
    elif name == "mult":
        # Testing the corners is enough
        pass
    elif name == "divide":
        if Bl != 0.0:
            # Case 1
            Xs1, Ys1 = proper_roots([1, 0, -lx / Bl], lx, ux, ly, uy, var='y', fn=(lambda v: lx))
            Xs2, Ys2 = proper_roots([1, 0, -ux / Bl], lx, ux, ly, uy, var='y', fn=(lambda v: ux))
            bndX = np.concatenate([bndX, Xs1, Xs2])
            bndY = np.concatenate([bndY, Ys1, Ys2])

        # Case 2 (not needed). If it's happens in one point of those boundaries, it happens in all.
        # Testing all corners is enough to cover this case
        # Case 3
        Xs5, Ys5 = proper_roots([Al ** 2, Bl], lx, ux, ly, uy, var='x', fn=(lambda v: 1 / Al))
        bndX = np.concatenate([bndX, Xs5])
        bndY = np.concatenate([bndY, Ys5])
    else:
        raise Exception("Function %s not supported yet" % name)

    # In the previous part, we found the position were the violation could be maximized,
    # but never actually computed the violation. In here, for all relevant points, we compute the violation
    # and then return that. A violation is negative (the function is below the plane, and we need to decrease C
    # in the lower bound case, so it all works out well)
    delta = np.min(f(bndX, bndY, name) - Al * bndX - Bl * bndY - Cl)
    return delta


def get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, name):
    # Cu calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if name == "sigmoid":
        # Case 2 only
        Xs1, Ys1 = proper_roots([1, -1, Au / ly], lx, ux, ly, uy, var='x', fn=(lambda v: ly))
        Xs2, Ys2 = proper_roots([1, -1, Au / uy], lx, ux, ly, uy, var='x', fn=(lambda v: uy))
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    elif name == "tanh":
        # Case 1
        Xs1, Ys1 = proper_roots([1, 0, -1, Bu / sigma(lx)], lx, ux, ly, uy, var='y', fn=(lambda v: lx))
        Xs2, Ys2 = proper_roots([1, 0, -1, Bu / sigma(ux)], lx, ux, ly, uy, var='y', fn=(lambda v: ux))
        # Case 2
        Xs3, Ys3 = proper_roots([1, -1, Au / np.tanh(ly)], lx, ux, ly, uy, var='x', fn=(lambda v: ly))
        Xs4, Ys4 = proper_roots([1, -1, Au / np.tanh(uy)], lx, ux, ly, uy, var='x', fn=(lambda v: uy))
        # Case 3
        Xs5, Ys5 = proper_roots([1, -2 - Bu, 1 + 2 * Bu, -Bu, -Au * Au],
                                lx, ux, ly, uy, var='x', fn=(lambda v: Au / v / (1 - v)))
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])
    elif name == "mult":
        # Testing the corners is enough
        pass
    elif name == "divide":
        # Case 1
        if Bu != 0.0:
            Xs1, Ys1 = proper_roots([1, 0, -lx / Bu], lx, ux, ly, uy, var='y', fn=(lambda v: lx))
            Xs2, Ys2 = proper_roots([1, 0, -ux / Bu], lx, ux, ly, uy, var='y', fn=(lambda v: ux))
            bndX = np.concatenate([bndX, Xs1, Xs2])
            bndY = np.concatenate([bndY, Ys1, Ys2])

        # Case 2 (not needed). If it's happens in one point of those boundaries, it happens in all.
        # Testing all corners is enough to cover this case
        # Case 3
        Xs5, Ys5 = proper_roots([Au ** 2, Bu], lx, ux, ly, uy, var='x', fn=(lambda v: 1 / Au))
        bndX = np.concatenate([bndX, Xs5])
        bndY = np.concatenate([bndY, Ys5])
    else:
        raise Exception("Not supported yet")

    delta = np.min(Au * bndX + Bu * bndY + Cu - f(bndX, bndY, name))
    return delta


def LB(lx, ux, ly, uy, name="tanh", n_samples=100):
    # Calculate Al, Bl, Cl by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])  # -4 because we want to have the corners
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])  # -4 because we want to have the corners

    model = Model()
    model.setParam('OutputFlag', 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Al')
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Bl')
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Cl')

    model.addConstrs((Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], name) \
                      for i in range(n_samples)), name='ctr')

    obj = LinExpr()
    obj = np.sum(f(X, Y, name)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr('x', model.getVars())
        delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, name)
        Cl += delta
        return Al, Bl, Cl
    else:
        return None, None, None


def UB(lx, ux, ly, uy, name="tanh", n_samples=100):
    # Calculate Au, Bu, Cu by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])

    model = Model()
    model.setParam('OutputFlag', 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Au')
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Bu')
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Cu')

    model.addConstrs((Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], name) \
                      for i in range(n_samples)), name='ctr')

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n_samples - np.sum(f(X, Y, name))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr('x', model.getVars())
        delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, name)
        Cu -= delta
        return Au, Bu, Cu
    else:
        return None, None, None


def bounds(lx, ux, ly, uy, name="tanh"):
    # Caller function to obtain lower and upper bounding planes.
    if type(lx) is torch.Tensor: lx = lx.item()
    if type(ux) is torch.Tensor: ux = ux.item()
    if type(ly) is torch.Tensor: ly = ly.item()
    if type(uy) is torch.Tensor: uy = uy.item()
    Al, Bl, Cl = LB(lx, ux, ly, uy, name)
    Au, Bu, Cu = UB(lx, ux, ly, uy, name)
    return Al, Bl, Cl, Au, Bu, Cu


# This is similar to normal LB, except they only sample from part of it

def LB_split(lx, ux, ly, uy, name="tanh", split_type=0, n_samples=200):
    # Get lower bound plane with triangular domain.
    X = np.random.uniform(lx, ux, n_samples)
    Y = np.random.uniform(lx, ux, n_samples)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam('OutputFlag', 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Al')
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Bl')
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Cl')

    model.addConstrs((Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], name) \
                      for i in range(n)), name='ctr')

    obj = LinExpr()
    obj = np.sum(f(X, Y, name)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n
    if split_type == 11:
        obj -= f(lx, uy, name) - Al * lx - Bl * uy - Cl
    elif split_type == 12:
        obj -= f(ux, ly, name) - Al * ux - Bl * ly - Cl
    elif split_type == 21:
        obj -= f(ux, uy, name) - Al * ux - Bl * uy - Cl
    elif split_type == 22:
        obj -= f(lx, ly, name) - Al * lx - Bl * ly - Cl
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr('x', model.getVars())
        Al, Bl, Cl = adjust_plane_down(Al, Bl, Cl, lx, ux, ly, uy, name)
        return Al, Bl, Cl, model.objVal / (n - 1)
    else:
        print("Couldn't find optimal solution! Model status code: %s" % model.status)
        import pdb; pdb.set_trace()
        return None, None, None, None


def UB_split(lx, ux, ly, uy, name="tanh", split_type=0, n_samples=200):
    # Get upper bound plane with triangular domain.
    X = np.random.uniform(lx, ux, n_samples)
    Y = np.random.uniform(lx, ux, n_samples)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam('OutputFlag', 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Au')
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Bu')
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='Cu')

    model.addConstrs((Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], name) \
                      for i in range(n)), name='ctr')

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n - np.sum(f(X, Y, name))
    if split_type == 11:
        obj += f(lx, uy, name) - Au * lx - Bu * uy - Cu
    elif split_type == 12:
        obj += f(ux, ly, name) - Au * ux - Bu * ly - Cu
    elif split_type == 21:
        obj += f(ux, uy, name) - Au * ux - Bu * uy - Cu
    elif split_type == 22:
        obj += f(lx, ly, name) - Au * lx - Bu * ly - Cu
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr('x', model.getVars())
        Au, Bu, Cu = adjust_plane_up(Au, Bu, Cu, lx, ux, ly, uy, name)
        return Au, Bu, Cu, model.objVal / (n - 1)
    else:
        print("Couldn't find optimal solution! Model status code: %s" % model.status)
        import pdb; pdb.set_trace()
        return None, None, None, None


def adjust_plane_down(Al, Bl, Cl, lx, ux, ly, uy, name):
    delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, name)
    Cl += delta
    return Al, Bl, Cl


def adjust_plane_up(Au, Bu, Cu, lx, ux, ly, uy, name):
    delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, name)
    Cu -= delta
    return Au, Bu, Cu


if __name__ == '__main__':
    print(LB(-1, 2, -2, 3))
    print(UB(-1, 2, -2, 3))
