import numpy as np


def get_ball(dist_fun, sz, radius):
    shape = np.ones([sz, sz])*64
    c = float(sz)/2
    for i in range(sz):
        for j in range(sz):
            if dist_fun(np.array([c, c]), np.array([i, j]))<radius:
                shape[i,j] = 256
    return shape


def l1(a, b):
    return np.abs(a-b).sum()


def l_inf(a, b):
    return np.abs(a-b).max()


def l2(a, b):
    return np.sqrt(((a-b)**2).sum())


def get_circle(sz, radius):
    return get_ball(l2, sz, radius)


def get_diamond(sz, radius):
    return get_ball(l1, sz, radius)


def get_square(sz, radius):
    return get_ball(l_inf, sz, radius)