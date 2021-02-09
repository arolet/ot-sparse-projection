import numpy as np
from ot_sparse_projection.histograms import project_simplex


def prox_l1(x, l):
    """ Proximal operator for the l1 norm

    Args:
        x: a vector
        l: a non-negative value
    Returns:
        argmin ||x-y||_2**2 + l*||y||_1**2
    """
    return np.sign(x) * np.maximum(np.abs(x) - l, 0)


def prox_l_inf_ball(x, l):
    """ Computes the projection on an l_inf ball

    Args:
        x: a vector
        l: a non-negative value
    Returns:
        the projection of x on the l_inf ball of radius l
        """
    return np.sign(x) * np.minimum(np.abs(x), l)


def prox_l_inf_ball_2(x, l):
    y = x.copy()
    y[y > l] = l
    y[y < -l] = -l
    return y


def project_l1(v, z=1):
    """ projection on an l1-norm ball

    Args:
        v: a vector
        z: radius of the l1 ball
    Returns:
        projected vector
    """
    vv = np.abs(v)
    if vv.sum() <= z:
        return v
    return np.multiply(np.sign(v), project_simplex(vv, z=z))


def prox_l2(x, l):
    """ Proximal operator for the l2 norm

    Args:
        x: a vector
        l: a non-negative value
    Returns:
        argmin ||x-y||_2**2 + l*||y||_2**2
    """
    return x / (2 * l + 1)
