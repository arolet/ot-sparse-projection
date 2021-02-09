import warnings
import numpy as np


def project_simplex(v, z=1):
    """ projects a non-negative vector v on the intersection of an l1-norm ball and the non-negative orthant
    Based on https://gist.github.com/mblondel/6f3b7aaad90606b98f71

    Args:
        v: a non-negative vector
        z: radius of the l1 ball
    Returns:
        projected vector
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def assert_histograms(x, y):
    """ Asserts that inputs are non-negative vectors with the same l1-norm

    Args:
        x: a vector
        y: a vector
    Returns:
        fails if x and y are not histograms with the same l1 norm
    """
    if np.any(x < 0):
        raise ValueError("Input x should be non-negative")
    if np.any(y < 0):
        raise ValueError("Input y should be non-negative")
    if np.any(np.abs(x.sum(axis=0) - y.sum(axis=0)) / x.sum() > 1e-5):
        raise ValueError("Input x and y should be have same sum")


def get_geometric_mean(x):
    """ Computes the geometric mean of a vector along the first axis

    Args:
        x: a matrix
    Returns:
        the geometric mean of x over the first axis
    """
    a = x.copy()
    nnz = (a != 0).sum(axis=0).reshape(1, -1)
    a[a == 0] = 1
    return np.exp(np.log(a).sum(axis=0).reshape(-1, 1) / nnz)


def get_min(x):
    """ Computes the minimum of non-zero values of x along the first axis

    Args:
        x: a matrix
    Returns:
        the minimum of non-zero values of x
    """
    a = x.copy()
    a[a == 0] = np.inf
    return a.min(axis=0).reshape(1, -1)


def rescale(x):
    """ rescale a matrix by its geometric mean computed along the first axis

    Args:
        x: a matrix
    Returns:
        x/mean, where x is the geometric mean computed along columns
    """
    tmp = get_geometric_mean(x)
    if np.isnan(tmp) or np.isinf(tmp):
        tmp = np.sqrt(x.max(axis=0).reshape(1, -1) * get_min(x))
    return np.divide(x, tmp)


def error(error_behavior):
    if error_behavior is "raise":
        raise FloatingPointError
    elif error_behavior is "warn":
        warnings.warn("Underflow encountered in log")


def log_with_zeros(u, error_behavior="warn"):
    """ computes a logarithm with optional error on 0 values

    Args:
        u: a matrix
        error_behavior: "warn" (default) for a warning on 0 values, for which the log is set to 0. "raise" for raising
            an error. "ignore" or any other value to ignore the error (useful for computing the entropy)
    Returns:
        log(u)
    """
    logu = np.log(u).ravel()
    zeros = np.where(u.ravel() == 0.)[0]
    if zeros.size > 0:
        error(error_behavior)
        logu[zeros] = 0
    return logu.reshape(u.shape)


def log_rescaled(v, error_behavior="warn"):
    """ computes the log of v, where v is rescaled so that the log is finite

    Args:
        v: a matrix
        error_behavior: "warn" (default) for a warning on 0 values, for which the log is set to 0. "raise" for raising an error
    Returns:
        log(v) + cst, where cst is a constant used to get finite values in the log
    """
    x = log_with_zeros(v, error_behavior=error_behavior)
    while np.isinf(x).any():
        v *= 2
        x = log_with_zeros(v, error_behavior=error_behavior)
    return x


def entropy(x):
    """ computes the entropy of x

    Args:
        x: a matrix
    Returns:
        the entropy computed along the first axis
    """
    return np.multiply(x, log_with_zeros(x, error_behavior="ignore")).sum(axis=0)
