from ot_sparse_projection.proximal import prox_l1
import numpy as np


def get_sparse_projection(x, sparsity):
    """ Computes a soft-thresholding on x yielding a desired sparsity
    Args:
        - x: a vector
        - sparsity: a real value between 0 and 1
    Returns:
        a soft-thresholded x yielding a desired sparsity
     """
    n = int(x.size * sparsity)
    y = np.abs(x).ravel()
    y.sort()
    val = y[-n]
    return prox_l1(x, val)


def thresholding(x, sparsity):
    """ Computes a hard-thresholding on x yielding a desired sparsity
    Args:
        - x: a vector
        - sparsity: a real value between 0 and 1
    Returns:
        a hard-thresholded x yielding a desired sparsity
     """
    n = int(x.size * sparsity)
    y = np.abs(x).ravel()
    y.sort()
    val = y[-n]
    y = x.copy()
    y[np.abs(y) <= val] = 0
    return y


def sparse_projection(x, transform, sparsity, projection=get_sparse_projection):
    """ Computes a sparse projection on the coefficients of x through transform
    
    Args:
        - x: a single-channel image
        - transform: a matrix object from matrix.py or dictionaries.py (or subclass)
        - sparsity: a real value between 0 and 1
        - projection: a sparse projection operator
    Returns:
         - Y: the projected image
         - Z: the projected coefficients
    """
    Z = transform.inverse_dot(x)
    Z = projection(Z, sparsity)
    Z = transform.reshape_coeffs(Z)
    Y = transform.dot(Z)
    return Y.reshape(x.shape), Z


def hard_thresholding(x, transform, sparsity):
    """ Computes a hard thresholding on the coefficients of x through transform

    Args:
        - x: a single-channel image
        - transform: a matrix object from matrix.py or dictionaries.py (or subclass)
        - sparsity: a real value between 0 and 1
    Returns:
         - Y: the projected image
         - Z: the projected coefficients
    """
    return sparse_projection(x, transform, sparsity, projection=thresholding)
