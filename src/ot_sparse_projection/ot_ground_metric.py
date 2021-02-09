from ot_sparse_projection.matrix import MatrixHandler
import numpy as np
from scipy.ndimage import convolve1d

CONVOLUTION = 'convolution'
DEFAULT = CONVOLUTION
MATRIX = 'matrix'


def get_distance_matrix(shape):
    x = np.indices(shape).reshape([len(shape), -1]).swapaxes(0, 1)
    dist = x.reshape([-1, 1, len(shape)]) - x.reshape([1, -1, len(shape)])
    dist = (dist ** 2).sum(axis=2)
    return np.sqrt(dist)


def get_handler(type, shape, gamma):
    if type == CONVOLUTION:
        return ConvolutionGroundMetricHandler(shape, gamma)
    if type == MATRIX:
        return MatrixGroundMetricHandler(get_distance_matrix(shape) ** 2.0, gamma)


class GroundMetricHandler(MatrixHandler):
    """An abstract adapter class for transportation cost matrix.

    Some cost matrix can be represented more efficiently than by storing the actual matrix
    """

    def __init__(self, gamma):
        """
        Arguments:
             gamma: the regularization strength for entropy-regulatized optimal transport
        """
        super(GroundMetricHandler, self).__init__()
        self.gamma = gamma

    def conjugate(self, x, h, entropy=0, grad=False):
        """ Computes the convex conjugate of ot_gamma(x,.) evaluated at h
        Arguments:
             x, h: the argument where we want to evaluate the function
             entropy: optional pre-computed entropy of x
             grad: if true, also returns the gradient
        Returns:
            obj, grad: obj, the convex conjugate of ot_gamma(x,.) evaluated at h; and optionally the gradient grad
        """
        max_e = h.max(axis=0).reshape(1, -1) - self.gamma * 20
        return self._conjugate_unsafe(x, h - max_e, entropy=entropy, grad=grad, add_log=max_e)

    def _conjugate_unsafe(self, x, h, entropy=0, grad=False, add_log=0):
        """ Computes the convex conjugate of ot_gamma(x,.) evaluated at h unsafely, that is without any rescaling
        Arguments:
             x, h: the argument where we want to evaluate the function
             entropy: optional pre-computed entropy of x
             grad: if true, also returns the gradient
             add_log: an additive rescaling used to avoid overflows due to exponentials
        Returns:
            obj, grad: obj, the convex conjugate of ot_gamma(x,.) evaluated at h; and optionally the gradient grad
        """
        alpha = np.exp(h / self.gamma)
        tmp = self.dot(alpha)
        obj = (entropy + np.multiply(x, np.log(tmp) + add_log / self.gamma).sum(axis=0).reshape(1, -1)) * self.gamma
        if not grad:
            return obj
        tmp = self.transpose_dot(np.divide(x, tmp))
        gradient = np.multiply(alpha, tmp)
        return obj, gradient


def get_1d_filter(sz, gamma, threshold):
    filter = np.exp(-np.abs(np.array(range(-sz, sz, 1))).astype(np.float64) ** 2.0 / gamma) - threshold
    return filter[filter > 0]


class ConvolutionGroundMetricHandler(GroundMetricHandler):
    """A representation of a cost matrix that amounts to euclidean distances on a grid, as is the case for grayscale images

    With an entropy regularized optimal transport, all we need is to be able to perform products with the exponentiated
     matrix, which in this case amount to gaussian blur (see Justin Solomon, Fernando de Goes, Gabriel Peyré,
     Marco Cuturi, Adrian Butscher, Andy Nguyen, Tao Du, and Leonidas Guibas. Convolutional wasserstein distances:
     Efficient optimal transportation on geometric domains. ACM Transactions on Graphics, 34(4), July 2015a.
     ISSN 0730-0301.)
    """

    def __init__(self, shape, gamma, threshold=1e-200):
        """
        Arguments:
            shape: a tupple representing the shape of input arguments along each dimension
            gamma: the regularization strength for entropy-regulatized optimal transport
            threshold: a threshold for the exponentiated cost matrix, so that underflows to 0 are reassigned to a small
                        value
        """
        GroundMetricHandler.__init__(self, gamma)
        self._shape = shape
        self._numFilters = len(shape)
        self._filters = [get_1d_filter(sz, gamma, threshold=threshold) for sz in shape]
        self._reshape = shape + (-1,)
        self.threshold = threshold

    def cols(self):
        return np.prod(self._shape)

    def rows(self):
        return np.prod(self._shape)

    def dot(self, x):
        y = x.reshape(self._reshape)
        for i in range(self._numFilters):
            y = convolve1d(y, self._filters[i], axis=i, mode='constant')
        return y.reshape(x.shape) + x * self.threshold

    def transpose_dot(self, x):
        return self.dot(x)


class MatrixGroundMetricHandler(GroundMetricHandler):
    """A cost matrix, actually represented as a matrix"""

    def __init__(self, C, gamma, threshold=1e-200):
        """
        Arguments:
            C: the transportation cost matrix
            gamma: the regularization strength for entropy-regulatized optimal transport
            threshold: a threshold for the exponentiated cost matrix, so that underflows to 0 are reassigned to a small
                        value
        """
        GroundMetricHandler.__init__(self, gamma)
        self._shape = C.shape
        self.M = np.exp(-C / gamma)
        self.M[self.M < threshold] = threshold

    def cols(self):
        return self._shape[1]

    def rows(self):
        return self._shape[0]

    def dot(self, x):
        return self.M.dot(x)

    def transpose_dot(self, x):
        return self.M.T.dot(x)
