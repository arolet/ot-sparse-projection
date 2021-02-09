import numpy as np

from ot_sparse_projection import proximal, optim, ot_ground_metric, log_utils
from ot_sparse_projection.histograms import entropy

TOL = 1e-8
EPS = 1e-15
MAX_ITER = 10000


class OtProjection(object):
    """ An abstract adapter class for computing optimal transport regularized projection of grayscale images

    We solve the projection problem by running FISTA on the duals defined in Rolet and Seguy 2021

    Subclasses need to define _fun, _proximal and to_primal
    """

    def __init__(self, D, gamma, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20, precision=TOL,
                 max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtProjection, self).__init__()
        self._shape = D._shape
        self._shape_dual = D._shape_wav
        self._D = D
        self._ground_metric = ot_ground_metric.get_handler(ground_metric_type, self._shape, gamma)
        self._X = None
        self._max_iter = max_iter
        self._tolerence = precision
        if log_period > 0:
            self._logger = log_utils.get_logger(True, indent_level=0, period=log_period)
        else:
            self._logger = log_utils.nothing()

    def _fun(self, x, grad=False):
        raise NotImplementedError()

    def _prox(self, x, l):
        raise NotImplementedError()

    def projection(self, X, linesearch_type=optim.BACKTRACKING, timer=optim.DEFAULT_TIMER()):
        """ Computes the actual projection

        The dual problem is fed to our implementation of FISTA

        Args:
            X (m-by-n numpy.array): a grayscale image
            linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
               otherwise use a fix step size which only checks for feasibility
            timer (optim.TimerInterface): an object which tracks the evolution of the objective with respect to time.
               basic implementation doesn't track anything

        Returns:
            recons (m-by-n numpy.array): the projected image
            y (m-by-n numpy.array): the projected coefficients
            obj (float): the value of the objective at the solution
        """
        self.setX(X)
        H = optim.fista(self._fun, self._prox, self.initial_point(), logger=self._logger, tol=self._tolerence,
                        prox_val=self.reg_value, alpha=.5, max_iter=self._max_iter, linesearch_type=linesearch_type,
                        timer=timer, step_size=self._get_step_size())
        obj, grad = self._fun(H, grad=True)
        obj = obj + self.reg_value(H)
        y, recons = self.to_primal(H, grad)
        return recons.reshape(self._shape), self._D.reshape_coeffs(y), obj

    def to_primal(self, H, grad):
        raise NotImplementedError()

    def setX(self, X):
        self._X = X.reshape(-1, 1)
        self.hx = entropy(self._X)

    def initial_point(self):
        return np.zeros(self._shape_dual).ravel()

    def reg_value(self, x):
        return 0

    def _get_step_size(self):
        return 100 / self._ground_metric.gamma


class OtSparseProjectionOrthogonal(OtProjection):
    """ A class for computing optimal transport sparse projection when the dictionary is orthonormal

    In this case, we can simply minimize OT^*(x, h)+R^*(-D^Th) directly with FISTA, since we have
    a formula for the proximal of the composition of R^* and D^T

    _fun is simply OT^*(x, h), _proximal is obtained through the formula
    """

    def __init__(self, D, gamma, lamb, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                 precision=TOL, max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            lamb (float): the l_1 regularization strength of the projection
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtSparseProjectionOrthogonal, self).__init__(D, gamma, ground_metric_type=ground_metric_type,
                                                           log_period=log_period, precision=precision,
                                                           max_iter=max_iter)
        D.assert_orthonormal()
        self._lamb = lamb

    def _fun(self, x, grad=False):
        hx = entropy(self._X)
        if grad:
            obj, gradient = self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=hx)
            return obj, gradient.ravel()
        return self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=hx)

    def _prox(self, x, l):
        return self._D.dot(proximal.prox_l_inf_ball(self._D.transpose_dot(x), self._lamb))

    def to_primal(self, H, grad):
        y = self._D.transpose_dot(grad)
        y[np.abs(self._D.transpose_dot(H)) < self._lamb - EPS] = 0
        return y, grad


class OtSparseProjectionInvertible(OtProjection):
    """ A class for computing optimal transport sparse projection when the dictionary is invertible

    In this case, we solve the problem min_g OT^*(x, D^{-1}g)+R^*(g) with FISTA
    """

    def __init__(self, D, gamma, lamb, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                 precision=TOL, max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            lamb (float): the l_1 regularization strength of the projection
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtSparseProjectionInvertible, self).__init__(D, gamma, ground_metric_type=ground_metric_type,
                                                           log_period=log_period, precision=precision,
                                                           max_iter=max_iter)
        self._lamb = lamb

    def _fun(self, x, grad=False):
        if grad:
            obj, gradient = self._ground_metric.conjugate(self._X, -self._D.inverse_transpose_dot(x).reshape(-1, 1),
                                                          grad=grad, entropy=self.hx)
            return obj, -self._D.inverse_dot(gradient).ravel()
        return self._ground_metric.conjugate(self._X, -self._D.inverse_transpose_dot(x).reshape(-1, 1), grad=grad,
                                             entropy=self.hx)

    def _prox(self, x, l):
        return proximal.prox_l_inf_ball(x, self._lamb)

    def to_primal(self, G, grad):
        # Note: here G is not directly the dual variable H, but -D^{T-1}H
        obj, grad = self._fun(G, grad=True)
        grad[np.abs(G) < self._lamb - EPS] = 0
        return -grad, -self._D.dot(grad)


class OtL2ProjectionInvertible(OtProjection):
    """ Same as OtSparseProjectionInvertible but for a l2 regularized projection
    """

    def __init__(self, D, gamma, lamb, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                 precision=TOL, max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            lamb (float): the l_2 regularization strength of the projection
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtL2ProjectionInvertible, self).__init__(D, gamma, ground_metric_type=ground_metric_type,
                                                       log_period=log_period, precision=precision, max_iter=max_iter)
        self._lamb = lamb

    def _fun(self, x, grad=False):
        if grad:
            obj, gradient = self._ground_metric.conjugate(self._X, -self._D.inverse_transpose_dot(x).reshape(-1, 1),
                                                          grad=grad, entropy=self.hx)
            return obj, -self._D.inverse_dot(gradient).ravel()
        return self._ground_metric.conjugate(self._X, -self._D.inverse_transpose_dot(x).reshape(-1, 1), grad=grad,
                                             entropy=self.hx)

    def _prox(self, x, l):
        return proximal.prox_l2(x, l / (4 * self._lamb))

    def to_primal(self, G, grad):
        # Note: here G is not directly the dual variable H, but -D^{T-1}H
        obj, grad = self._fun(G, grad=True)
        return -grad, -self._D.dot(grad)

    def reg_value(self, x):
        return -(x ** 2).sum() / (4 * self._lamb)


class OtFiltering(OtProjection):
    """ Low-pass filter on the first k coefficients of an invertible dictionary
    """

    def __init__(self, D, gamma, k, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20, precision=TOL,
                 max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            k (int): the number of coefficients which are not set to zero
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtFiltering, self).__init__(D, gamma, ground_metric_type=ground_metric_type,
                                          log_period=log_period, precision=precision, max_iter=max_iter)
        self._k = k

    def _fun(self, x, grad=False):
        if grad:
            obj, gradient = self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=self.hx)
            return obj, gradient.ravel()
        return self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=self.hx)

    def _prox(self, x, l):
        return self._D.high_pass_filter(x, self._k)[0]

    def to_primal(self, H, grad):
        grad, y = self._D.low_pass_filter(grad, self._k)
        return y, grad


class OtFilteringSpecificPattern(OtProjection):
    """ pass filter on specified coefficients of an invertible dictionary
    """

    def __init__(self, D, gamma, sparsity_pattern, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                 precision=TOL, max_iter=MAX_ITER):
        """
        Args:
            D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
            gamma (float): the regularization strength of optimal transport
            sparsity_pattern (array): the mask array for the coefficients which are kept
            ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
            log_period (int): how often logging messages should be printed
            precision (float): the precision to which the dual problem is solved
            max_iter (int): max number of iterations
        """
        super(OtFilteringSpecificPattern, self).__init__(D, gamma, ground_metric_type=ground_metric_type,
                                                         log_period=log_period, precision=precision, max_iter=max_iter)
        self._sparsity_pattern = sparsity_pattern

    def _fun(self, x, grad=False):
        if grad:
            obj, gradient = self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=self.hx)
            return obj, gradient.ravel()
        return self._ground_metric.conjugate(self._X, x.reshape(-1, 1), grad=grad, entropy=self.hx)

    def _prox(self, x, l):
        return self._D.pass_filter(x, self._sparsity_pattern)[0]

    def to_primal(self, H, grad):
        grad, y = self._D.block_filter(grad, self._sparsity_pattern)
        return y, grad


def wasserstein_image_filtering_orthogonal_dictionary(X, D, gamma, lamb, linesearch_type=optim.BACKTRACKING,
                                                      timer=optim.DEFAULT_TIMER(),
                                                      ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                                                      precision=TOL):
    """ Convenience method for computing a sparse image projection of an image on an orthonormal basis

    Args:
        X (m-by-n numpy.array): a grayscale image
        D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
        gamma (float): the regularization strength of optimal transport
        lamb (float): the l_1 regularization strength of the projection
        linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
           otherwise use a fix step size which only checks for feasibility
        timer (optim.TimerInterface): an object which tracks the evolution of the objective with respect to time.
           basic implementation doesn't track anything
        ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
        log_period (int): how often logging messages should be printed
        precision (float): the precision to which the dual problem is solved
    """
    projector = OtSparseProjectionOrthogonal(D, gamma, lamb, ground_metric_type=ground_metric_type,
                                             log_period=log_period, precision=precision)
    return projector.projection(X, linesearch_type=linesearch_type, timer=timer)


def wasserstein_image_filtering_invertible_dictionary(X, D, gamma, lamb, linesearch_type=optim.BACKTRACKING,
                                                      timer=optim.DEFAULT_TIMER(),
                                                      ground_metric_type=ot_ground_metric.DEFAULT,
                                                      log_period=20, precision=TOL):
    """ Convenience method for computing a sparse image projection of an image on an invertible basis

    Args:
        X (m-by-n numpy.array): a grayscale image
        D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
        gamma (float): the regularization strength of optimal transport
        lamb (float): the l_1 regularization strength of the projection
        linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
           otherwise use a fix step size which only checks for feasibility
        timer (optim.TimerInterface): an object which tracks the evolution of the objective with respect to time.
           basic implementation doesn't track anything
        ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
        log_period (int): how often logging messages should be printed
        precision (float): the precision to which the dual problem is solved
    """
    projector = OtSparseProjectionInvertible(D, gamma, lamb, ground_metric_type=ground_metric_type,
                                             log_period=log_period, precision=precision)
    return projector.projection(X, linesearch_type=linesearch_type, timer=timer)


def wasserstein_image_low_pass(X, D, k, gamma, linesearch_type=optim.BACKTRACKING, timer=optim.DEFAULT_TIMER(),
                               ground_metric_type=ot_ground_metric.DEFAULT, log_period=20, tol=TOL):
    projector = OtFiltering(D, gamma, k, ground_metric_type=ground_metric_type, log_period=log_period,
                            precision=tol)
    return projector.projection(X, linesearch_type=linesearch_type, timer=timer)


def wasserstein_image_projection_forward_backward(X, D, gamma, lamb, timer=optim.DEFAULT_TIMER(),
                                                  ground_metric_type=ot_ground_metric.DEFAULT, max_iter=optim.MAX_ITER):
    """ Convenience method for computing a sparse image projection of an image using a primal-dual method

    Args:
        X (m-by-n numpy.array): a grayscale image
        D (matrix.Matrix): the basis on which to project. Standard implementations in dictionaries.py
        gamma (float): the regularization strength of optimal transport
        lamb (float): the l_1 regularization strength of the projection
        linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
           otherwise use a fix step size which only checks for feasibility
        timer (optim.TimerInterface): an object which tracks the evolution of the objective with respect to time.
           basic implementation doesn't track anything
        ground_metric_type (string): how to represent the cost matrix, defaults to ot_ground_metric.CONVOLUTION
        max_iter (int): max number of iterations
    """
    ground_metric = ot_ground_metric.get_handler(ground_metric_type, D._shape, gamma)
    X = X.reshape(-1, 1)
    hx = entropy(X)

    def fun(x, grad=False):
        if grad:
            obj, gradient = ground_metric.conjugate(X, x.reshape(-1, 1), grad=grad, entropy=hx)
            return obj, gradient.ravel()
        return ground_metric.conjugate(X, x.reshape(-1, 1), grad=grad, entropy=hx)

    def prox(x, l):
        return proximal.prox_l1(x, l * lamb)

    h, y, obj = optim.forward_backward(fun, prox, D, 1. / gamma, timer=timer,
                                       logger=log_utils.get_logger(True, indent_level=0, period=1000), max_iter=max_iter)
    return D.dot(y), y, obj
