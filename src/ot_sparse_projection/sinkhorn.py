import numpy as np
import warnings
from ot_sparse_projection import log_utils
from ot_sparse_projection.histograms import assert_histograms, rescale, log_with_zeros, log_rescaled


def divide_with_zeros(x, y):
    div = (x / y).ravel()
    zeros = np.where(x.ravel() == 0)[0]
    if zeros.size > 0:
        div[zeros] = 0
    return div.reshape(x.shape)


class SinkhornAlgorithm(object):
    """ A class for computing entropy-regularized optimal transport with the Sinkhorn algorithm.
    """

    def __init__(self, ground_metric, check_convergence_period=20, logger=log_utils.get_logger(True, 1, 1000),
                 convergence_threshold=1e-4):
        """
        Args:
            ground_metric (ot_ground_metric.GroundMetricHandler): an object representing a ground metric matrix.
            check_convergence_period (int): the number of iteration between two convergence checks.
            logger (function): a logging function.
            convergence_threshold (float): the precision to which the problem is solved.
        """
        self.__ground_metric = ground_metric
        self.__u = np.ones((ground_metric.cols(), 1))
        self.__check_convergence_period = check_convergence_period
        self.__logger = logger
        self.__convergence_threshold = convergence_threshold

    def compute(self, x, y, u=None, grad=False):
        """ Computes entropy-regularized optimal transport between x and y

        Args:
            x (array): an array of non-negative values.
            y (array): an array of non-negative values, with np.abs(y).sum(axis=1)==np.abs(x).sum(axis=1).
            u (array): optional initialization of the dual variables.
            grad (boolean): whether or not to return the gradient.

        Returns:
            obj: the value of the regularized optimal transport (a vector if x or y have 2 columns or more)
            grap (optional): the gradient with respect to y
        """
        assert_histograms(x, y)
        if u is not None:
            self.__u = u
        else:
            self.__u = rescale(self.__u)
        self.__assert_correct_size(x, y)
        self.__compute(x, y)

        obj = self.__ground_metric.gamma * (x.T.dot(log_with_zeros(self.__u)) + y.T.dot(log_with_zeros(self.__v)) - 1
                                     + self.__u.T.dot(self.__ground_metric.dot(self.__v)))

        if grad:
            return obj, self.get_gradient(self.__v)
        return obj

    def __assert_correct_size(self, x, y):
        if x.shape != self.__u.shape:
            warnings.warn("u does not have the correct shape")
            self.__u = np.ones(x.shape)

    def __compute(self, x, y, restart=1):
        u = self.__u
        self._iteration = 0
        with np.errstate(divide="raise", over="raise"):
            try:
                while self.__check_convergence(x, y):
                    self.__iteration(x, y)
                    u = self.__u
            except FloatingPointError:
                if restart > 1:
                    self.__u = rescale(u)
                elif restart == 1:
                    self.__u = np.ones(self.__u.shape)
                if restart > 0:
                    warnings.warn("Precision error in sinkhorn, rescaling dual variables")
                    self.__compute(x, y, restart=restart - 1)
                else:
                    warnings.warn("Precision error in sinkhorn, early finish")

    def __iteration(self, x, y):
        self.__v = self.__compute_v(y, self.__u)
        self.__u = self.__compute_u(x, self.__v)

    def __compute_v(self, y, u):
        return divide_with_zeros(y, self.__ground_metric.transpose_dot(u))

    def __compute_u(self, x, v):
        return divide_with_zeros(x, self.__ground_metric.dot(v))

    def __check_convergence(self, x, y):
        self._iteration += 1
        if self._iteration % self.__check_convergence_period != 0:
            return True

        criterion = self.__compute_convergence_criterion(self.__u, self.__v, x, y)
        self.__logger("Iteration %d, criterion %s" % (self._iteration, criterion), i=self._iteration)
        return criterion > self.__convergence_threshold

    def __compute_convergence_criterion(self, u, v, x, y):
        tmp = np.multiply(v, self.__ground_metric.transpose_dot(self.__u)) - y
        return np.abs(tmp).sum() / u.shape[1]

    def get_gradient(self, v, error_behavior="warn"):
        with np.errstate(divide=error_behavior):
            v = log_rescaled(v) * self.__ground_metric.gamma
        return v - v.mean(axis=0).reshape(1, -1)
