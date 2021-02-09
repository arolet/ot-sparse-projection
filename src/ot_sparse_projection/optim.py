import math
import time
import warnings

import numpy as np

from ot_sparse_projection import log_utils

MAX_ITER = 500
BACKTRACKING = "backtracking"
FEASIBLE = "feasible"
NO_BACKTRACKING = "none"
DECREASING = "decreasing"


class TimerInterface(object):
    """ An object used to track the evolution of the objective of an optimization method with respect to time

     This basic implementation is a dummy which does nothing
     """

    def register(self, *args, **kwargs):
        pass

    def reset(self):
        self.__init__()


class Timer(TimerInterface):
    """ An object used to track the evolution of the objective of an optimization method with respect to time

     This basic implementation registers the objective and the time at which it was registered. It is assumed that
     computational time for the objective is negligible
     """

    def __init__(self):
        self.times = []
        self.elapsed = 0
        self.stamp = None
        self.__stamp()
        self.objectives = []

    def __stamp(self):
        self.stamp = time.time()

    def __get_elapsed(self):
        return time.time() - self.stamp

    def register(self, obj):
        self.elapsed = self.elapsed + self.__get_elapsed()
        self.objectives.append(obj)
        self.times.append(self.elapsed)
        self.__stamp()


DEFAULT_TIMER = TimerInterface


def return_0(x):
    return 0


def fista(obj, prox, x0, max_iter=MAX_ITER, max_linesearch=100, alpha=.3, beta=.6, tol=1e-3, prox_val=return_0,
          logger=log_utils.nothing, step_size=1, linesearch_type=BACKTRACKING, timer=DEFAULT_TIMER()):
    """ Implementation of the FISTA algorithm for minimizing a sum of a smooth function and one with a proximal operator

    Args:
        obj (function): the smooth part of the objective. obj(x, true) returns the value and the gradient evaluated at x
        prox (function): the proximal operator for the non-smooth part of the gradient
        x0 (numpy.array): the initial point
        max_iter (int): the maximum number of outer iterations
        max_linesearch (int): the maximum number of iterations in a single linesearch
        alpha (float): rate of decrease parameter in line-search
        beta (float): step-size multiplier in line-search
        tol (float): the precision to which the problem is solved
        prox_val (function): function that evaluates the non-smooth part of the objective. Defaults to a 0 constant
        logger (log_utils.Logger): logger function to periodically print optimization information
        step_size (float): the initial gradient step size
        linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
           otherwise use a fix step size which only checks for feasibility
        timer (TimerInterface): an object which tracks the evolution of the objective with respect to time.
           basic implementation doesn't track anything

    Returns:
        x (numpy.array of same dimension as x0): the optimizer, or current point if max_iter was reached
    """
    x = x0
    v = x
    y = x

    if linesearch_type == FEASIBLE:
        linesearch = backtracking_linesearch_for_feasible_point
    elif linesearch_type == BACKTRACKING:
        linesearch = backtracking_linesearch
    elif linesearch_type == NO_BACKTRACKING:
        linesearch = no_backtracking
    elif linesearch_type == DECREASING:
        linesearch = backtracking_linesearch
    else:
        raise ValueError("Unknown linesearch type")
    timer.reset()
    for it in range(1, max_iter + 1):
        f, grad = obj(y, True)
        x_prev = x
        x, sqdist, step_size, fu = linesearch(obj, prox, f, y, grad, step_size,
                                              max_linesearch=max_linesearch, alpha=alpha, beta=beta)

        fu = fu + prox_val(x)
        criterion = (np.sqrt(sqdist)) / (step_size * (1 + (np.sqrt((y ** 2).sum()))))
        logger("{}. Objective: {}, criterion: {}, step_size: {}".format(it + 1, float(fu), float(criterion), step_size),
               i=it)

        timer.register(float(fu))
        if criterion <= tol:
            logger("Converged. Iteration {}. Objective: {}, criterion: {}, step_size: {}".format(it + 1, float(fu),
                                                                                                 float(criterion),
                                                                                                 step_size))
            return x

        theta = 2. / (it + 1)
        y = (1 - theta) * x + theta * v
        v = (1 - 1 / theta) * x_prev + x_prev / theta
        step_size /= beta
        if step_size < 1e-15:
            step_size = 1e-15

    warnings.warn(("Maximum number of iterations reached. Objective: {}, "
                  "criterion: {}, step_size: {}").format(float(f), float(criterion), step_size))
    return x


def fista_descent(obj, prox, x0, max_iter=MAX_ITER, max_linesearch=100, alpha=.3, beta=.6, tol=1e-3, prox_val=return_0,
                  logger=log_utils.nothing, step_size=1, linesearch_type=BACKTRACKING, timer=DEFAULT_TIMER()):
    """ A variation of the FISTA algorithm where the objective decreases at each iteration

    I haven't used/tested this function much, use with caution

    Args:
        obj (function): the smooth part of the objective. obj(x, true) returns the value and the gradient evaluated at x
        prox (function): the proximal operator for the non-smooth part of the gradient
        x0 (numpy.array): the initial point
        max_iter (int): the maximum number of outer iterations
        max_linesearch (int): the maximum number of iterations in a single linesearch
        alpha (float): rate of decrease parameter in line-search
        beta (float): step-size multiplier in line-search
        tol (float): the precision to which the problem is solved
        prox_val (function): function that evaluates the non-smooth part of the objective. Defaults to a 0 constant
        logger (log_utils.Logger): logger function to periodically print optimization information
        step_size (float): the initial gradient step size
        linesearch_type (string): if "backtracking" (default), uses a step-length computed by backtracking linesearch
           otherwise use a fix step size which only checks for feasibility
        timer (TimerInterface): an object which tracks the evolution of the objective with respect to time.
           basic implementation doesn't track anything

    Returns:
        x (numpy.array of same dimension as x0): the optimizer, or current point if max_iter was reached
    """
    y = x0
    x = y
    theta = 1.

    if linesearch_type == FEASIBLE:
        linesearch = backtracking_linesearch_for_feasible_point
    elif linesearch_type == BACKTRACKING:
        linesearch = backtracking_linesearch
    elif linesearch_type == NO_BACKTRACKING:
        linesearch = no_backtracking
    elif linesearch_type == DECREASING:
        linesearch = backtracking_linesearch
    else:
        raise ValueError("Unknown linesearch type")
    timer.reset()
    for it in range(1, max_iter + 1):
        f, grad = obj(y, True)

        u, sqdist, step_size, fu = linesearch(obj, prox, f, y, grad, step_size,
                                              max_linesearch=max_linesearch, alpha=alpha, beta=beta)

        criterion = (np.sqrt(sqdist)) / (step_size * (1 + (np.sqrt((y ** 2).sum()))))
        logger("{}. Objective: {}, criterion: {}, step_size: {}".format(it + 1, float(f), float(criterion), step_size),
               i=it)

        timer.register(float(f))
        if criterion <= tol:
            logger(("Converged. Iteration {}. Objective: {}, "
                   "criterion: {}, step_size: {}").format(it + 1, float(f), float(criterion), step_size))
            return u
        fx = obj(x)
        if fu <= fx:
            x_next = u
        else:
            x_next = x

        theta_next = 2. / (it + 2)
        v = x + (u - x) / theta
        y = (1 - theta_next) * x_next + theta_next * v
        theta = theta_next
        step_size /= beta
        x = x_next

    warnings.warn(("Maximum number of iterations reached. Objective: {},"
                  " criterion: {}, step_size: {}").format(float(f), float(criterion), step_size))
    return u


def backtracking_linesearch(obj, prox, f_old, y, grad, step_size, max_linesearch=100, alpha=.3, beta=.6):
    for it in range(1, max_linesearch + 1):
        y_proj = prox(y - grad * step_size, step_size)
        diff = (y_proj - y).ravel()
        sqdist = (diff * diff).sum()

        f = obj(y_proj)
        Q = f_old + (diff * grad).sum() + alpha * sqdist / step_size

        if f <= Q and not math.isinf(f):
            return y_proj, sqdist, step_size, f

        step_size *= beta
        it += 1
    warnings.warn("Linesearch did not converge")

    return y_proj, sqdist, step_size, f


def backtracking_linesearch_for_decreasing(obj, prox, f_old, y, grad, step_size, max_linesearch=100, alpha=.3, beta=.6):
    for it in range(1, max_linesearch + 1):
        y_proj = prox(y - grad * step_size, step_size)
        diff = (y_proj - y).ravel()
        sqdist = (diff * diff).sum()

        f = obj(y_proj)

        if f <= f_old and not math.isinf(f):
            return y_proj, sqdist, step_size, f

        step_size *= beta
        it += 1
    warnings.warn("Linesearch did not converge")

    return y_proj, sqdist, step_size, f


def backtracking_linesearch_for_feasible_point(obj, prox, f_old, y, grad, step_size, max_linesearch=100, alpha=.3,
                                               beta=.6):
    for it in range(1, max_linesearch + 1):
        y_proj = prox(y - grad * step_size, step_size)
        diff = (y_proj - y).ravel()
        sqdist = (diff * diff).sum()

        f = obj(y_proj)

        if not np.isinf(f) and not np.isnan(f):
            return y_proj, sqdist, step_size * beta, f

        step_size *= beta
        it += 1
    warnings.warn("Linesearch did not converge")

    return y_proj, sqdist, step_size, f


def no_backtracking(obj, prox, f_old, y, grad, step_size, max_linesearch=100, alpha=.3, beta=.6):
    y_proj = prox(y - grad * step_size, step_size)
    diff = (y_proj - y).ravel()
    sqdist = (diff * diff).sum()
    return y_proj, sqdist, step_size * beta, obj(y_proj)


def forward_backward(func, prox_r, lin, func_lipschitz, lin_norm=1., h0=None, y0=None,
                     max_iter=2000000, tol=1e-8, alpha=1. / 3, logger=log_utils.nothing, timer=DEFAULT_TIMER()):
    tau = 1. / func_lipschitz
    sigma = 1. * func_lipschitz / (4 * lin_norm)
    if h0 is None:
        h = np.zeros((lin.rows(),))
    else:
        h = h0
    if y0 is None:
        y = np.zeros((lin.cols()))
    else:
        y = y0
    h_prev = h
    y_prev = y
    timer.reset()

    for it in range(max_iter):
        xi = (alpha + 1) * h - alpha * h_prev
        eta = (alpha + 1) * y - alpha * y_prev
        h_prev = h
        y_prev = y
        obj, grad = func(xi, True)
        h = xi - tau * (grad - lin.dot(eta))
        xi_bar = 2 * h - xi
        y = prox_r(eta - sigma * lin.transpose_dot(xi_bar), sigma)
        criterion = (np.sqrt(((h_prev - h) ** 2).sum())) / (tau * (1 + (np.sqrt((h ** 2).sum()))))
        logger("{}. Objective: {}, criterion: {}".format(it + 1, float(obj),
                                                         float(criterion)), i=it)
        timer.register(float(obj))
        if criterion <= tol:
            logger("Converged. Iteration {}. Objective: {}, criterion: {}".format(it + 1, float(obj),
                                                                                  float(criterion)))
            return h, y, obj

    warnings.warn("Maximum number of iterations reached. Objective: {}, criterion: {}".format(float(obj),
                                                                                              float(criterion)))
    return h, y, obj


def check_taylor_expansion(f, x, dir=None, epsilon=1e-8):
    if dir is None:
        dir = np.random.rand(*x.shape)
    dir *= epsilon / np.sqrt((dir ** 2).sum())
    obj, gradient = f(x, grad=True)
    obj2 = f(x + dir, grad=False)
    return np.abs(obj2 - obj - (gradient.ravel() * dir).sum(axis=0).reshape(1, -1)).sum() / np.abs(epsilon * obj)


def check_gradient(fun, x, epsilon=1e-12):
    from scipy.optimize import check_grad

    def obj(y):
        return float(fun(y, grad=False))

    def gradient(y):
        return fun(y, grad=True)[1].ravel()
    o = obj(x)
    grad = gradient(x)
    return check_grad(obj, gradient, x, epsilon=epsilon) / np.sqrt((grad ** 2).sum())
