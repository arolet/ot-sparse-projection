from os import makedirs
from os.path import join, pardir, exists

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

from ot_sparse_projection import ot_sparse_projection, misc, ot_ground_metric, sinkhorn, optim
from ot_sparse_projection.dictionaries import get_filter_handler
from ot_sparse_projection.optim import Timer
from ot_sparse_projection.proximal import project_simplex

rc('text', usetex=True)

n = 64

folder = join(pardir, 'img', 'timing')
if not exists(folder):
    makedirs(folder)
sinkhorn_precisions = [-2, -4, -6]
# sinkhorn_precisions = [-2]
search_type = optim.BACKTRACKING
filter_type = 'bior4.4'
gamma = 1
lamb = 2000
imName = 'racoon'

class PrimalL2(ot_sparse_projection.OtProjection):

    def __init__(self, D, gamma, lamb, ground_metric_type=ot_ground_metric.DEFAULT, log_period=20,
                 sinkhorn_convergence_threshold=1e-4, max_iter=500):
        super(PrimalL2, self).__init__(D, gamma, ground_metric_type, log_period, max_iter=max_iter)
        self.start_point = None
        self.lamb = float(lamb)
        self.sinkh = sinkhorn.SinkhornAlgorithm(self._ground_metric, convergence_threshold=sinkhorn_convergence_threshold)

    def _fun(self, x, grad=False):
        Dx = self._D.dot(x).reshape(-1,1)
        Dx[Dx<0] = 0

        if grad:
            obj, gradient = self.sinkh.compute(self._X, Dx, grad=grad, u=np.ones((1,1)))
            obj = obj + (x**2).sum()*self.lamb
            gradient = self._D.transpose_dot(gradient).ravel() + (2*self.lamb)*x
            return obj, gradient
        return self.sinkh.compute(self._X, Dx, grad=grad, u=np.ones((1,1))) + (x**2).sum()*self.lamb

    def _prox(self, x, l):
        return self._D.inverse_dot(project_simplex(self._D.dot(x), z=self._X.sum()))

    def to_primal(self, H, grad):
        return H, self._D.dot(H)

    def initial_point(self):
        if self.start_point is None:
            x = np.ones(self._shape).ravel()
        else:
            x = self.start_point
        x = x*self._X.sum()/x.sum()
        return self._D.inverse_dot(x)

    def _get_step_size(self):
        return 1e-7


def compute_projection(im, projector):
    timer = Timer()
    im, y, obj = projector.projection(im, timer=timer, linesearch_type=search_type)
    return im, y, np.array(timer.times).ravel(), np.array(timer.objectives).ravel()


def get_gap(obj_primal, min_primal):
    return (obj_primal-min_primal)/(1+np.abs(min_primal))

def loglog(x, y, label):
    indices = y>0
    plt.loglog(x[indices], y[indices], label=label)


im, scaling = misc.get_image(imName, n)

filter_handler = get_filter_handler(im, filter_type)
im_dual, y_dual, t_dual, obj_dual = compute_projection(im, ot_sparse_projection.OtL2ProjectionInvertible(filter_handler,
                                                                                                         gamma, lamb, log_period=100, precision=1e-14, max_iter=10000))

tmp = PrimalL2(filter_handler, gamma, lamb, log_period=1, sinkhorn_convergence_threshold=1e-2)
tmp.setX(im.ravel())
min_primal_objective = tmp._fun(y_dual.ravel()).ravel()
min_primal_from_dual = min_primal_objective
print(min_primal_from_dual)

primal_objectives = []
primal_times = []
for eps in sinkhorn_precisions:
    projector = PrimalL2(filter_handler, gamma, lamb, log_period=10, max_iter=1000, sinkhorn_convergence_threshold=10**eps)
    im_primal, y_primal, t_primal, obj_primal = compute_projection(im, projector)
    primal_objectives.append(obj_primal)
    primal_times.append(t_primal)
    min_primal_objective = obj_primal.min()

print(min_primal_objective)
print(min_primal_from_dual)
for i in range(len(sinkhorn_precisions)):
    loglog(primal_times[i], get_gap(primal_objectives[i], min(min_primal_objective, min_primal_from_dual)),
               'Primal method, $\sigma=1e{}$'.format(sinkhorn_precisions[i]))
loglog(t_dual, get_gap(obj_dual, obj_dual.min()), label='Dual method')
plt.xlabel('Time (s)')
plt.ylabel('Optimality gap')
plt.legend()
plt.savefig(join(folder, 'primal_times_{}_{}.eps'.format(search_type, filter_type)))
plt.show()

