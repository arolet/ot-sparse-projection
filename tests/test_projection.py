import numpy as np

from ot_sparse_projection import dictionaries, ot_sparse_projection, optim

GAMMA = .1
n = 32
K = int(n / 2)
LAMB = 1


def check_gradient_for_projection(projector, x):
    projector.setX(x)
    e = optim.check_gradient(projector._fun, np.zeros(x.size).ravel(), epsilon=1e-6)
    print(e)
    np.testing.assert_array_less(e, 1e-6)


def check_gradient(shape, type, orthonormal=False):
    x = np.random.rand(*shape)
    filter_handler = dictionaries.get_filter_handler(x, type)
    projector = ot_sparse_projection.OtFiltering(filter_handler, GAMMA, K)
    check_gradient_for_projection(projector, x)
    projector = ot_sparse_projection.OtSparseProjectionInvertible(filter_handler, GAMMA, LAMB)
    check_gradient_for_projection(projector, x)
    if orthonormal:
        projector = ot_sparse_projection.OtSparseProjectionOrthogonal(filter_handler, GAMMA, LAMB)
        check_gradient_for_projection(projector, x)


def test_orthonormal_wavelets():
    check_gradient((n, n), 'db1', orthonormal=True)
    check_gradient((n, n), 'db2', orthonormal=True)


def test_biorthonormal_wavelets():
    check_gradient((n, n), 'bior4.4')
    check_gradient((n, n), 'bior1.3')


def test_fft():
    check_gradient((n, n), 'dct', orthonormal=True)
