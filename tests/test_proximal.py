import numpy as np

from ot_sparse_projection import proximal


def test_l_inf_proximal():
    check_l_inf_proximal(100, 1)
    check_l_inf_proximal(100, .1)
    check_l_inf_proximal(100, 29)
    check_l_inf_proximal(100, np.arange(1, 101))


def check_l_inf_proximal(n, l):
    x = np.random.rand(n)
    y = proximal.prox_l_inf_ball(x, l) + l * proximal.prox_l1(x / l, 1)
    np.testing.assert_array_almost_equal(x, y)
