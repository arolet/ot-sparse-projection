import numpy as np

from ot_sparse_projection import ot_ground_metric


def test_convolution_handler():
    assert_same_dot_result((64, 64), 1)
    assert_same_dot_result((64, 64), .1)
    assert_same_dot_result((64, 64), 10)
    assert_same_dot_result((32, 32, 2), 1)
    assert_same_dot_result((32, 32, 2), .1)
    assert_same_dot_result((32, 32, 2), 10)


def assert_same_dot_result(shape, gamma):
    x = np.random.rand(*shape).reshape(-1, 1)
    y = ot_ground_metric.get_handler(ot_ground_metric.CONVOLUTION, shape, gamma).dot(x)
    yy = ot_ground_metric.get_handler(ot_ground_metric.MATRIX, shape, gamma).dot(x)
    np.testing.assert_almost_equal(y, yy)
