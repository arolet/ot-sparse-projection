import numpy as np

from ot_sparse_projection import dictionaries


def check_filter_handler(shape, type, orthonormal=False):
    x = np.random.normal(0, 1, shape)
    filter_handler = dictionaries.get_filter_handler(x, type)
    np.testing.assert_array_almost_equal(x.ravel(), filter_handler.dot(filter_handler.inverse_dot(x)).ravel())
    np.testing.assert_array_almost_equal(x.ravel(),
                                         filter_handler.inverse_transpose_dot(filter_handler.transpose_dot(x)).ravel())
    if orthonormal:
        np.testing.assert_array_almost_equal(x.ravel(), filter_handler.dot(
            filter_handler.transpose_dot(x)).ravel())


def check_filter_handler_2(shape, type, orthonormal=False):
    x = np.random.normal(0, 1, shape)
    filter_handler = dictionaries.get_filter_handler(x, type)
    n = np.prod(filter_handler._shape_wav)
    for i in range(n):
        x = np.zeros((n, 1))
        x[i] = 1
        np.testing.assert_array_almost_equal(x.ravel(), filter_handler.dot(filter_handler.inverse_dot(x)).ravel())
        np.testing.assert_array_almost_equal(x.ravel(), filter_handler.inverse_transpose_dot(
            filter_handler.transpose_dot(x)).ravel())
        if orthonormal:
            np.testing.assert_array_almost_equal(x.ravel(), filter_handler.dot(
                filter_handler.transpose_dot(x)).ravel())


def get_matrix(n, column_generator):
    m = np.zeros((n, n))
    for i in range(n):
        x = np.zeros((n, 1))
        x[i] = 1
        m[i, :] = column_generator(x).reshape(1, -1)
    return m


def check_filter_handler_3(shape, type, orthonormal=False):
    x = np.random.normal(0, 1, shape)
    filter_handler = dictionaries.get_filter_handler(x, type)
    n = np.prod(filter_handler._shape_wav)
    m = get_matrix(n, filter_handler.dot)
    m_inv = get_matrix(n, filter_handler.inverse_dot)
    m_T = get_matrix(n, filter_handler.transpose_dot)
    m_inv_T = get_matrix(n, filter_handler.inverse_transpose_dot)

    np.testing.assert_array_almost_equal(m, m_T.T)
    np.testing.assert_array_almost_equal(m_inv, m_inv_T.T)
    np.testing.assert_array_almost_equal(m_inv, np.linalg.inv(m))
    np.testing.assert_array_almost_equal(m_T, np.linalg.inv(m_inv_T))
    if orthonormal:
        np.testing.assert_array_almost_equal(m, m_inv_T)
        np.testing.assert_array_almost_equal(m_inv, m_T)


def test_orthonormal_wavelets():
    check_filter_handler((32, 32), 'db1', orthonormal=True)
    check_filter_handler((32, 32), 'db2', orthonormal=True)


def test_biorthonormal_wavelets():
    check_filter_handler((32, 32), 'bior4.4')
    check_filter_handler((32, 32), 'bior1.3')


def test_fft():
    check_filter_handler_3((32, 32), 'dct', orthonormal=True)
