from ot_sparse_projection.matrix import MatrixHandler
import pywt
import scipy.fftpack as fp
import numpy as np


class Transform(MatrixHandler):
    """An abstract adapter class for Fourier and Wavelet transforms, acting on 2d single channel images.

    Note:
        In general, you should not instantiate Transform or its subclasses yourself. Instead, use
        get_filter_handler which will initiate all parameters for you.
    """

    def __init__(self, shape):
        """
        Args:
            shape (str): the shape of images this transform will be applied to.

        """
        self._shape = shape
        self._shape_wav = shape

    def cols(self):
        return np.prod(self._shape_wav)

    def rows(self):
        return np.prod(self._shape)

    def reshape_coeffs(self, x):
        """Reshape the transform coefficients to a matrix shape so that they can be passed to dot

        Args:
            x: the transform coefficients

        Returns:
            the number of columns of this matrix
        """
        return x.reshape(self._shape)

    def inverse_dot(self, x):
        """Applies the transform to the image x

        Args:
            x: the image to transform

        Returns:
            the coefficients of the transform of x
        """
        raise NotImplementedError()

    def inverse_transpose_dot(self, x):
        """Applies the transpose of the transform to the coefficients x

        Note:
            In the orthonormal case, this is dot(.)

        Args:
            x: the coefficients

        Returns:
            the result of the transpose of this transform applied to x
        """
        raise NotImplementedError()

    def high_pass_filter(self, x, k):
        """Applies a Euclidean pass filter on the top k*k coefficients of the transform of x

        Args:
            x: the image to be filtered
            k: the size of the filter

        Returns:
            the filtered image with all but the k*k top coefficients removed
        """
        tmp = self.reshape_coeffs(self.inverse_dot(x))
        tmp[:k, :k] = 0
        return self.dot(tmp.ravel()), tmp

    def low_pass_filter(self, x, k):
        """Applies a Euclidean pass filter on the bottom k*k coefficients of the transform of x

        Args:
            x: the image to be filtered
            k: the size of the filter

        Returns:
            the filtered image with all but the k*k bottom coefficients removed
        """
        tmp = self.reshape_coeffs(self.inverse_dot(x))
        tmp[k:] = 0
        tmp[:, k:] = 0
        return self.dot(tmp.ravel()), tmp

    def pass_filter(self, x, pattern):
        """Applies a Euclidean pass filter on the coefficients of the transform of x

        Args:
            x: the image to be filtered
            pattern: a mask matrix represented the coefficients to be kept

        Returns:
            the filtered image with all the masked coefficients removed
        """
        tmp = self.reshape_coeffs(self.inverse_dot(x))
        tmp[pattern] = 0
        return self.dot(tmp.ravel()), tmp

    def block_filter(self, x, pattern):
        """Applies a Euclidean pass filter on the coefficients of the transform of x

        Note:
            This is a convenience method for pass_filter(x, np.logical_not(pattern))

        Args:
            x: the image to be filtered
            pattern: a mask matrix represented the coefficients to be removed

        Returns:
            the filtered image with all but the masked coefficients removed
        """
        tmp = self.reshape_coeffs(self.inverse_dot(x))
        tmp[np.logical_not(pattern)] = 0
        return self.dot(tmp.ravel()), tmp

    def assert_orthonormal(self):
        """An assertion which fails if this transform is not represented by an orthonormal matrix

        In general, transforms are not orthonormal, the default is thus to fail unless specifically defined otherwise
        """
        raise AssertionError("Dictionary is not orthonormal")


def im2freq(data):
    """ Applies a discrete Fourier transform

    Args:
        f: the 2d image to be transformed
    Returns:
        the Fourier coefficients, as a matrix of the same size as f
    """
    return fp.dctn(data, norm='ortho')


def freq2im(f):
    """ Applies an inverse discrete Fourier transform

    Args:
        f: the coefficients of an image
    Returns:
        the image which Fourier coefficients correspond to f
    """
    return fp.idctn(f, norm='ortho')


class DctHandler(Transform):
    """A class representing a discrete Fourier transform, acting on 2d single channel images.

    Note:
        In general, you should not instantiate Transform or its subclasses yourself. Instead, use
        get_filter_handler which will initiate all parameters for you.
    """

    def __init__(self, shape):
        """
        Args:
            shape (str): the shape of images this transform will be applied to.

        """
        Transform.__init__(self, shape)

    def transpose_dot(self, x):
        return im2freq(x.reshape(self._shape)).ravel()

    def dot(self, x):
        return freq2im(x.reshape(self._shape)).ravel()

    def inverse_dot(self, x):
        return self.transpose_dot(x)

    def inverse_transpose_dot(self, x):
        return self.dot(x)

    def assert_orthonormal(self):
        """This transform is orthonormal, pass"""
        pass


class WaveletHandler(Transform):
    """A class representing a wavelet transform, acting on 2d single channel images.

    Note:
        In general, you should not instantiate Transform or its subclasses yourself. Instead, use
        get_filter_handler which will initiate all parameters for you.
    """

    def __init__(self, shape, type='db1'):
        Transform.__init__(self, shape)
        self.mode = 'periodization'
        self.type = type
        coeffs, self.slices = pywt.coeffs_to_array(pywt.wavedec2(np.ones(shape), self.type, mode=self.mode))
        self._shape_wav = coeffs.shape

    def dec(self, x, level=None):
        return pywt.wavedec2(x.reshape(self._shape), self.type, mode=self.mode, level=level)

    def rec(self, x):
        return pywt.waverec2(x, self.type, mode=self.mode).ravel()

    def dot(self, x):
        return self.rec(self.array_to_coeffs(x))

    def array_to_coeffs(self, x):
        return pywt.array_to_coeffs(x.reshape(self._shape_wav), self.slices, output_format='wavedec2')

    def coeffs_to_array(self, x):
        return pywt.coeffs_to_array(x)[0].ravel()

    def inverse_dot(self, x):
        return pywt.coeffs_to_array(self.dec(x))[0].ravel()

    def reshape_coeffs(self, x):
        return x.reshape(self._shape_wav)


class OrthogonalWaveletHandler(WaveletHandler):

    def transpose_dot(self, x):
        return self.inverse_dot(x)

    def inverse_transpose_dot(self, x):
        return self.dot(x)

    def assert_orthonormal(self):
        pass


class BiorthogonalWaveletHandler(WaveletHandler):

    def __init__(self, shape, type='bior4.4'):
        super(BiorthogonalWaveletHandler, self).__init__(shape, type)
        self._dual_type = 'rbio' + type[-3:]

    def transpose_dot(self, x):
        return pywt.coeffs_to_array(pywt.wavedec2(x.reshape(self._shape_wav), self._dual_type, mode=self.mode))[
            0].ravel()

    def inverse_transpose_dot(self, x):
        return pywt.waverec2(pywt.array_to_coeffs(x.reshape(self._shape), self.slices, output_format='wavedec2'),
                             self._dual_type, mode=self.mode).ravel()


def get_filter_handler(im, filter_type):
    """ factory function to get a basis handler object based on the image shape and the type of basis required

    Args:
        im (numpy.array): an image
        filter_type (string): 'dct' for a 2-d discrete Fourier transform, otherwise a wavelet type compatible with pyWavelet
    """
    if filter_type == 'dct':
        return DctHandler(im.shape)
    wav = pywt.Wavelet(filter_type)
    if wav.orthogonal:
        return OrthogonalWaveletHandler(im.shape, type=filter_type)
    if filter_type[:4] == 'bior':
        return BiorthogonalWaveletHandler(im.shape, type=filter_type)
    raise ValueError('Unkown decomposition basis')
