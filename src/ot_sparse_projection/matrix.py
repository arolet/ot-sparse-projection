class MatrixHandler(object):
    """An adapter for things that can be represented as a matrix.

    In this package, we use it for the transport cost matrix, as well as for wavelet and fourier transforms.
        """

    def dot(self, x):
        """Right product of the matrix with a vector.

        Args:
            x: the vector to be multiplied.

        Returns:
            the product Mx of this matrix with x
        """
        raise NotImplementedError()

    def transpose_dot(self, x):
        """Right product of the transpose of this matrix with a vector.

        Args:
            x: the vector to be multiplied.

        Returns:
            the product M.Tx of the transpose of this matrix with x
        """
        raise NotImplementedError()

    def cols(self):
        """The number of columns of this matrix

        Returns:
            the number of columns of this matrix
        """
        raise NotImplementedError()

    def rows(self):
        """The number of rows of this matrix

        Returns:
            the number of rows of this matrix
        """
        raise NotImplementedError()
