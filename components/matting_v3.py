"""
The major part of this code is taken from https://github.com/MarcoForte/closed-form-matting.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse

import tensorflow as tf



class MattingLaplacian(tf.linalg.LinearOperator):
    r"""
        Compute the matting laplacian matrix using method described in:
        Fast Matting Using Large Kernel Matting Laplacian Matrices, He et al.

        Parameters
        ----------
        image: tf.Tensor(shape=(H,W,C), dtype=tf.float32)
            input image
        epsilon: float
            regularization parameter
        window_radius: int
            radius of the window
    """
    def __init__(self, image, epsilon=1e-5, window_radius=1, fname=None):
        super(MattingLaplacian, self).__init__(
            image.dtype,
            graph_parents=[image],
            is_self_adjoint=True,
            is_positive_definite=True,
            name='MattingLaplacian'
        )
        self.size = image.shape
        self.laplacian = tf.cast(
            self.compute_laplacian(image.numpy(), eps=epsilon, win_rad=window_radius),
            image.dtype
        )

    # LinearOperator methods
    def _shape(self):
        H, W, _ = self.size
        return tf.TensorShape((H*W,H*W))

    def _shape_tensor(self):
        H, W, _ = self.size
        return tf.constant((H*W,H*W), dtype=tf.int32)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return tf.sparse.sparse_dense_matmul(self.laplacian, x)

    @staticmethod
    def _rolling_block(A, block=(3, 3)):
        """Applies sliding window to given matrix."""
        shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
        strides = (A.strides[0], A.strides[1]) + A.strides
        return as_strided(A, shape=shape, strides=strides)


    def compute_laplacian(self, img, eps=1e-5, win_rad=1):
        """Computes Matting Laplacian for a given image.
        Args:
            img: 3-dim numpy matrix with input image
            mask: mask of pixels for which Laplacian will be computed.
                If not set Laplacian will be computed for all pixels.
            eps: regularization parameter controlling alpha smoothness
                from Eq. 12 of the original paper. Defaults to 1e-7.
            win_rad: radius of window used to build Matting Laplacian (i.e.
                radius of omega_k in Eq. 12).
        Returns: sparse matrix holding Matting Laplacian.
        """

        win_size = (win_rad * 2 + 1) ** 2
        h, w, d = img.shape
        # Number of window centre indices in h, w axes
        c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
        win_diam = win_rad * 2 + 1

        indsM = np.arange(h * w).reshape((h, w))
        ravelImg = img.reshape(h * w, d)
        win_inds = self._rolling_block(indsM, block=(win_diam, win_diam))

        win_inds = win_inds.reshape(c_h, c_w, win_size)

        win_inds = win_inds.reshape(-1, win_size)
        winI = ravelImg[win_inds]

        win_mu = np.mean(winI, axis=1, keepdims=True)
        win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

        inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(d))

        X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
        vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

        nz_indsCol = np.tile(win_inds, win_size).ravel()
        nz_indsRow = np.repeat(win_inds, win_size).ravel()
        nz_indsVal = vals.ravel()
        L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
        indices = np.mat([L.row, L.col]).transpose()
        return tf.SparseTensor(indices, L.data, L.shape)



if __name__ == '__main__':
    import time
    H, W, C = 200, 200, 3

    image = tf.random.uniform((H,W,C), dtype=tf.float64)
    t1 = time.time()
    matting = MattingLaplacian(image)
    t2 = time.time()
    p = tf.random.uniform((H*W,C), dtype=image.dtype)
    t3 = time.time()
    loss = tf.reduce_sum(p * matting.matmul(p))
    t4 = time.time()
    tf.print(loss)
    print(t4-t3, t2-t1)
