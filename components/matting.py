from numpy import savez, load
import os

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
        # params
        # self.eps = epsilon
        self.radius = window_radius

        self.size = image.shape
        self.image = tf.expand_dims(image, 3)
        prod_image = self.image @ self._transpose(self.image)

        # list of indices of picture
        H, W, C = self.size
        idx = tf.range(H*W)
        self.indices = tf.stack((idx//W, idx%W), axis=1)

        if fname is not None and os.path.isfile(fname):
            # load stats from files
            npzfile = load(fname, mmap_mode='r')
            self.mu = tf.constant(npzfile['mu'], dtype=image.dtype)
            self.window_size = tf.constant(npzfile['window_size'], dtype=image.dtype)
            self.delta_inv = tf.constant(npzfile['delta_inv'], dtype=image.dtype)

        else:
            # compute integral images
            iimg = self.integral_image(self.image)
            prod_iimg = self.integral_image(prod_image)

            # compute stats
            mu, n, sigma = self.windows_stats(iimg, prod_iimg, self.indices, self.radius, batch_shape=(H,W))
            delta = sigma + epsilon/n * tf.eye(C, dtype=image.dtype, batch_shape=(H,W))
            self.delta_inv = tf.linalg.inv(delta)
            self.mu = mu/n
            self.window_size = n
            if fname:
                savez(fname,
                    mu=self.mu.numpy(),
                    window_size=self.window_size.numpy(),
                    delta_inv=self.delta_inv.numpy()
                )

    @tf.function
    def L_operator(self, p):
        r"""
            Operator L such that L(p) = Mp where M denotes the matting laplacian matrix

            Parameters
            ----------
            p: tf.Tensor(shape=(H,W,1), dtype=tf.float32)
                input vector

            Returns
            -------
            tf.Tensor(shape=(H,W,1), dtype=tf.float32)
                matting laplacian matrix * p
        """
        H, W, C = self.size
        # TODO: see if reshape could be done earlier
        p = tf.expand_dims(p, -1)
        # assert self.image.shape == p.shape, str(self.image.shape) + str(p.shape)

        ip_iimg = self.integral_image(self.image * p)
        ip_mean = self.windows_stats(ip_iimg, None, self.indices, self.radius, batch_shape=(H,W))[0]
        ip_mean /= self.window_size
        p_bar = self.windows_stats(self.integral_image(p), None, self.indices, self.radius, batch_shape=(H,W))[0]
        p_bar /= self.window_size

        a_star = self.delta_inv @ (ip_mean - self.mu*p_bar)
        b_star = p_bar - self._transpose(a_star) @ self.mu

        a_star_sum = self.windows_stats(
            self.integral_image(a_star),
            None, self.indices, self.radius, batch_shape=(H,W)
        )[0]
        b_star_sum = self.windows_stats(
            self.integral_image(b_star),
            None, self.indices, self.radius, batch_shape=(H,W)
        )[0]

        Lp = self.window_size*p - (self._transpose(a_star_sum) @ self.image + b_star_sum)
        return tf.squeeze(Lp, -1)

    # TODO: parallel
    def __call__(self, v):
        return tf.map_fn(self.L_operator, v)


    def _window_stats(self, iimg, prod_iimg, center, radius):
        zero = tf.zeros(iimg[0,0].shape, dtype=iimg.dtype)
        top_left = tf.maximum(center-radius-1, -1)
        xmin, ymin = top_left[0], top_left[1]
        bottom_right = tf.minimum(center+radius+1, iimg.shape[:2])
        xmax, ymax = bottom_right[0]-1, bottom_right[1]-1

        tl = iimg[xmin,ymin] if tf.minimum(xmin,ymin) >= 0 else zero
        bl = iimg[xmax,ymin] if tf.minimum(xmax,ymin) >= 0 else zero
        tr = iimg[xmin,ymax] if tf.minimum(xmin,ymax) >= 0 else zero
        br = iimg[xmax,ymax] if tf.minimum(xmax,ymax) >= 0 else zero
        n = self._flatten(tf.cast((xmax-xmin)*(ymax-ymin), iimg.dtype))
        mu = self._flatten(br + tl - bl - tr)
        if prod_iimg is None:
            return tf.concat((mu, n), axis=0)

        tl_prod = prod_iimg[xmin,ymin] if tf.minimum(xmin,ymin) >= 0 else zero
        bl_prod = prod_iimg[xmax,ymin] if tf.minimum(xmax,ymin) >= 0 else zero
        tr_prod = prod_iimg[xmin,ymax] if tf.minimum(xmin,ymax) >= 0 else zero
        br_prod = prod_iimg[xmax,ymax] if tf.minimum(xmax,ymax) >= 0 else zero
        # print(tl, bl, br, tr)
        # print((br_prod + tl_prod - bl_prod - tr_prod - (tf.expand_dims(mu, 1) @ tf.expand_dims(mu, 0))/n)/(n-1))
        sigma = self._flatten((
            br_prod + tl_prod - bl_prod - tr_prod \
            - (tf.expand_dims(mu, 1) @ tf.expand_dims(mu, 0))/n
        )/n)

        return tf.concat((mu, n, sigma), axis=0)

    def windows_stats(self, iimg, prod_iimg=None, centers=None, radius=None, batch_shape=(-1,)):
        r"""
            Integral image based fast construction of mean and covariance matrix.
            See https://www.merl.com/publications/docs/TR2006-043.pdf for details.

            Parameters
            ----------
            iimg: tf.Tensor(shape=(H,W,C,1), dtype=tf.float32)
                integral image
            prod_iimg: tf.Tensor(shape=(H,W,C,C), dtype=tf.float32)
                integral image of product channels
            centers: tf.Tensor(shape=(...,2), dtype=tf.int32)
                coordinates of the centers of the windows
            radius: int or tf.Tensor(dtype=tf.int32)
                radius of the window. Must have shape of center or shape 1.
            batch_shape: tuple
                shape of the output tensors

            Returns
            -------
            mu: tf.Tensor(shape=(...,C,1), dtype=tf.float32)
                sum of the image pixels in the window
            n: tf.Tensor(shape=(...,1,1), dtype=tf.float32)
                size of the window
            sigma: tf.Tensor(shape=(...,C,C), dtype=tf.float32)
                covariance matrix of the image pixels in the window
        """
        centers = self.indices if centers is None else centers
        radius = self.radius if radius is None else radius
        f = lambda c: self._window_stats(iimg, prod_iimg, c, radius)
        stats = tf.map_fn(f, centers, dtype=iimg.dtype)
        C = iimg.shape[2]

        if prod_iimg is None:
            return (
                tf.reshape(stats[...,:C], batch_shape+(C,1)),   # mu
                tf.reshape(stats[...,C], batch_shape+(1,1)),    # n
            )
        return (
            tf.reshape(stats[...,:C], batch_shape+(C,1)),   # mu
            tf.reshape(stats[...,C], batch_shape+(1,1)),    # n
            tf.reshape(stats[...,C+1:], batch_shape+(C,C))  # sigma
        )

    @staticmethod
    def integral_image(img):
        # type: tf.Tensor -> tf.Tensor
        r"""
            Compute the integral image S, i.e. the array such that
            $\forall i, j, S[i,j] = \sum_{i'<=i, j'<=j} img[i',j']$

            Parameters
            ----------
            img: tf.Tensor(shape=(H,W,...), dtype=tf.float32)
                input image

            Returns
            -------
            tf.Tensor(shape=(H,W,...), dtype=tf.float32)
                integral image
        """
        iimg = tf.identity(img)
        for i in (0,1):
            iimg = tf.cumsum(iimg, axis=i)
        return iimg

    # utils
    @staticmethod
    def _flatten(tensor):
        return tf.reshape(tensor, (-1,))

    @staticmethod
    def _transpose(tensor):
        return tf.transpose(tensor, perm=(0,1,3,2))

    @staticmethod
    @tf.function
    def parallel_map(*args, **kwargs):
        return tf.map_fn(*args, **kwargs)



    # LinearOperator methods
    def _shape(self):
        H, W, _ = self.size
        return tf.TensorShape((H*W,H*W))

    def _shape_tensor(self):
        H, W, _ = self.size
        return tf.constant((H*W,H*W), dtype=tf.int32)

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        H, W, C = self.size
        p = tf.reshape(x, (H,W,-1,1))
        p_bar = self.windows_stats(self.integral_image(p), batch_shape=(H,W))[0]
        p_bar /= self.window_size # (H,W,C',1)
        # print('p_bar')
        # tf.print(tf.squeeze(p_bar))

        ip = self.image @ self._transpose(p) # (H,W,C,C'), ip[i,j] = I[i,j]*p[0]|...|I[i,j]*p[C'-1]
        # print('ip')
        # tf.print(tf.squeeze(ip))
        ip_mean = self.windows_stats(self.integral_image(ip), batch_shape=(H,W))[0]
        ip_mean /= self.window_size
        # print('ip_mean')
        # tf.print(tf.squeeze(ip_mean))
        a_star = self.delta_inv @ (ip_mean - self.mu @ self._transpose(p_bar)) # (H,W,C,C'), a_star[i,j,:,c] = a*_{i,j} wrt channel c
        # print('a_star')
        # tf.print(tf.squeeze(a_star))
        b_star = p_bar - self._transpose(a_star) @ self.mu # (H,W,C',1)

        a_star_sum = self.windows_stats(self.integral_image(a_star), batch_shape=(H,W))[0]
        b_star_sum = self.windows_stats(self.integral_image(b_star), batch_shape=(H,W))[0]

        q1 = self.window_size*p - (self._transpose(a_star_sum) @ self.image + b_star_sum)

        tmp = p_bar + self._transpose(a_star) @ (self.image - self.mu)
        tmp_mean = self.windows_stats(self.integral_image(tmp), batch_shape=(H,W))[0]
        q2 = self.window_size*p - tmp_mean
        print('q')
        tf.print(tf.squeeze(q1))
        tf.print(tf.squeeze(q2))
        return tf.reshape(q2, (H*W,-1))

    def add_border(self, img):
        H, W, _ = self.size
        r = self.radius
        return tf.image.pad_to_bounding_box(img, r, r, H+2*r, W+2*r)

    def compute_sums(self, iimg):
        H, W, _ = self.size
        r = self.radius

        r0, c0, r1, c1 = 2*r, 2*r, H, W
        sums = iimg[r0:,c0:] + iimg[:r1,:c1] - iimg[r0:,:c1] - iimg[:r1,c0:]

        return sums

    def compute_covs(self, Q, sums):
        H, W, _ = self.size
        r = self.radius
        n = (2*r+1)**2

        r0, c0, r1, c1 = 2*r, 2*r, H, W
        tf.print(Q.shape, sums.shape)
        return (Q[r0:,c0:] + Q[:r1,:c1] - Q[r0:,:c1] - Q[:r1,c0:] - (sums @ self._transpose(sums))/n)/n


if __name__ == '__main__':
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    import scipy.ndimage
    import scipy.sparse
    import scipy.sparse.linalg
    import tensorflow as tf
    import time

    H, W, C = 5, 5, 3

    image = tf.random.uniform((H,W,C), dtype=tf.float32)
    # t1 = time.time()
    # L1 = compute_matting_laplacian(image.numpy())
    # t2 = time.time()
    matting = MattingLaplacian(image)
    bimage = tf.expand_dims(matting.add_border(image), -1)
    t1 = time.time()
    iimg = matting.integral_image(bimage)
    sums = matting.compute_sums(iimg)
    t2 = time.time()
    prod_image = bimage @ matting._transpose(bimage)
    prod_iimg = matting.integral_image(prod_image)
    covs = matting.compute_covs(prod_iimg, sums)
    tf.print(covs.shape)
    # tf.print(tf.squeeze(image))
    # tf.print(tf.squeeze(iimg))
    # tf.print(tf.squeeze(matting.compute_sums(iimg)))
    print(t2-t1)
