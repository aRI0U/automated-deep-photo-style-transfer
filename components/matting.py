from numpy import savez, load
import os

import tensorflow as tf

class MattingLaplacian:
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
        # params
        # self.eps = epsilon
        self.radius = window_radius

        self.shape = image.shape
        self.image = tf.expand_dims(image, 3)
        prod_image = self.image @ tf.transpose(self.image, perm=(0,1,3,2))

        # list of indices of picture
        H, W, C = self.shape
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
            iimg = self.integral_image(self.image, axis=[0,1])
            prod_iimg = self.integral_image(prod_image, axis=[0,1])

            # compute stats
            mu, n, sigma = self.windows_stats(iimg, prod_iimg, self.indices, self.radius, batch_shape=(H,W))
            delta = sigma + epsilon/n * tf.eye(C, batch_shape=(H,W))
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
        H, W, C = self.shape
        # TODO: see if reshape could be done earlier
        p = tf.expand_dims(p, -1)
        # assert self.image.shape == p.shape, str(self.image.shape) + str(p.shape)

        ip_iimg = self.integral_image(self.image * p, axis=[0,1])
        ip_mean = self.windows_stats(ip_iimg, None, self.indices, self.radius, batch_shape=(H,W))[0]
        ip_mean /= self.window_size
        p_bar = self.windows_stats(self.integral_image(p, axis=[0,1]), None, self.indices, self.radius, batch_shape=(H,W))[0]
        p_bar /= self.window_size

        a_star = self.delta_inv @ (ip_mean - self.mu*p_bar)
        b_star = p_bar - tf.transpose(a_star, perm=(0,1,3,2)) @ self.mu

        a_star_sum = self.windows_stats(
            self.integral_image(a_star, axis=[0,1]),
            None, self.indices, self.radius, batch_shape=(H,W)
        )[0]
        b_star_sum = self.windows_stats(
            self.integral_image(b_star, axis=[0,1]),
            None, self.indices, self.radius, batch_shape=(H,W)
        )[0]

        Lp = self.window_size*p - (tf.transpose(a_star_sum, perm=(0,1,3,2)) @ self.image + b_star_sum)
        return tf.squeeze(Lp, -1)

    # TODO: parallel
    def __call__(self, v):
        return tf.map_fn(self.L_operator, v)


    def _window_stats(self, iimg, prod_iimg, center, radius):
        zero = tf.constant(0, dtype=iimg.dtype, shape=iimg[0,0].shape)
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
        )/(n-1))

        return tf.concat((mu, n, sigma), axis=0)

    def windows_stats(self, iimg, prod_iimg, centers, radius, batch_shape=(-1,)):
        r"""
            Integral image based fast construction of mean and covariance matrix.
            See https://www.merl.com/publications/docs/TR2006-043.pdf for details.

            Parameters
            ----------
            iimg: tf.Tensor(shape=(H,W,C,1), dtype=tf.float32)
                integral image
            prod_iimg: tf.Tensor(shape=(H,W,C,C), dtype=tf.float32)
                integral image of product channels
            centers: tf.Tensor(shape=(...,2), dtype=int32)
                coordinates of the centers of the windows
            radius: int or tf.Tensor
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
    def integral_image(img, axis=None):
        # type: tf.Tensor -> tf.Tensor
        r"""
            Compute the integral image S, i.e. the array such that
            $\forall i, j, S[i,j] = \sum_{i'<=i, j'<=j} img[i',j']$

            Parameters
            ----------
            img: tf.Tensor(shape=(H,W,...), dtype=tf.float32)
                input image
            axis: iterable
                axis along cumulative sum is done (choose (0,1) for a classical integral image)

            Returns
            -------
            tf.Tensor(shape=(H,W,...), dtype=tf.float32)
                integral image
        """
        iimg = tf.identity(img)
        axis = range(img.ndim) if axis is None else axis
        for i in axis:
            iimg = tf.cumsum(iimg, axis=i)
        return iimg

    @staticmethod
    def _flatten(tensor):
        return tf.reshape(tensor, (-1,))

    @staticmethod
    @tf.function
    def parallel_map(*args, **kwargs):
        return tf.map_fn(*args, **kwargs)







if __name__ == '__main__':
    import unittest
    import tensorflow_probability as tfp

    class TestStatsMethods(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super(TestStatsMethods, self).__init__(*args, **kwargs)
            self.H, self.W, self.C = 24, 19, 3
            self.r = 5
            self.image = tf.random.uniform((self.H, self.W, self.C))

            matting = MattingLaplacian(self.image, epsilon=1e-2, window_radius=self.r)
            self.mu, self.n, self.sigma = matting.windows_stats(matting.iimg, matting.prod_iimg, matting.indices, self.r, batch_shape=(self.H, self.W))

            self.tolerance = 1e-5

        def assertEq(self, t1, t2):
            msg = 'pred: {}\ngt: {}\nerror={}'.format(str(t1.numpy()), str(t2.numpy()), tf.reduce_sum(tf.abs(t1-t2)).numpy())
            self.assertTrue(tf.reduce_sum(tf.square(t1-t2)) < self.tolerance, msg=msg)

        def test_sum_topleft_corner(self):
            v = self.image[0:self.r+1, 0:self.r+1]
            pred = self.mu[0,0]
            gt = self.compute_sum(v)
            self.assertEq(pred, gt)

        def test_sum_left_border(self):
            v = self.image[0:self.r+1, 0:2+self.r+1]
            pred = self.mu[0,2]
            gt = self.compute_sum(v)
            self.assertEq(pred, gt)

        def test_sum_bottomright_corner(self):
            v = self.image[-self.r-1:, -self.r-1:]
            pred = self.mu[-1,-1]
            gt = self.compute_sum(v)
            self.assertEq(pred, gt)

        def test_sum_center(self):
            v = self.image[16-self.r:16+self.r+1, 12-self.r:12+self.r+1]
            pred = self.mu[16,12]
            gt = self.compute_sum(v)
            self.assertEq(pred, gt)

        def test_cov_topleft_corner(self):
            v = self.image[0:self.r+1, 0:self.r+1]
            pred = self.sigma[0,0]
            gt = self.compute_cov(v)
            self.assertEq(pred, gt)

        def test_cov_left_border(self):
            v = self.image[0:self.r+1, 0:2+self.r+1]
            pred = self.sigma[0,2]
            gt = self.compute_cov(v)
            self.assertEq(pred, gt)

        def test_cov_bottomright_corner(self):
            v = self.image[-self.r-1:, -self.r-1:]
            pred = self.sigma[-1,-1]
            gt = self.compute_cov(v)
            self.assertEq(pred, gt)

        def test_cov_center(self):
            v = self.image[16-self.r:16+self.r+1, 12-self.r:12+self.r+1]
            pred = self.sigma[16,12]
            gt = self.compute_cov(v)
            self.assertEq(pred, gt)

        @staticmethod
        def compute_sum(tensor):
            return tf.expand_dims(tf.reduce_sum(tensor, axis=(0,1)), 1)

        @staticmethod
        def compute_cov(tensor):
            return tfp.stats.covariance(tf.reshape(tensor, (-1, 3)), tf.reshape(tensor, (-1,3)))

    unittest.main()
