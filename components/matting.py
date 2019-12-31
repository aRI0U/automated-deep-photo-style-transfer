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
    def __init__(self, image, epsilon=1e-5, window_radius=1):
        # params
        self.eps = epsilon
        self.radius = window_radius

        self.shape = image.shape
        self.image = tf.expand_dims(image, 3)
        self.prod_image = self.image @ tf.transpose(self.image, perm=(0,1,3,2))

        # compute integral images
        self.iimg = self.integral_image(self.image, axis=[0,1])
        self.prod_iimg = self.integral_image(self.prod_image, axis=[0,1])

        # list of indices of picture
        H, W, C = self.shape
        idx = tf.range(H*W)
        self.indices = tf.stack((idx//W, idx%W), axis=1)

        # compute stats
        mu, n, sigma = self.windows_stats(self.iimg, self.prod_iimg, self.indices, self.radius, batch_shape=(H,W))
        delta = sigma + epsilon/n * tf.eye(C, batch_shape=(H,W))
        self.delta_inv = tf.linalg.inv(delta)
        self.mu = mu/n
        self.window_size = n


    def L_operator(self, p):
        r"""
            Operator L such that L(p) = Mp where M denotes the matting laplacian matrix

            Parameters
            ----------
            p: tf.Tensor(shape=???, dtype=tf.float32)
                input vector

            Returns
            -------
            tf.Tensor
                matting laplacian matrix * p
        """
        H, W, C = self.shape

        ip_iimg = self.integral_image(self.image * p, axis=[0,1])
        ip_mean = self.windows_stats(ip_iimg, None, self.indices, self.radius, batch_shape=(H,W))

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

        Lp = self.window_size*p - (tf.transpose(a_star_sum, perm=(0,1,3,2)) @ image + b_star_sum)

        return self._flatten(Lp)


    def __call__(v):
        return tf.map_fn(self.L_operator, v)


    @tf.function
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
            iimg: tf.Tensor(shape=(H,W,C,1), dtype='a)
                integral image
            prod_iimg: tf.Tensor(shape=(H,W,C,C), dtype='a)
                integral image of product channels
            centers: tf.Tensor(shape=(...,2), dtype=int32)
                coordinates of the centers of the windows
            radius: int or tf.Tensor
                radius of the window. Must have shape of center or shape 1.
            batch_shape: tuple
                shape of the output tensors

            Returns
            -------
            mu: tf.Tensor(shape=(...,C,1), dtype='a)
                sum of the image pixels in the window
            n: tf.Tensor(shape=(...,1,1), dtype='a)
                size of the window
            sigma: tf.Tensor(shape=(...,C,C), dtype='a)
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
            $\forall i, j, S[i,j] = \sum_{i'<=i, j'<j} img[i',j']$

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


if __name__ == '__main__':
    # tests
    import tensorflow_probability as tfp
    H, W, C = 24, 19, 3
    radius = 1
    image = tf.random.uniform((H, W, C))
    matting = MattingLaplacian(image, epsilon=1e-2, window_radius=radius)

    mu, n, sigma = matting.windows_stats(matting.image, matting.prod_image, matting.indices, matting.radius, batch_shape=(H,W))

    tf.print(mu[0, 2])
    v1 = image[0:radius+1,0:2+radius+1]
    tf.print(tf.reduce_sum(v1, axis=[0,1]))

    tf.print(mu[16,12])
    v2 = image[16-radius:16+radius+1,12-radius:12+radius+1]
    tf.print(tf.reduce_sum(v2))


    print('\n\n')
    v3 = image[0:radius+1,:radius+1]
    tf.print(v3)
    tf.print(matting.iimg[:2,:2])
    tf.print(mu[0,0])
    tf.print(tf.reduce_sum(v3, axis=(0,1)))
    print('\n\n')


    tf.print(sigma[0,2])
    tf.print(tfp.stats.covariance(tf.reshape(v1, (-1,C)), tf.reshape(v1, (-1,C))))
    tf.print(sigma[16,12])
    tf.print(tfp.stats.covariance(tf.reshape(v2, (-1,C)), tf.reshape(v2, (-1,C))))
