from numpy import load, savez
import numpy as np
import tensorflow as tf
# TODO: docs

class MattingLaplacian(tf.linalg.LinearOperator):
    r"""
        Matting laplacian operator as defined in:
        Fast matting using large kernel matting Laplacian matrices, He et al.
    """
    def __init__(self, image, epsilon=1e-5, window_radius=1):
        r"""
            Initialize the matting laplacian

            Parameters
            ----------
            image: tf.Tensor(shape=(H,W,C), dtype=tf.float64)
                content image
            epsilon: float
                regulariration term
            window_radius: int
                size of the window for means and covariance matrices
        """
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
        self.image = self.add_border(tf.expand_dims(image, -1))

        # compute integral images
        iimg = self.integral_image(self.image)
        prod_image = self.image @ self._transpose(self.image)
        prod_iimg = self.integral_image(prod_image)

        H, W, C = self.size
        n = self.window_area = (2*self.radius+1)**2
        # windows_sizes = self.add_border(tf.ones((H,W,1,1)), mode='CONSTANT')
        # n = self.window_area = self.compute_sums(self.integral_image(windows_sizes))

        # compute stats
        sums = self.compute_sums(iimg)
        sigma = self.compute_covs(prod_iimg, sums)
        self.means = sums/n
        self.delta_inv = tf.linalg.inv(sigma + epsilon/n * tf.eye(C, batch_shape=(H,W), dtype=image.dtype))

    # LinearOperator methods

    def _shape(self):
        H, W, _ = self.size
        return tf.TensorShape((H*W,H*W))

    def _shape_tensor(self):
        H, W, _ = self.size
        return tf.constant((H*W,H*W), dtype=tf.int32)

    def _matmul2(self, x, adjoint=False, adjoint_arg=False):
        def pprint(a):
            print(np.squeeze(a))
        H, W, C = self.size
        C1 = x.shape[-1]
        r = self.radius
        x = x.numpy().reshape((H,W,C1))
        image = self.crop_border(self.image).numpy()
        means = self.means.numpy()
        delta_inv = self.delta_inv.numpy()
        w = self.window_area

        out = np.zeros((H*W,C1))
        for c1 in range(C1):
            p = x[:,:,c1,np.newaxis]
            # print('pok')
            # pprint(p)
            p_tmp = np.zeros((H+2*r,W+2*r,1))
            p_tmp[r:H+r,r:W+r] = p
            # print('p_tmpok')
            # pprint(p_tmp)
            p_mean = np.zeros((H,W,1))
            for i in range(H):
                for j in range(W):
                    p_mean[i,j] = np.mean(p_tmp[i:i+2*r+1,j:j+2*r+1], axis=(0,1))
            # print('p_meanok')
            # pprint(p_mean)
            # compute Ip
            Ip = np.zeros((H+2*r,W+2*r,C,1))
            for i in range(H):
                for j in range(W):
                    Ip[i+r,j+r] = image[i,j]*p[i,j]
            # print('Ipok')
            # pprint(Ip)
            Ip_mean = np.zeros((H,W,C,1))
            for i in range(H):
                for j in range(W):
                    Ip_mean[i,j] = np.mean(Ip[i:i+2*r+1,j:j+2*r+1], axis=(0,1))
            # print('Ip_meanok')
            # pprint(Ip_mean)
            # compute a_star
            a = np.zeros((H,W,C,1))
            for i in range(H):
                for j in range(W):
                    a[i,j] = delta_inv[i,j] @ (Ip_mean[i,j] - means[i,j]*p_mean[i,j])
            # print('delta_inv')
            # pprint(delta_inv)
            # print('means')
            # pprint(means)
            # print('aok e-4')
            # pprint(a)
            b = np.zeros((H,W,1))
            for i in range(H):
                for j in range(W):
                    b[i,j] = p_mean[i,j] - a[i,j].T @ means[i,j]
            # print('bok')
            # pprint(b)
            a_tmp = np.zeros((H+2*r,W+2*r,C,1))
            a_tmp[r:H+r,r:W+r] = a
            b_tmp = np.zeros((H+2*r,W+2*r,1))
            b_tmp[r:H+r,r:W+r] = b
            # print('ab_tmp')
            # pprint(a_tmp)
            # pprint(b_tmp)
            a_sum = np.zeros((H,W,C,1))
            b_sum = np.zeros((H,W,1))
            for i in range(H):
                for j in range(W):
                    a_sum[i,j] = np.sum(a_tmp[i:i+2*r+1,j:j+2*r+1], axis=(0,1))
                    b_sum[i,j] = np.sum(b_tmp[i:i+2*r+1,j:j+2*r+1], axis=(0,1))
            # print('ab_sumok')
            # pprint(a_sum)
            # pprint(b_sum)
            q = np.zeros((H,W))
            for i in range(H):
                for j in range(W):
                    q[i,j] = w*p[i,j] - (a_sum[i,j].T @ image[i,j] + b_sum[i,j])
            # print('qok')
            # pprint(q)
            out[:,c1] = q.reshape(H*W)

        return tf.constant(out, dtype=self.image.dtype)

    @tf.function
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        # def pprint(a,b):
        #     print(a)
        #     tf.print(tf.squeeze(b))
        H, W, C = self.size
        p = self.add_border(tf.reshape(x, (H,W,-1,1)))
        p_mean = self.compute_sums(self.integral_image(p), normalize=True) # (H,W,C',1)
        # pprint('p_mean', p_mean)

        ip = self.image @ self._transpose(p) # (H+2r,W+2r,C,C'), ip[i,j] = I[i,j]*p[0]|...|I[i,j]*p[C'-1]
        # pprint('ip', ip)
        ip_mean = self.compute_sums(self.integral_image(ip), normalize=True) # (H,W,C,C')
        # pprint('ip_mean', ip_mean)
        a_star = self.delta_inv @ (ip_mean - self.means @ self._transpose(p_mean)) # (H,W,C,C'), a_star[i,j,:,c] = a*_{i,j} wrt channel c
        b_star = p_mean - self._transpose(a_star) @ self.means # (H,W,C',1)
        # tf.print(a_star.shape, b_star.shape, self.image.shape)
        a_star_sum = self.compute_sums(self.integral_image(self.add_border(a_star)))
        b_star_sum = self.compute_sums(self.integral_image(self.add_border(b_star)))

        q = self.window_area*self.crop_border(p) - \
             (self._transpose(a_star_sum) @ self.crop_border(self.image) + b_star_sum)

        # tmp = p_bar + self._transpose(a_star) @ (self.image - self.mu)
        # tmp_mean = self.windows_stats(self.integral_image(tmp), batch_shape=(H,W))[0]
        # q2 = self.window_size*p - tmp_mean
        # print('q')
        # tf.print(tf.squeeze(q1))
        # tf.print(tf.squeeze(q2))
        return tf.reshape(q, (H*W,-1))


    # add and crop borders to images to compute means over full windows
    @tf.function
    def add_border(self, img, mode='SYMMETRIC'):
        r"""
            Add a border of width r around the image

            Parameters
            ----------
            img: tf.Tensor(shape=(H,W,...), dtype=tf.float64)

            Returns
            -------
            tf.Tensor(shape=(H+2*r+1,W+2*r+1,...), dtype=tf.float64)
        """
        r = self.radius
        # paddings = 2*[[r+1,r]] + (int(tf.rank(img))-2)*[[0,0]]
        # tf.print(paddings)
        # return tf.pad(img, tf.constant(paddings), mode=mode)
        return tf.pad(img, tf.constant([[r+1,r],[r+1,r],[0,0],[0,0]]), mode=mode)

    def crop_border(self, img):
        H, W, _ = self.size
        r = self.radius
        return img[r+1:H+r+1,r+1:W+r+1]

    # stats
    def compute_sums(self, iimg, normalize=False):
        H, W, _ = self.size
        r = self.radius
        r0, c0, r1, c1 = 2*r+1, 2*r+1, H, W
        sums = iimg[r0:,c0:] + iimg[:r1,:c1] - iimg[r0:,:c1] - iimg[:r1,c0:]
        # print('>>>>')
        # tf.print(iimg.shape)
        # tf.print(tf.squeeze(iimg))
        # print(r0, c0, r1, c1)
        # tf.print(tf.squeeze(sums))
        # print('<<<<')

        return sums/self.window_area if normalize else sums

    def compute_covs(self, Q, sums):
        H, W, _ = self.size
        r = self.radius
        n = self.window_area

        r0, c0, r1, c1 = 2*r+1, 2*r+1, H, W
        return (Q[r0:,c0:] + Q[:r1,:c1] - Q[r0:,:c1] - Q[:r1,c0:] - (sums @ self._transpose(sums))/n)/n

    @staticmethod
    def _transpose(tensor):
        return tf.transpose(tensor, perm=(0,1,3,2))

    @staticmethod
    def integral_image(img):
        # type: tf.Tensor -> tf.Tensor
        r"""
            Compute the integral image S, i.e. the array such that
            $\forall i, j, S[i,j] = \sum_{i'<=i, j'<=j} img[i',j']$

            Parameters
            ----------
            img: tf.Tensor(shape=(H,W,...), dtype=self.dtype)
                input image

            Returns
            -------
            tf.Tensor(shape=(H,W,...), dtype=self.dtype)
                integral image
        """
        iimg = tf.identity(img)
        for i in (0,1):
            iimg = tf.cumsum(iimg, axis=i)
        return iimg


if __name__ == '__main__':
    H, W, C = 4, 4, 3
    image = tf.random.uniform((H,W,C))
    m = MattingLaplacian(image)
    w = m.add_border(tf.ones((H,W,1,1)), mode='CONSTANT')
    tf.print(tf.squeeze(m.compute_sums(m.integral_image(w))))
