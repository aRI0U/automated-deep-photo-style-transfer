import numpy as np

import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

import tensorflow as tf


def matting_laplacian(image, consts=None, epsilon=1e-5, window_radius=1):
    print("Compute matting laplacian started")

    num_window_pixels = (window_radius * 2 + 1) ** 2
    height, width, channels = image.shape
    if consts is None:
        consts = np.zeros(shape=(height, width))

    # compute erosion with window square as mask
    consts = scipy.ndimage.morphology.grey_erosion(consts, footprint=np.ones(
        shape=(window_radius * 2 + 1, window_radius * 2 + 1)))

    num_image_pixels = width * height

    # value and index buffers for laplacian in COO format
    laplacian_indices = []
    laplacian_values = []

    # cache pixel indices in a matrix
    pixels_indices = np.reshape(np.array(range(num_image_pixels)), newshape=(height, width), order='F')

    # iterate over image pixels
    for y in range(window_radius, width - window_radius):
        for x in range(window_radius, height - window_radius):
            if consts[x, y]:
                continue

            window_x_start, window_x_end = x - window_radius, x + window_radius + 1
            window_y_start, window_y_end = y - window_radius, y + window_radius + 1
            window_indices = pixels_indices[window_x_start:window_x_end, window_y_start:window_y_end].ravel()
            window_values = image[window_x_start:window_x_end, window_y_start:window_y_end, :]
            window_values = window_values.reshape((num_window_pixels, channels))

            mean = np.mean(window_values, axis=0).reshape(channels, 1)
            cov = np.matmul(window_values.T, window_values) / num_window_pixels - np.matmul(mean, mean.T)

            tmp0 = np.linalg.inv(cov + epsilon / num_window_pixels * np.identity(channels))

            tmp1 = window_values - np.repeat(mean.transpose(), num_window_pixels, 0)
            window_values = (1 + np.matmul(np.matmul(tmp1, tmp0), tmp1.T)) / num_window_pixels

            ind_mat = np.broadcast_to(window_indices, (num_window_pixels, num_window_pixels))

            laplacian_indices.extend(zip(ind_mat.ravel(order='F'), ind_mat.ravel(order='C')))
            laplacian_values.extend(window_values.ravel())

    # create sparse matrix in coo format
    laplacian_coo = scipy.sparse.coo_matrix((laplacian_values, zip(*laplacian_indices)),
                                            shape=(num_image_pixels, num_image_pixels))

    # compute final laplacian
    sum_a = laplacian_coo.sum(axis=1).T.tolist()[0]
    laplacian_coo = (scipy.sparse.diags([sum_a], [0], shape=(num_image_pixels, num_image_pixels)) - laplacian_coo) \
        .tocoo()

    # create a sparse tensor from the coo laplacian
    indices = np.mat([laplacian_coo.row, laplacian_coo.col]).transpose()
    laplacian_tf = tf.to_float(tf.SparseTensor(indices, laplacian_coo.data, laplacian_coo.shape))

    return lambda x: tf.sparse.sparse_dense_matmul(laplacian_tf, x)







def fast_matting_laplacian(image, epsilon=1e-5, window_radius=1):
    r"""
        Compute the matting laplacian matrix using method described in:
        Fast Matting Using Large Kernel Matting Laplacian Matrices, He et al.

        Parameters
        ----------
        image: tf.Tensor
            (H,W,C) image
        epsilon: float
            regularization parameter
        window_radius:
            radius of the window

        Returns
        -------
        tf.Tensor -> tf.Tensor
            operator M such that M(x) = Lx where L denotes the matting laplacian
    """
    # print("Compute matting laplacian started")
    def L(p, image=image, epsilon=epsilon, window_radius=window_radius):
        # type: tf.Tensor[H*W] -> tf.Tensor[H*W]
        # TODO: make clean tf2 st no need to add .as_list()
        H, W, C = image.shape
        image = tf.expand_dims(image, 3)
        p = tf.reshape(p, (H,W,1,1))

        # compute integral images
        iimg = integral_image(image, axis=[0,1])
        prod_img = image @ tf.transpose(image, perm=(0,1,3,2))
        prod_iimg = integral_image(prod_img, axis=[0,1])
        idx = tf.range(H*W)
        indices = tf.stack((idx//W, idx%W), axis=-1)

        # compute stats of image
        mu, n, sigma = windows_stats(iimg, prod_iimg, indices, window_radius, batch_shape=(H,W))
        mu /= n

        # compute other necessary quantities
        ip_iimg = integral_image(image * p, axis=[0,1])
        ip_mean = windows_stats(ip_iimg, None, indices, window_radius, batch_shape=(H,W))[0]/n

        p_bar = windows_stats(integral_image(p, axis=[0,1]), None, indices, window_radius, batch_shape=(H,W))[0]/n

        delta = sigma + epsilon/n * tf.eye(C, batch_shape=(H,W))
        delta_inv = tf.linalg.inv(delta)

        a_star = delta_inv @ (ip_mean - mu * p_bar)
        b_star = p_bar - tf.transpose(a_star, perm=(0,1,3,2)) @ mu

        a_star_sum, _ = windows_stats(
            integral_image(a_star, axis=[0,1]),
            None, indices, window_radius, batch_shape=(H,W)
        )
        b_star_sum, _ = windows_stats(
            integral_image(b_star, axis=[0,1]),
            None, indices, window_radius, batch_shape=(H,W)
        )

        Lp = n*p - (tf.transpose(a_star_sum, perm=(0,1,3,2)) @ image + b_star_sum)

        return _flatten(Lp)

    # return lambda v: tf.map_fn(L, v)

    # dummy
    return lambda x: x

#@tf.function
def integral_image(img, axis=None):
    # type: tf.Tensor -> tf.Tensor
    iimg = tf.identity(img)
    axis = range(img.ndim) if axis is None else axis
    for i in axis:
        iimg = tf.cumsum(iimg, axis=i)
    return iimg

def _flatten(tensor):
    return tf.reshape(tensor, (-1,))

#@tf.function
def _window_stats(iimg, prod_iimg, center, radius):
    # xmin, ymin = tf.math.maximum(center-radius-1, -1)
    zero = tf.constant(0, dtype=iimg.dtype, shape=iimg[0,0].shape)
    top_left = tf.maximum(center-radius-1, -1)
    xmin, ymin = top_left[0], top_left[1]
    bottom_right = tf.minimum(center+radius+1, iimg.shape[:2])
    xmax, ymax = bottom_right[0]-1, bottom_right[1]-1

    tl = iimg[xmin,ymin] if tf.minimum(xmin,ymin) >= 0 else zero
    bl = iimg[xmax,ymin] if tf.minimum(xmax,ymin) >= 0 else zero
    tr = iimg[xmin,ymax] if tf.minimum(xmin,ymax) >= 0 else zero
    br = iimg[xmax,ymax] if tf.minimum(xmax,ymax) >= 0 else zero
    n = _flatten(tf.cast((xmax-xmin)*(ymax-ymin), iimg.dtype))
    mu = _flatten(br + tl - bl - tr)
    if prod_iimg is None:
        return tf.concat((mu, n), axis=0)

    tl_prod = prod_iimg[xmin,ymin] if tf.minimum(xmin,ymin) >= 0 else zero
    bl_prod = prod_iimg[xmax,ymin] if tf.minimum(xmax,ymin) >= 0 else zero
    tr_prod = prod_iimg[xmin,ymax] if tf.minimum(xmin,ymax) >= 0 else zero
    br_prod = prod_iimg[xmax,ymax] if tf.minimum(xmax,ymax) >= 0 else zero

    sigma = _flatten((
        br_prod + tl_prod - bl_prod - tr_prod \
        - (tf.expand_dims(mu, 1) @ tf.expand_dims(mu, 0))/n
    )/(n-1))

    return tf.concat((mu, n, sigma), axis=0)

def windows_stats(iimg, prod_iimg, centers, radius, batch_shape=(-1,)):
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

        Returns
        -------
        mu: tf.Tensor(shape=(...,C,1), dtype='a)
            sum of the image pixels in the window
        n: tf.Tensor(shape=(...,1,1), dtype='a)
            size of the window
        sigma: tf.Tensor(shape=(...,C,C), dtype='a)
            covariance matrix of the image pixels in the window
    """
    f = lambda c: _window_stats(iimg, prod_iimg, c, radius)
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



def batch_sparse_matmul(M, x):
    f = lambda v: tf.sparse.sparse_dense_matmul(M, v)
    return tf.map_fn(f, x)
