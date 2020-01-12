import numpy as np
import os
import time

import tensorflow as tf

from components.matting_v2 import MattingLaplacian

class Loss:
    r"""
        Loss functions are computed within this class
    """
    def __init__(self, content_target, style_target, args, content_masks=None, style_masks=None):
        self.content_target = content_target
        self.style_target = style_target
        self.content_masks = content_masks
        self.style_masks = style_masks

        self.loss_names = {
            'content': 'Content loss',
            'style':   'Style loss',
            'photo':   'Photorealism regualarization'
        }
        self.loss_functions = {
            'content': self.iter_on_layers(self.calculate_layer_content_loss),
            'style':   self.iter_on_layers(self.calculate_layer_style_loss),
            'photo':   self.calculate_photorealism_regularization
        }
        self.loss_weights = {
            'content': args.content_weight,
            'style':   args.style_weight,
            'photo':   args.regularization_weight
        }
        assert self.loss_names.keys() == self.loss_functions.keys() == self.loss_weights.keys()

        self.matting_params = {
            'epsilon': args.matting_epsilon,
            'window_radius': args.matting_window_radius,
        }

        # TODO: put that stuff in main
        if args.matting_dir:
            root = os.path.join(
                args.matting_dir,
                str(args.matting_epsilon),
                str(args.matting_window_radius)
            )
            os.makedirs(root, exist_ok=True)
            self.matting_params['fname'] = os.path.join(root, os.path.splitext(args.content_image)[0]+'.npz')
        self.matting_laplacian = None

    def initialize_matting_laplacian(self, image):
        self.matting_laplacian = MattingLaplacian(image, **self.matting_params)

    def __call__(self, image, outputs):
        return self.compute_loss(image, outputs)



    def compute_loss(self, image, outputs):
        content_output, style_output = outputs['content'], outputs['style']

        loss_values = {}

        calculate_content_loss = self.loss_functions['content']
        loss_values['content'] = calculate_content_loss(self.content_target, content_output)

        calculate_style_loss = self.loss_functions['style']
        loss_values['style'] = calculate_style_loss(self.style_target, style_output)

        if self.loss_weights['photo'] > 0:
            calculate_photorealism_regularization = self.loss_functions['photo']
            loss_values['photo'] = calculate_photorealism_regularization(image)

        # compute total loss
        total_loss = tf.add_n([w*loss for loss, w in dict_zip(loss_values, self.loss_weights)])

        # fill dict
        loss_dict = {name: loss for loss, name in dict_zip(loss_values, self.loss_names)}
        loss_dict['Total loss'] = total_loss

        return loss_dict

    @staticmethod
    def iter_on_layers(func):
        def wrapped(*args, **kwargs):
            if len(args) == 0:
                return tf.constant(0)
            return tf.add_n([func(*[a[key] for a in args], **kwargs) for key in args[0].keys()])/len(args)
        return wrapped


    ### CONTENT LOSS
    @staticmethod
    def calculate_layer_content_loss(target, output):
        return tf.reduce_mean(input_tensor=tf.math.squared_difference(target, output))


    ### STYLE LOSS
    @staticmethod
    def calculate_gram_matrix(convolution_layer, mask):
        matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
        mask_reshaped = tf.reshape(mask, shape=[matrix.shape[0], 1])
        matrix_masked = matrix * mask_reshaped
        return tf.matmul(matrix_masked, matrix_masked, transpose_a=True)

    def calculate_layer_style_loss(self, target, output):
        # TODO: use only tf, no numpy
        # scale masks to current layer
        content_masks, style_masks = self.content_masks, self.style_masks
        output_size = output.shape[1:3]
        target_size = target.shape[1:3]

        exist_masks = content_masks is not None and style_masks is not None

        def resize_masks(masks, size):
            return [tf.image.resize(mask, size) for mask in masks]

        if exist_masks:
            style_masks = resize_masks(style_masks, target_size)
            content_masks = resize_masks(content_masks, output_size)
        else:
            style_masks = [tf.constant(1., shape=target_size)]
            content_masks = [tf.constant(1., shape=output_size)]

        feature_map_count = np.float32(output.shape[3])
        feature_map_size = np.float32(output.shape[1]) * np.float32(output.shape[2])

        means_per_channel = []
        for content_mask, style_mask in zip(content_masks, style_masks):
            transfer_gram_matrix = self.calculate_gram_matrix(output, content_mask)
            style_gram_matrix = self.calculate_gram_matrix(target, style_mask)

            mean = tf.reduce_mean(input_tensor=tf.math.squared_difference(style_gram_matrix, transfer_gram_matrix))
            means_per_channel.append(mean / (2 * tf.square(feature_map_count) * tf.square(feature_map_size)))

        style_loss = tf.reduce_sum(input_tensor=means_per_channel)

        return style_loss


    ### NIMA LOSS
    @staticmethod
    def compute_nima_loss(image):
        model = nima.get_nima_model(image)

        def mean_score(scores):
            scores = tf.squeeze(scores)
            si = tf.range(1, 11, dtype=tf.float32)
            return tf.reduce_sum(input_tensor=tf.multiply(si, scores), name='nima_score')

        nima_score = mean_score(model.output)

        nima_loss = tf.identity(10.0 - nima_score, name='nima_loss')
        return nima_loss


    ### PHOTOREALISM REGULARIZATION
    def calculate_photorealism_regularization(self, image):
        # image.shape = (1, H, W, C
        HW = self.matting_laplacian.shape[-1]
        p = tf.reshape(image, (HW, -1))
        return tf.reduce_sum(p * self.matting_laplacian.matmul(p))

def dict_zip(*dicts):
    for k in dicts[0].keys():
        yield [d[k] for d in dicts]
