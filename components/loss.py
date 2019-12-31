import numpy as np

import tensorflow as tf

class Loss:
    r"""
        Loss functions are computed within this class
    """
    def __init__(self, content_target, style_target, args, content_masks=None, style_masks=None):
        self.content_target = content_target
        self.style_target = style_target
        self.content_masks = content_masks
        self.style_masks = style_masks

        self.loss_names = [
            'Content loss',
            'Style loss'
        ]
        self.loss_functions = [
            self.iter_on_layers(self.calculate_layer_content_loss),
            self.iter_on_layers(self.calculate_layer_style_loss)
        ]
        self.loss_weights = [
            args.content_weight,
            args.style_weight
        ]


    def __call__(self, outputs):
        return self.compute_loss(outputs)



    def compute_loss(self, outputs):
        content_output, style_output = outputs['content'], outputs['style']

        loss_values = []

        calculate_content_loss = self.loss_functions[0]
        loss_values.append(calculate_content_loss(self.content_target, content_output))

        calculate_style_loss = self.loss_functions[1]
        loss_values.append(calculate_style_loss(self.style_target, style_output))

        # compute total loss
        total_loss = tf.add_n([w*loss for w, loss in zip(self.loss_weights, loss_values)])

        # fill dict
        loss_dict = {key: value for key, value in zip(self.loss_names, loss_values)}
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
