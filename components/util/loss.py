import numpy as np

import tensorflow as tf


def iter_on_layers(func):
    def wrapped(*args, **kwargs):
        if len(args) == 0:
            return tf.constant(0)
        return tf.add_n([func(*[a[key] for a in args], **kwargs) for key in args[0].keys()])/len(args)
    return wrapped


### CONTENT LOSS

def calculate_layer_content_loss(target, output):
    return tf.reduce_mean(input_tensor=tf.math.squared_difference(target, output))

calculate_content_loss = iter_on_layers(calculate_layer_content_loss)


### STYLE LOSS

def calculate_gram_matrix(convolution_layer, mask):
    matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
    mask_reshaped = tf.reshape(mask, shape=[matrix.shape[0], 1])
    matrix_masked = matrix * mask_reshaped
    return tf.matmul(matrix_masked, matrix_masked, transpose_a=True)

def calculate_layer_style_loss(target, output, content_masks=None, style_masks=None):
    # scale masks to current layer
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
        transfer_gram_matrix = calculate_gram_matrix(output, content_mask)
        style_gram_matrix = calculate_gram_matrix(target, style_mask)

        mean = tf.reduce_mean(input_tensor=tf.math.squared_difference(style_gram_matrix, transfer_gram_matrix))
        means_per_channel.append(mean / (2 * tf.square(feature_map_count) * tf.square(feature_map_size)))

    style_loss = tf.reduce_sum(input_tensor=means_per_channel)

    return style_loss

calculate_style_loss = iter_on_layers(calculate_layer_style_loss)


### NIMA LOSS

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




def compute_loss(targets, outputs, args, content_masks, style_masks):
    content_target, style_target = targets
    content_output, style_output = outputs['content'], outputs['style']

    loss_names, loss_values, loss_weights = [], [], []

    loss_names.append('Content loss')
    loss_values.append(calculate_content_loss(content_target, content_output))
    loss_weights.append(args.content_weight)

    loss_names.append('Style loss')
    loss_values.append(calculate_style_loss(style_target, style_output))#, content_masks=content_masks, style_masks=style_masks))
    loss_weights.append(args.style_weight)



    # compute total loss
    total_loss = tf.add_n([w*loss for w, loss in zip(loss_weights, loss_values)])

    # fill dict
    loss_dict = {key: value for key, value in zip(loss_names, loss_values)}
    loss_dict['Total loss'] = total_loss

    return loss_dict
