import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19

def vgg_layers(layer_names, shape):
    vgg = vgg19.VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=shape,
        pooling='max'
    )
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    return model

class StyleContentModel(tf.keras.Model):
    def __init__(self, content_layers, style_layers, shape=None):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(content_layers+style_layers, shape)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.limit = len(content_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = 255 * inputs
        preprocessed_input = vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        content_outputs, style_outputs = outputs[:self.limit], outputs[self.limit:]

        content_dict = {layer_name: value
                        for layer_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {layer_name: value
                      for layer_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
