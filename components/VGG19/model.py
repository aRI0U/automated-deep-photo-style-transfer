import os
import numpy as np

import tensorflow as tf
import tensorflow.keras.applications.vgg19 as vgg19

# from tensorpack.models.conv2d import *
# from tensorpack.models.pool import *
# from tensorpack.tfutils.argscope import *
# from tensorpack.tfutils.sessinit import *
# from tensorpack.tfutils.symbolic_functions import *
#
# from components.path import WEIGHTS_DIR
#
# """
# Subset of the VGG19 model with all convolution layers, trained on ImageNet
# """
#
# VGG_MEAN = [103.939, 116.779, 123.68]
#
#
# def preprocess(image, to_tensor=False):
#     # TODO: make clean architecture to be able to always convert to tensor
#     if to_tensor:
#         image = tf.cast(image, dtype=tf.float32)
#     return image[:, :, :, ::-1] - VGG_MEAN
#
#
# def postprocess(image):
#     return np.round((image + VGG_MEAN)[:, :, :, ::-1], decimals=0)
#
#
# def load_weights():
#     param_dict = np.load(os.path.join(WEIGHTS_DIR, 'VGG19/vgg19.npz'))
#     return DictRestore(dict(param_dict))
#
#
# class VGG19ConvSub:
#     def __init__(self, image):
#         with argscope(Conv2D, kernel_shape=3, nl=tf.identity):
#             self.conv1_1 = Conv2D("conv1_1", image, 64)
#             self.relu1_1 = tf.nn.relu(self.conv1_1, "relu1_1")
#             self.conv1_2 = Conv2D("conv1_2", self.relu1_1, 64)
#             self.relu1_2 = tf.nn.relu(self.conv1_2, "relu1_2")
#             self.pool1 = MaxPooling("pool1", self.relu1_2, 2)
#
#             self.conv2_1 = Conv2D("conv2_1", self.pool1, 128)
#             self.relu2_1 = tf.nn.relu(self.conv2_1, "relu2_1")
#             self.conv2_2 = Conv2D("conv2_2", self.relu2_1, 128)
#             self.relu2_2 = tf.nn.relu(self.conv2_2, "relu2_2")
#             self.pool2 = MaxPooling("pool2", self.relu2_2, 2)
#
#             self.conv3_1 = Conv2D("conv3_1", self.pool2, 256)
#             self.relu3_1 = tf.nn.relu(self.conv3_1, "relu3_1")
#             self.conv3_2 = Conv2D("conv3_2", self.relu3_1, 256)
#             self.relu3_2 = tf.nn.relu(self.conv3_2, "relu3_2")
#             self.conv3_3 = Conv2D("conv3_3", self.relu3_2, 256)
#             self.relu3_3 = tf.nn.relu(self.conv3_3, "relu3_3")
#             self.conv3_4 = Conv2D("conv3_4", self.relu3_3, 256)
#             self.relu3_4 = tf.nn.relu(self.conv3_4, "relu3_4")
#             self.pool3 = MaxPooling("pool3", self.relu3_4, 2)
#
#             self.conv4_1 = Conv2D("conv4_1", self.pool3, 512)
#             self.relu4_1 = tf.nn.relu(self.conv4_1, "relu4_1")
#             self.conv4_2 = Conv2D("conv4_2", self.relu4_1, 512)
#             self.relu4_2 = tf.nn.relu(self.conv4_2, "relu4_2")
#             self.conv4_3 = Conv2D("conv4_3", self.relu4_2, 512)
#             self.relu4_3 = tf.nn.relu(self.conv4_3, "relu4_3")
#             self.conv4_4 = Conv2D("conv4_4", self.relu4_3, 512)
#             self.relu4_4 = tf.nn.relu(self.conv4_4, "relu4_4")
#             self.pool4 = MaxPooling("pool4", self.relu4_4, 2)
#
#             self.conv5_1 = Conv2D("conv5_1", self.pool4, 512)
#             self.relu5_1 = tf.nn.relu(self.conv5_1, "relu5_1")



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
    def __init__(self, content_layers, style_layers, shape):
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
