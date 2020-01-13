import os

import tensorflow as tf

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow.keras.layers as layers

from components.util import WEIGHTS_DIR

class NIMAModel(tf.keras.Model):
    def __init__(self):
        super(NIMAModel, self).__init__(name='NIMAModel')
        self.inception = InceptionResNetV2(
            include_top=False,
            input_shape=(None,None,3),
            pooling='avg',
            weights=None
        )
        self.dropout = layers.Dropout(0.25)
        self.dense = layers.Dense(10, activation='softmax')

        self.load_weights(os.path.join(WEIGHTS_DIR, 'NIMA/inception_resnet_weights.h5'), by_name=True)
        self.trainable = False

    def call(self, inputs):
        x = self.inception(inputs)
        x = self.dropout(x)
        return self.dense(x)


def preprocess(image):
    return (image / 127.5) - 1.0


def postprocess(image):
    return (image + 1.0) * 127.5


if __name__ == '__main__':
    from datetime import datetime
    from style_transfer import load_image, save_image, adam_variables_initializer, compute_nima_loss
    import tensorflow as tf

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

    result_dir = 'result_' + timestamp
    os.mkdir(result_dir)

    content_image = load_image("content.png")
    content_image = preprocess(content_image)

    with tf.Session() as sess:
        transfer_image = tf.Variable(content_image)

        sess.run(tf.global_variables_initializer())

        nima_loss = compute_nima_loss(transfer_image)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        train_op = optimizer.minimize(nima_loss, var_list=[transfer_image])
        sess.run(adam_variables_initializer(optimizer, [transfer_image]))

        min_loss, best_image = float("inf"), None
        for i in range(1000 + 1):
            _, result_image, n_loss = sess.run(
                fetches=[train_op, transfer_image, nima_loss])

            if i % 1 == 0:
                print(
                    "Iteration: {0:5} \t "
                    "NIMA loss: {1:15.2f} \t".format(i, n_loss))

            if n_loss < min_loss:
                min_loss, best_image = n_loss, result_image

            if i % 100 == 0:
                save_image(postprocess(best_image), os.path.join(result_dir, "iter_{}.png".format(i)))
