import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import tensorflow as tf

from components.VGG19.model import StyleContentModel
from components.segmentation import compute_segmentation
from components.semantic_merge import merge_segments, reduce_dict, mask_for_tf, extract_segmentation_masks
from components.loss import Loss


print(tf.__version__)
print(tf.executing_eagerly())


def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.compat.v1.variables_initializer(adam_vars)



def calculate_photorealism_regularization(output, content_image, matting_method):
    # normalize content image and out for matting and regularization computation
    content_image = content_image / 255.0
    output = output / 255.0

    # compute matting laplacian
    if matting_method == 'fast':
        from components.matting import fast_matting_laplacian
        matting = fast_matting_laplacian(content_image[0, ...])
    else:
        from components.matting import matting_laplacian
        matting = matting_laplacian(content_image[0, ...])

    # compute photorealism regularization loss
    v = tf.reshape(tf.transpose(output), (output.shape[-1], -1))

    regularization_channels = tf.expand_dims(v,1) @ tf.expand_dims(matting(v), 2)

    regularization = tf.reduce_sum(input_tensor=regularization_channels)
    return regularization





def load_image(filename, dtype):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.cond(
        tf.image.is_jpeg(image_string),
        lambda: tf.image.decode_jpeg(image_string, channels=3),
        lambda: tf.image.decode_png(image_string, channels=3)
    )
    image = tf.image.convert_image_dtype(image_decoded, dtype)
    image = tf.expand_dims(image, 0)
    return image

def save_image(image, file):
    tf.io.write_file(file.name, image)

def tensor_to_image(tensor):
    r"""
        Taken from https://www.tensorflow.org/tutorials/generative/style_transfer
    """
    # tensor = np.array(tensor, dtype=np.uint8)
    # image = image[0, :, :, :]
    # image = np.clip(image, 0, 255)
    #
    # return Image.fromarray(image)
    image = tf.squeeze(tf.cast(255*tensor, dtype=tf.uint8))
    return tf.image.encode_png(image)


def change_filename(dir_name, filename, suffix, extension=None):
    r"""
        Adds a suffix at the end of file

        Example
        -------
        >>> change_filename('.', 'image.png', '_seg')
        ./image_seg.png
    """
    path, ext = os.path.splitext(filename)
    if extension is None:
        extension = ext
    return os.path.join(dir_name, path + suffix + extension)


def write_metadata(args, load_segmentation):
    # collect metadata and write to transfer dir
    meta = {
        "init": args.init,
        "iter": args.iter,
        "content": args.content_image,
        "style": args.style_image,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "regularization_weight": args.regularization_weight,
        "nima_weight": args.nima_weight,
        "semantic_thresh": args.semantic_thresh,
        "similarity_metric": args.similarity_metric,
        "load_segmentation": load_segmentation,
        "adam": {
            "learning_rate": args.adam_lr,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon
        }
    }
    file = experiment_path / 'meta.json'
    with file.open('w+') as f:
        f.write(json.dumps(meta, indent=4))


if __name__ == "__main__":

    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    base = parser.add_argument_group('Base options')
    expr = parser.add_argument_group('Experiment parameters')
    param = parser.add_argument_group('Hyperparameters')
    dirs = parser.add_argument_group('Storage directories')
    misc = parser.add_argument_group('Miscellaneous')

    base.add_argument("-c", "--content_image", type=str, help="Content image path", default="blanc.jpg")
    base.add_argument("-s", "--style_image", type=str, help="Style image path", default="bear.jpeg")
    base.add_argument("-o", "--output_image", type=str, help="Output image path, default: result.jpg",
                        default="result.jpg")

    expr.add_argument("--dtype", type=str, help="dtype of the input and output images., default: float32",
                        default="float32")
    expr.add_argument("--init", type=str, help="Initialization image., default: content",
                        choices=["noise", "content", "style"],
                        default="content")
    expr.add_argument("--iter", type=int, help="Number of iterations, default: 4000",
                        default=1000)
    expr.add_argument("--similarity_metric", type=str,
                        help="Semantic similarity metric for label grouping., default: li",
                        choices=["li", "wpath", "jcn", "lin", "wup", "res"],
                        default="li")
    # For more information on the similarity metrics: http://gsi-upm.github.io/sematch/similarity/#word-similarity

    # weights
    param.add_argument("--content_weight", type=float,
                        help="Weight of the content loss., default: 1",
                        default=1)
    param.add_argument("--style_weight", type=float,
                        help="Weight of the style loss., default: 100",
                        default=1e2)
    param.add_argument("--regularization_weight", type=float,
                        help="Weight of the photorealism regularization.",
                        default=1e4)
    param.add_argument("--nima_weight", type=float,
                        help="Weight for nima loss.",
                        default=1e5)
    # Adam parameters
    param.add_argument("--adam_lr", type=float,
                        help="Learning rate for the adam optimizer., default: 1.0",
                        default=1e-1)
    param.add_argument("--adam_beta1", type=float,
                        help="Beta1 for the adam optimizer., default: 0.9",
                        default=0.9)
    param.add_argument("--adam_beta2", type=float,
                        help="Beta2 for the adam optimizer., default: 0.999",
                        default=0.999)
    param.add_argument("--adam_epsilon", type=float,
                        help="Epsilon for the adam optimizer., default: 1e-08",
                        default=1e-08)
    # matting laplacian matric parameters
    param.add_argument("--matting_epsilon", type=float,
                        help="Epsilon regularization for matting laplacian computing., default=1e-5",
                        default=1e-5)
    param.add_argument("--matting_window_radius", type=int,
                        help="Size of the windows considered by matting laplacian., default=3",
                        default=3)
    param.add_argument("--semantic_thresh", type=float, help="Semantic threshold for label grouping., default: 0.5",
                        default=0.5)

    dirs.add_argument("--logs_dir", type=Path, help="Path to tensorboard logs., default: ./logs",
                        default=Path('logs'))
    dirs.add_argument("--results_dir", type=Path, help='Where results are stored., default: ./experiments',
                        default=Path('experiments'))
    dirs.add_argument("--seg_dir", type=Path, help='Where segmented images are stored., default: ./raw_seg',
                        default=Path('raw_seg'))

    misc.add_argument("--gpu", type=str, help="Comma separated list of GPU(s) to use.",
                        default="0")
    misc.add_argument("--experiment_name", type=str, help="Name of the experiment., default: <timestamp>",
                        default=None)
    misc.add_argument("--intermediate_result_interval", type=int,
                        help="Interval of iterations until a intermediate result is saved., default: 100",
                        default=20)
    misc.add_argument("--print_loss_interval", type=int,
                        help="Interval of iterations until the current loss is printed to console., default: 1",
                        default=10)

    args = parser.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if not args.experiment_name:
        from datetime import datetime
        args.experiment_name = datetime.now().strftime('%Y-%m-%d_%H:%M')
    experiment_path = args.results_dir / args.experiment_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    args.seg_dir.mkdir(parents=True, exist_ok=True)

    # check if manual segmentation masks are available
    content_segmentation_filename = change_filename(args.seg_dir, args.content_image, '_seg', '.png')
    style_segmentation_filename = change_filename(args.seg_dir, args.style_image, '_seg', '.png')
    load_segmentation = os.path.exists(content_segmentation_filename) and os.path.exists(style_segmentation_filename)

    write_metadata(args, load_segmentation)

    # args.content_image = args.data_dir / args.content_image
    # args.style_image = args.data_dir / args.style_image

    """Check if image files exist"""
    for file in [args.content_image, args.style_image]:
        if not os.path.exists(file):
            print("Image file {} does not exist.".format(file))
            exit(1)

    tf.keras.backend.set_floatx(args.dtype)

    content_image = load_image(args.content_image, args.dtype)
    style_image = load_image(args.style_image, args.dtype)

    # use existing if available
    if (load_segmentation):
        print("Load segmentation from files.")
        content_segmentation_image = cv2.imread(content_segmentation_filename)
        style_segmentation_image = cv2.imread(style_segmentation_filename)
        content_segmentation_masks = extract_segmentation_masks(content_segmentation_image)
        style_segmentation_masks = extract_segmentation_masks(style_segmentation_image)
    # otherwise compute it
    else:
        print("Create segmentation.")
        content_segmentation, style_segmentation = compute_segmentation(args.content_image, args.style_image)

        cv2.imwrite(change_filename(args.seg_dir, args.content_image, '_seg_raw', '.png'), content_segmentation)
        cv2.imwrite(change_filename(args.seg_dir, args.style_image, '_seg_raw', '.png'), style_segmentation)

        content_segmentation_masks, style_segmentation_masks = merge_segments(content_segmentation, style_segmentation,
                                                                              args.semantic_thresh, args.similarity_metric)

    cv2.imwrite(change_filename(args.seg_dir, args.content_image, '_seg', '.png'),
                reduce_dict(content_segmentation_masks, content_image))
    cv2.imwrite(change_filename(args.seg_dir, args.style_image, '_seg', '.png'),
                reduce_dict(style_segmentation_masks, style_image))

    if args.init == "noise":
        random_noise_scaling_factor = 0.0001
        random_noise = np.random.randn(*content_image.shape).astype(np.float32)
        init_image = vgg.postprocess(random_noise * random_noise_scaling_factor).astype(np.float32)
    elif args.init == "content":
        init_image = load_image(args.content_image, args.dtype)
    elif args.init == "style":
        init_image = load_image(args.style_image, args.dtype)
    else:
        print("Init image parameter {} unknown.".format(args.init))
        exit(1)

    iterations_dir = experiment_path / 'iter'
    iterations_dir.mkdir(exist_ok=True)





    ##################################

    #####     STYLE TRANSFER     #####

    ##################################


    print("Style transfer started")

    print('Initializing features extractor...', end='\t', flush=True)
    content_layers = ['block4_conv2']
    style_layers = ['block%d_conv1' % (i+1) for i in range(5)]

    features_extractor = StyleContentModel(content_layers, style_layers, shape=(None,None,3))
    print('Done.')

    content_target = features_extractor(content_image)['content']
    style_target = features_extractor(style_image)['style']

    compute_loss = Loss(
        content_target,
        style_target,
        args,
        #mask_for_tf(content_segmentation_masks),
        #mask_for_tf(style_segmentation_masks)
    )

    if args.regularization_weight > 0:
        print('Initializing matting laplacian...', end='\t', flush=True)
        t0 = time.time()
        compute_loss.initialize_matting_laplacian(tf.cast(tf.squeeze(content_image), tf.float64))
        t1 = time.time()
        print('Done. {:.3f}s'.format(t1-t0))

    # TODO: summary cf tf1

    optimizer = tf.optimizers.Adam(
        learning_rate=args.adam_lr,
        beta_1=args.adam_beta1,
        beta_2=args.adam_beta2,
        epsilon=args.adam_epsilon
    )

    # TODO: init_image instead of content_image
    transfer_image = tf.Variable(content_image, trainable=True)

    @tf.function
    def train_step(image):
        # forward
        with tf.GradientTape() as tape:
            outputs = features_extractor(image)
            loss_dict = compute_loss(image, outputs)

        total_loss = loss_dict['Total loss']

        # backward
        grad = tape.gradient(total_loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0, 1))
        return loss_dict

    min_loss = float('inf')
    best_image = None

    epoch_duration = 0

    writer = tf.summary.create_file_writer(str(args.logs_dir / args.experiment_name))

    for i in range(1, args.iter + 1):
        # summary_writer.add_summary(summary, i)
        t0 = time.time()
        loss_dict = train_step(transfer_image)
        t1 = time.time()
        epoch_duration += t1 - t0

        if i % args.print_loss_interval == 0:
            tf.print("[Iter {}]".format(i), end='\t')
            for loss_name, loss_value in loss_dict.items():
                tf.print('{}: {:<15.3f}'.format(loss_name, loss_value), end='')
            tf.print()

        with writer.as_default():
            for loss_name, loss_value in loss_dict.items():
                tf.summary.scalar(loss_name, loss_value, step=i)
        #
        if loss_dict['Total loss'].numpy() < min_loss:
            min_loss, best_image = loss_dict['Total loss'], tensor_to_image(transfer_image)
        #
        if i % args.intermediate_result_interval == 0:
            save_image(best_image, iterations_dir / "iter_{}.png".format(i))
            with writer.as_default():
                tf.summary.image('Transfer image', transfer_image, step=i)

    print("Style transfer finished. Average time per epoch: {:.5f}s\n".format(epoch_duration/args.iter))
    # save_image(result, os.path.join(args.results_dir, "final_transfer_image.png"))
