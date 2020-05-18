import argparse
import os

import tensorflow as tf
from tensorflow import keras

from dataset import Decoder

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", type=str, required=True, 
                    help="Image file or folder path.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width, this parameter will affect the output "
                         "shape of the model, default is 100, so this model "
                         "can only predict up to 24 characters.")
parser.add_argument("-k", "--keep_ratio", action="store_true",
                    help="Whether keep the image ratio.")
parser.add_argument("-m", "--model", type=str, required=True, 
                    help="The saved model.")
parser.add_argument("--channels", type=int, default=1, help="Image channels, "
                    "0: Use the number of channels in the image, "
                    "1: Grayscale image, "
                    "3: RGB image")
args = parser.parse_args()


def read_img_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=args.channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if args.keep_ratio:
        image_width = round(32 * img.shape[1] / img.shape[0])
    img = tf.image.resize(img, (32, args.image_width))
    return img


if os.path.isdir(args.images):
    img_paths = os.listdir(args.images)
    img_paths = [os.path.join(args.images, path) for path in img_paths]
    imgs = list(map(read_img_and_preprocess, img_paths))
    imgs = tf.stack(imgs)
else:
    img_paths = [args.images]
    img = read_img_and_preprocess(args.images)
    imgs = tf.expand_dims(img, 0)

with open(args.table_path, "r") as f:
    inv_table = [char.strip() for char in f]

model = keras.models.load_model(args.model, compile=False)
decoder = Decoder(inv_table)

y_pred = model.predict(imgs)
for path, g_pred, b_pred in zip(img_paths, 
                                decoder.decode(y_pred, method='greedy'),
                                decoder.decode(y_pred, method='beam_search')):
    print("Path: {}, greedy: {}, beam search: {}".format(path, g_pred, b_pred))