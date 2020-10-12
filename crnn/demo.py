import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from utils import Decoder

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images', type=str, required=True, 
                    help='Image file or folder path.')
parser.add_argument('-t', '--table_path', type=str, required=True, 
                    help='The path of table file.')
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='The saved model.')
parser.add_argument('-w', '--img_width', type=int, default=0, 
                    help='Image width, this parameter will affect the output '
                         'shape of the model, default is 0')
parser.add_argument('--img_channels', type=int, default=3, 
                    help='0: Use the number of channels in the image, '
                         '1: Grayscale image, 3: RGB image')
args = parser.parse_args()


def read_img_and_preprocess(path, img_width=0, img_height=32):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=args.img_channels)
    if not img_width:
        img_shape = tf.shape(img)
        scale_factor = img_height / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
    img = tf.image.resize(img, (img_height, img_width)) / 255.0
    return img


with open(args.table_path, 'r') as f:
    table = [char.strip() for char in f]

model = keras.models.load_model(args.model, compile=False)

decoder = Decoder(table)

p = Path(args.images)
if p.is_dir():
    img_paths = p.iterdir()
else:
    img_paths = [p]

for img_path in img_paths:
    img = read_img_and_preprocess(str(img_path))
    img = tf.expand_dims(img, 0)
    y_pred = model(img)
    y_pred = decoder.decode(y_pred, method='greedy')[0]
    print(f'Path: {img_path}, greedy pred: {y_pred}')