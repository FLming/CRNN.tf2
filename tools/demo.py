import argparse
from pathlib import Path

import yaml
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, required=True, 
                    help='Image file or folder path.')
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--model', type=str, required=True, 
                    help='The saved model.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']


def read_img_and_resize(path, shape):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=shape[2])
    if shape[1] is None:
        img_shape = tf.shape(img)
        scale_factor = shape[0] / img_shape[0]
        img_width = scale_factor * tf.cast(img_shape[1], tf.float64)
        img_width = tf.cast(img_width, tf.int32)
    else:
        img_width = shape[1]
    img = tf.image.resize(img, (shape[0], img_width))
    return img


model = keras.models.load_model(args.model, compile=False)

p = Path(args.images)
img_paths = p.iterdir() if p.is_dir() else [p]
for img_path in img_paths:
    img = read_img_and_resize(str(img_path), config['img_shape'])
    img = tf.expand_dims(img, 0)
    outputs = model(img)
    print(f'Path: {img_path}, y_pred: {outputs[0].numpy()}, '
          f'probability: {outputs[1].numpy()}')