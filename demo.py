import os
import argparse

import tensorflow as tf
from tensorflow import keras

from dataset import Decoder

def read_image(path):
    img = tf.io.read_file(path)
    try:
        img = tf.io.decode_jpeg(img, channels=1)
    except Exception:
        print("Invalid image: {}".format(path))
        global num_invalid
        return tf.zeros((32, args.image_width, 1))
    img = tf.image.convert_image_dtype(img, tf.float32)
    if args.keep_ratio:
        width = round(32 * img.shape[1] / img.shape[0])
    else: 
        width = args.image_width
    img = tf.image.resize(img, (32, width))
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, 
                        help="Images file path.")
    parser.add_argument("-t", "--table_path", type=str, required=True, 
                        help="The path of table file.")
    parser.add_argument("-w", "--image_width", type=int, default=100, 
                        help="Image width(>=16).")
    parser.add_argument("-k", "--keep_ratio", action="store_true",
                        help="Whether keep the ratio.")
    parser.add_argument("-m", "--model", type=str, required=True, 
                        help="The saved model path.")
    args = parser.parse_args()

    if args.images is not None:
        if os.path.isdir(args.images):
            imgs_path = os.listdir(args.images)
            img_paths = [os.path.join(args.images, img_path) 
                         for img_path in imgs_path]
            imgs = list(map(read_image, img_paths))
            imgs = tf.stack(imgs)
        else:
            img_paths = [args.images]
            img = read_image(args.images)
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