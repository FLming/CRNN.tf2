import os
import time
import argparse

import tensorflow as tf

from dataset import map_to_chars

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--images", type=str, 
                    help="Images file path.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-k", "--keep_ratio", action="store_true",
                    help="Whether keep the ratio.")
parser.add_argument("--model", type=str, required=True, 
                    help="The SavedModel path or h5 file.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    inv_table = [char.strip() for char in f]
num_classes = len(inv_table)
blank_index = num_classes - 1

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

def greedy_decode(logits):
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    decoded = map_to_chars(decoded, inv_table, blank_index)
    return decoded, neg_sum_logits

def beam_search_decode(logits):
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length)
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    decoded = map_to_chars(decoded, inv_table, blank_index)
    return decoded, log_probabilities

if __name__ == "__main__":
    if args.images is not None:
        if os.path.isdir(args.images):
            imgs_path = os.listdir(args.images)
            img_paths = [os.path.join(args.images, img_path) 
                         for img_path in imgs_path]
            imgs = list(map(read_image, img_paths))
            imgs = tf.stack(imgs)
        else:
            img_paths = args.images
            img = read_image(args.images)
            imgs = tf.expand_dims(img, 0)

    model = tf.keras.models.load_model(args.model)
    print("Restored from {}".format(args.model))

    logits = model(imgs, training=False)

    g_decoded, neg_sum_logits = greedy_decode(logits)
    b_decoded, log_probabilities = beam_search_decode(logits)
    for path, g_pred, b_pred in zip(imgs_path, g_decoded, b_decoded):
        print("Path: {}".format(path))
        print("\tGreedy: {}".format(g_pred))
        print("\tBeam search: {}".format(b_pred))