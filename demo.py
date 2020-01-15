import os
import time
import argparse

import tensorflow as tf

import arg
from dataset import map_to_chars

parser = argparse.ArgumentParser(parents=[arg.parser])
parser.add_argument("-i", "--images", type=str, 
                    help="Images file path.")
parser.add_argument("-a", "--annotation", type=str, help="Groundtruth file.")
parser.add_argument("--model", type=str, required=True, 
                    help="The SavedModel path or h5 file.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

def read_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [args.image_height, args.image_width])
    return img

def greedy_decode(logits):
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
    return decoded, neg_sum_logits

def beam_search_decode(logits):
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length)
    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
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
            img = read_image(args.images)
            imgs = tf.expand_dims(img, 0)

    if args.annotation is not None:
        # parse ICDAR2013 dataset gt.
        with open(args.annotation) as f:
            content = f.readlines()
            content = [line.strip().split(",") for line in content]
            gt = {v[0]: v[1].strip(' "') for v in content}
    
    model = tf.keras.models.load_model(args.model)
    print("Restored from {}".format(args.model))

    logits = model(imgs, training=False)

    decoded, neg_sum_logits = greedy_decode(logits)
    count = 0
    num_correct_g = 0
    print("*************** Greedy ***************")
    for path, pred in zip(imgs_path, decoded):
        if args.annotation is not None:
            if not gt[path].isalnum():
                continue
            if gt[path].lower() == pred.lower():
                num_correct_g += 1
            else:
                print(f"Path: {path}, gt: {gt[path]}, prediction: {pred}")
            count += 1
        else:
            print(f"Path: {path}, prediction: {pred}")
    
    decoded, log_probabilities = beam_search_decode(logits)
    num_correct_b = 0
    print("*************** Beam search ***************")
    for path, pred in zip(imgs_path, decoded):
        if args.annotation is not None:
            if not gt[path].isalnum():
                continue
            if gt[path].lower() == pred.lower():
                num_correct_b += 1
            else:
                print(f"Path: {path}, gt: {gt[path]}, prediction: {pred}")
        else:
            print(f"Path: {path}, prediction: {pred}")
    
    if args.annotation is not None:
        print(f"[Summary] Greedy accuracy: {num_correct_g / count}"
              f", beam search accuracy: {num_correct_b / count}")