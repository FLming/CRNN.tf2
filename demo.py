import argparse
import time

import tensorflow as tf

from model import CRNN
from dataset import map_to_chars

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, required=True)
parser.add_argument("-t", "--table_path", type=str, help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, help="Image width(>=16).")
parser.add_argument("--checkpoint", type=str, help="The checkpoint path.")
parser.add_argument("--image_height", type=int, default=32, help="Image height(32). If you change this, you should change the structure of CNN.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

if __name__ == "__main__":
    img = tf.io.read_file(args.image_path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [args.image_height, args.image_width])
    
    model = CRNN(NUM_CLASSES)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        exit(0)

    y_pred = model(tf.expand_dims(img, 0), training=False)

    # beam_search
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                            sequence_length=[y_pred.shape[1]]*y_pred.shape[0])
    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
    print("[Beam search] prediction: {}, log probabilities: {}".format(decoded[0], log_probabilities[0][0]))
    # greedy
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                       sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                                       merge_repeated=True)

    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
    print("[Greedy] prediction: {}, neg sum logits: {}".format(decoded[0], neg_sum_logits[0][0]))