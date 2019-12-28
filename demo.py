import argparse
import time

import tensorflow as tf

import base_arg
from dataset import map_to_chars

parser = argparse.ArgumentParser(parents=[base_arg.parser])
parser.add_argument("-i", "--image_path", type=str, required=True, help="Image file path.")
parser.add_argument("--model", type=str, required=True, help="The SavedModel path or h5 file.")
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
    
    model = tf.keras.models.load_model(args.model)
    print("Restored from {}".format(args.model))

    y_pred = model(tf.expand_dims(img, 0), training=False)

    # greedy
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                       sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                                       merge_repeated=True)

    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
    print("[Greedy] prediction: {}, neg sum logits: {}".format(decoded[0], neg_sum_logits[0][0]))
    # beam_search
    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                            sequence_length=[y_pred.shape[1]]*y_pred.shape[0])
    decoded = tf.sparse.to_dense(decoded[0], default_value=BLANK_INDEX).numpy()
    decoded = map_to_chars(decoded, INT_TO_CHAR, blank_index=BLANK_INDEX)
    print("[Beam search] prediction: {}, log probabilities: {}".format(decoded[0], log_probabilities[0][0]))