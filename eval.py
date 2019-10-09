import argparse
import time

import numpy as np
import tensorflow as tf

from model import CRNN
from dataset import OCRDataLoader, map_to_chars

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--annotation_path", type=str, help="The path of annnotation file.")
parser.add_argument("--image_height", type=int, default=32, help="Image height(32). If you change this, you should change the structure of CNN.")
parser.add_argument("-w", "--image_width", type=int, default=100, help="Image width(>=16).")
parser.add_argument("-t", "--table_path", type=str, help="The path of table file.")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--checkpoint_path", type=str, help="The checkpoint path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    int_to_char = [char.strip() for char in f]
num_classes = len(int_to_char)
blank_index = num_classes - 1 # Make sure the blank index is what.

def eval(num_classes):
    dataloader = OCRDataLoader(args.annotation_path, 
                               args.image_height, 
                               args.image_width, 
                               table_path=args.table_path,
                               blank_index=blank_index,
                               shuffle=True, 
                               batch_size=args.batch_size)
    model = CRNN(num_classes)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_path))

    num_right = 0
    for X, y in dataloader():
        start_time = time.perf_counter()
        y_pred = model(X)
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                           sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                                           merge_repeated=True)
        decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
        y = tf.sparse.to_dense(y, default_value=blank_index).numpy()

        for y, y_pred in zip(map_to_chars(y, int_to_char, blank_index=blank_index),
                             map_to_chars(decoded, int_to_char, blank_index=blank_index)):
            if y == y_pred:
                num_right += 1

    print("right: {}, total: {}, acc: {}".format(num_right, len(dataloader), num_right / len(dataloader)))

if __name__ == "__main__":
    eval(num_classes)