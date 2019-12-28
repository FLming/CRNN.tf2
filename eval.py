import argparse
import time

import numpy as np
import tensorflow as tf

import base_arg
from model import CRNN
from dataset import OCRDataLoader, map_and_count

parser = argparse.ArgumentParser(parents=[base_arg.parser])
parser.add_argument("-a", "--annotation_paths", type=str, required=True, help="The paths of annnotation file.")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size.")
parser.add_argument("--checkpoint", type=str, required=True, help="The checkpoint path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

@tf.function
def eval_one_step(model, X, Y):
    y_pred = model(X, training=False)
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                       sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                                       merge_repeated=True)
    return decoded, neg_sum_logits

if __name__ == "__main__":
    dataloader = OCRDataLoader(args.annotation_paths, 
                               args.image_height, 
                               args.image_width, 
                               table_path=args.table_path,
                               blank_index=BLANK_INDEX,
                               shuffle=True, 
                               batch_size=args.batch_size)
    print("Num of eval samples: {}".format(len(dataloader)))
    print("Num of classes: {}".format(NUM_CLASSES))
    print("Blank index is {}".format(BLANK_INDEX))
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start at {}".format(localtime))
    
    model = CRNN(NUM_CLASSES, args.backbone)
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        exit(0)

    num_correct_samples = 0
    for index, (X, Y) in enumerate(dataloader()):
        start_time = time.perf_counter()
        decoded, neg_sum_logits = eval_one_step(model, X, Y)
        end_time = time.perf_counter()
        count = map_and_count(decoded, Y, INT_TO_CHAR)
        num_correct_samples += count
        print("[{} / {}] Num of correct samples({:.4f}s/{}): {}".format((index + 1) * args.batch_size, 
                                                                        len(dataloader), 
                                                                        end_time - start_time, 
                                                                        args.batch_size,
                                                                        num_correct_samples))
    print("Total: {}, Correct: {}, Accuracy: {}".format(len(dataloader), 
                                                        num_correct_samples, 
                                                        num_correct_samples / len(dataloader)))