import sys
import argparse
import time

import numpy as np
import tensorflow as tf

import arg
from model import CRNN
from dataset import OCRDataLoader, map_and_count

parser = argparse.ArgumentParser(parents=[arg.parser])
parser.add_argument("-a", "--annotation_paths", type=str, required=True, 
                    help="The paths of annnotation file.")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("--checkpoint", type=str, required=True, 
                    help="The checkpoint path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

@tf.function
def eval_one_step(model, x, y):
    logits = model(x, training=False)
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    return decoded, neg_sum_logits

if __name__ == "__main__":
    eval_dl = OCRDataLoader(
        args.annotation_paths, 
        args.image_height, 
        args.image_width, 
        table_path=args.table_path,
        blank_index=BLANK_INDEX,
        batch_size=args.batch_size)
    print(f"Num of eval samples: {len(eval_dl)}")
    print(f"Num of classes: {NUM_CLASSES}")
    print(f"Blank index is {BLANK_INDEX}")
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(f"Start at {localtime}")
    
    model = CRNN(NUM_CLASSES, args.backbone)
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print(f"Restored from {tf.train.latest_checkpoint(args.checkpoint)}")
    else:
        print("Initializing fail, check checkpoint")
        sys.exit()

    num_correct_samples = 0
    for index, (x, y) in enumerate(eval_dl()):
        start_time = time.perf_counter()
        decoded, neg_sum_logits = eval_one_step(model, x, y)
        end_time = time.perf_counter()
        cnt = map_and_count(decoded, y, INT_TO_CHAR)
        num_correct_samples += cnt
        print(f"[{(index + 1) * args.batch_size} / {len(eval_dl)}] "
              f"Num of correct samples: {num_correct_samples} "
              f"({end_time - start_time:.4f}s / {args.batch_size})")
    print(f"Total: {len(eval_dl)}, Correct: {num_correct_samples}, "
          f"Accuracy: {num_correct_samples / len(eval_dl) * 100}")