import argparse

import tensorflow as tf

import base_arg
from model import CRNN

parser = argparse.ArgumentParser(parents=[base_arg.parser])
parser.add_argument("--checkpoint", type=str, required=True, help="The checkpoint path.")
parser.add_argument("-f", "--format", type=str, choices=("tf", "h5"), required=True, help="Format H5 or SavedModel(tf).")
parser.add_argument("-o", "--output", type=str, required=True, help="The output path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)

if __name__ == "__main__":
    model = CRNN(NUM_CLASSES, args.backbone)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        exit(0)

    model.save(args.output, save_format=args.format)
    print("Format: {}, saved to {}".format(args.format, args.output))