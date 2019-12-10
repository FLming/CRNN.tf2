import argparse

import tensorflow as tf

from model import CRNN

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--table_path", type=str, help="The path of table file.")
parser.add_argument("--checkpoint", type=str, help="The checkpoint path.")
parser.add_argument("-f", "--format", type=str, help="Format H5 or SavedModel(tf).")
parser.add_argument("-o", "--output", type=str, help="The output path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)

if __name__ == "__main__":
    model = CRNN(NUM_CLASSES)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        exit(0)

    if args.format == "tf":
        model.save(args.output, save_format="tf")
        print("Format: SavedModel, saved to {}".format(args.output))
    elif args.format == "h5":
        model.save(args.output, save_format="h5")
        print("Format: H5, saved to {}".format(args.output))