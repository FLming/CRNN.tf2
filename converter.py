import sys
import argparse

import tensorflow as tf

from model import crnn

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-c", "--checkpoint", type=str, required=True, 
                    help="The checkpoint path.")
parser.add_argument("-f", "--format", type=str, choices=("tf", "h5"), 
                    required=True, help="Format H5 or SavedModel(tf).")
parser.add_argument("-o", "--output", type=str, required=True, 
                    help="The output path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    inv_table = [char.strip() for char in f]
num_classes = len(inv_table)

if __name__ == "__main__":
    model = crnn(num_classes)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint))
    if tf.train.latest_checkpoint(args.checkpoint):
        print("Restored from {}".format(
            tf.train.latest_checkpoint(args.checkpoint)))
    else:
        print("Initializing fail, check checkpoint")
        sys.exit()

    model.save(args.output, save_format=args.format)
    print(f"Format: {args.format}, saved to {args.output}")