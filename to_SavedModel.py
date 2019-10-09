import argparse

import tensorflow as tf

from model import CRNN

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-t", "--table_path", type=str, help="The path of table file.")
parser.add_argument("--checkpoint_path", type=str, help="The checkpoint path.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    int_to_char = [char.strip() for char in f]
num_classes = len(int_to_char)
blank_index = num_classes - 1 # Make sure the blank index is what.

def to_SavedModel(num_classes):
    model = CRNN(num_classes)
    model.build(input_shape=(1, 100, 32, 1))
    model.summary()
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(args.checkpoint_path))
    tf.saved_model.save(model, "SavedModel/1/")

if __name__ == "__main__":
    to_SavedModel(num_classes)