import argparse
import time

import numpy as np
import tensorflow as tf

from model import CRNN
from dataset import OCRDataLoader

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-ta", "--train_annotation_path", type=str, required=True, help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_path", type=str, help="The path of val data annotation file.")
parser.add_argument("-t", "--table_path", type=str, required=True, help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("-e", "--epochs", type=int, default=5, help="Num of epochs to train.")
parser.add_argument("-r", "--learning_rate", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--checkpoint", type=str, help="The checkpoint path. (Restore)")
parser.add_argument("--max_to_keep", type=int, default=5, help="Max num of checkpoint to keep.")
parser.add_argument("--val_freq", type=int, default=1, help="Val interval.")
parser.add_argument("--save_freq", type=int, default=1, help="Saved interval.")
parser.add_argument("--image_height", type=int, default=32, help="Image height(32). If you change this, you should change the structure of CNN.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

@tf.function
def train_one_step(model, X, Y, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss = tf.nn.ctc_loss(labels=Y,
                              logits=tf.transpose(y_pred, perm=[1, 0, 2]),
                              label_length=None,
                              logit_length=[y_pred.shape[1]]*y_pred.shape[0],
                              blank_index=BLANK_INDEX)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    return loss

if __name__ == "__main__":
    train_dataloader = OCRDataLoader(args.train_annotation_path, 
                                     args.image_height, 
                                     args.image_width, 
                                     table_path=args.table_path,
                                     blank_index=BLANK_INDEX,
                                     shuffle=True, 
                                     batch_size=args.batch_size)
    print("Num of training samples: {}.".format(len(train_dataloader)))
    if args.val_annotation_path:
        val_dataloader = OCRDataLoader(args.val_annotation_path,
                                       args.image_height,
                                       args.image_width,
                                       table_path=args.table_path,
                                       blank_index=BLANK_INDEX,
                                       batch_size=args.batch_size)
        print("Num of val samples: {}.".format(len(val_dataloader)))
    print("Num of classes: {}.".format(NUM_CLASSES))
    print("Blank index is {}.".format(BLANK_INDEX))
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start at {}.".format(localtime))

    model = CRNN(NUM_CLASSES)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if args.checkpoint:
        localtime = args.checkpoint.rstrip("/").split("/")[-1]
    manager = tf.train.CheckpointManager(checkpoint, directory="./tf_ckpts/{}".format(localtime), max_to_keep=args.max_to_keep)
    summary_writer = tf.summary.create_file_writer("./logs/{}".format(localtime))
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}.".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    avg_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)

    for epoch in range(1, args.epochs + 1):
        with summary_writer.as_default():
            for X, Y in train_dataloader():
                loss = train_one_step(model, X, Y, optimizer)
                tf.summary.scalar("loss", loss, step=optimizer.iterations)
                avg_loss.update_state(loss)
            print("[{} / {}] Mean loss: {}.".format(epoch, args.epochs, avg_loss.result()))
            avg_loss.reset_states()
            if (epoch - 1) % args.save_freq == 0:
                saved_path = manager.save(checkpoint_number=epoch)
                print("Model saved to {}.".format(saved_path))
            