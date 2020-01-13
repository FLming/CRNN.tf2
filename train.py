import argparse
import time

import numpy as np
import tensorflow as tf

import arg
from model import CRNN
from dataset import OCRDataLoader, map_and_count

parser = argparse.ArgumentParser(parents=[arg.parser])
parser.add_argument("-ta", "--train_annotation_paths", type=str, required=True, 
                    help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, 
                    help="The path of val data annotation file.")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-e", "--epochs", type=int, default=5, 
                    help="Num of epochs to train.")
parser.add_argument("-r", "--learning_rate", type=float, default=0.001, 
                    help="Learning rate.")
parser.add_argument("--checkpoint", type=str, 
                    help="The checkpoint path. (Restore)")
parser.add_argument("--max_to_keep", type=int, default=5, 
                    help="Max num of checkpoint to keep.")
parser.add_argument("--save_freq", type=int, default=1, 
                    help="Save and validate interval.")
args = parser.parse_args()

with open(args.table_path, "r") as f:
    INT_TO_CHAR = [char.strip() for char in f]
NUM_CLASSES = len(INT_TO_CHAR)
BLANK_INDEX = NUM_CLASSES - 1 # Make sure the blank index is what.

def dataloader():
    train_dl = OCRDataLoader(
        args.train_annotation_paths, 
        args.image_height, 
        args.image_width, 
        table_path=args.table_path,
        blank_index=BLANK_INDEX,
        shuffle=True, 
        batch_size=args.batch_size)
    print(f"Num of training samples: {len(train_dl)}")
    if args.val_annotation_paths:
        val_dl = OCRDataLoader(
            args.val_annotation_paths,
            args.image_height,
            args.image_width,
            table_path=args.table_path,
            blank_index=BLANK_INDEX,
            batch_size=args.batch_size)
        print(f"Num of val samples: {len(val_dl)}")
    else:
        val_dl = None
    print(f"Num of classes: {NUM_CLASSES}")
    print(f"Blank index is {BLANK_INDEX}")
    return train_dl, val_dl

@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        loss = tf.nn.ctc_loss(
            labels=y,
            logits=logits,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=BLANK_INDEX)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_one_step(model, x, y):
    logits = model(x, training=False)
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    loss = tf.nn.ctc_loss(
        labels=y,
        logits=logits,
        label_length=None,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=BLANK_INDEX)
    loss = tf.reduce_mean(loss)
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    return loss, decoded

if __name__ == "__main__":
    train_dl, val_dl = dataloader()
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(f"Start at {localtime}")

    model = CRNN(NUM_CLASSES, args.backbone)
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args.learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if args.checkpoint:
        localtime = args.checkpoint.rstrip("/").split("/")[-1]
    manager = tf.train.CheckpointManager(
        checkpoint, 
        directory="tf_ckpts/{}".format(localtime), 
        max_to_keep=args.max_to_keep)
    
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch")

    train_summary_writer = tf.summary.create_file_writer(
        f"logs/{localtime}/train")
    val_summary_writer = tf.summary.create_file_writer(
        f"logs/{localtime}/val")

    avg_loss = tf.keras.metrics.Mean(name="loss")

    for epoch in range(1, args.epochs + 1):
        with train_summary_writer.as_default():
            for x, y in train_dl():
                loss = train_one_step(model, optimizer, x, y)
                avg_loss.update_state(loss)
                tf.summary.scalar("loss", loss, optimizer.iterations)
            tf.summary.scalar("avg_loss", avg_loss.result(), epoch)
        avg_loss.reset_states()
        if not (epoch - 1) % args.save_freq:
            checkpoint_path = manager.save(checkpoint_number=epoch)
            print(f"Model saved to {checkpoint_path}")
            if val_dl is not None:
                num_correct_samples = 0
                with val_summary_writer.as_default():
                    for x, y in val_dl():
                        loss, decoded = val_one_step(model, x, y)
                        cnt = map_and_count(decoded, y, INT_TO_CHAR)
                        avg_loss.update_state(loss)
                        num_correct_samples += cnt
                    tf.summary.scalar("avg_loss", avg_loss.result(), epoch)
                    tf.summary.scalar("val_accuracy(line, greedy decoder)",
                                      num_correct_samples / len(val_dl) * 100,
                                      step=epoch)
                avg_loss.reset_states()
        