import time
import argparse

import numpy as np
import tensorflow as tf

from model import crnn
from dataset import OCRDataLoader, map_and_count

parser = argparse.ArgumentParser()
parser.add_argument("-ta", "--train_annotation_paths", type=str, required=True, 
                    nargs="+", help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, 
                    nargs="+", help="The path of val data annotation file.")
parser.add_argument("-tf", "--train_parse_funcs", type=str, required=True,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-vf", "--val_parse_funcs", type=str,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
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

def load_data():
    train_dl = OCRDataLoader(
        args.train_annotation_paths, 
        args.train_parse_funcs,
        args.image_width,
        args.table_path,
        args.batch_size,
        True)
    print("Num of training samples: {}".format(len(train_dl)))
    if args.val_annotation_paths:
        val_dl = OCRDataLoader(
            args.val_annotation_paths, 
            args.val_parse_funcs, 
            args.image_width,
            args.table_path,
            args.batch_size)
        print("Num of val samples: {}".format(len(val_dl)))
    print("Num of classes: {}".format(train_dl.num_classes))
    print("Blank index is {}".format(train_dl.blank_index))
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
            blank_index=-1)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def train(model, optimizer, dl, log_freq=10):
    avg_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    steps = round(len(dl) / args.batch_size) + 1
    for i, (x, y) in enumerate(dl()):
        loss = train_one_step(model, optimizer, x, y)
        avg_loss.update_state(loss)
        if tf.equal(optimizer.iterations % log_freq, 0):
            print("{:.0%} Trainging... loss: {:.6f}".format(
                (i + 1) / steps, avg_loss.result()), end='')
            tf.summary.scalar("loss", avg_loss.result(), 
                              step=optimizer.iterations)
            avg_loss.reset_states()
            if i < steps - 1:
                print(end='\r')
    print()

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
        blank_index=-1)
    loss = tf.reduce_mean(loss)
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)
    return loss, decoded

def val(model, dl, step):
    avg_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
    steps = round(len(dl) / args.batch_size) + 1
    total_cnt = 0
    for i, (x, y) in enumerate(dl()):
        loss, decoded = val_one_step(model, x, y)
        avg_loss.update_state(loss)
        cnt = map_and_count(decoded, y, dl.inv_table, dl.blank_index)
        total_cnt += cnt
        print("{:.0%} Valuating...".format((i + 1) / steps), end='')
        if i < steps - 1:
            print(end='\r')
    accuracy = total_cnt / len(dl)
    print("Total: {}, Accuracy: {:.2%}".format(total_cnt, accuracy))
    tf.summary.scalar("loss", avg_loss.result(), step=step)
    tf.summary.scalar("accuracy", accuracy, step=step)
    avg_loss.reset_states()


if __name__ == "__main__":
    train_dl, val_dl = load_data()
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start at {}".format(localtime))

    model = crnn(train_dl.num_classes)
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

    for epoch in range(1, args.epochs + 1):
        print("Epoch {}:".format(epoch))
        with train_summary_writer.as_default():
            train(model, optimizer, train_dl)
        if not (epoch - 1) % args.save_freq:
            checkpoint_path = manager.save(optimizer.iterations)
            print("Model saved to {}".format(checkpoint_path))
            if val_dl is not None:
                with val_summary_writer.as_default():
                    val(model, val_dl, optimizer.iterations)