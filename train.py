import argparse
import time

import numpy as np
import tensorflow as tf

from model import CRNN
from dataset import OCRDataLoader, map_and_count

parser = argparse.ArgumentParser()
parser.add_argument("-ta", "--train_annotation_paths", type=str, required=True, help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, help="The path of val data annotation file.")
parser.add_argument("-t", "--table_path", type=str, required=True, help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("-e", "--epochs", type=int, default=5, help="Num of epochs to train.")
parser.add_argument("-r", "--learning_rate", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--checkpoint", type=str, help="The checkpoint path. (Restore)")
parser.add_argument("--max_to_keep", type=int, default=5, help="Max num of checkpoint to keep.")
parser.add_argument("--save_freq", type=int, default=1, help="Save and validate interval.")
parser.add_argument("--image_height", type=int, default=32, help="Image height(32). If you change this, you should change the structure of CNN.")
parser.add_argument("--backbone", type=str, default="VGG", help="The backbone of CRNNs, available now is VGG and ResNet.")
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

@tf.function
def val_one_step(model, X, Y):
    y_pred = model(X, training=False)
    loss = tf.nn.ctc_loss(labels=Y,
                          logits=tf.transpose(y_pred, perm=[1, 0, 2]),
                          label_length=None,
                          logit_length=[y_pred.shape[1]]*y_pred.shape[0],
                          blank_index=BLANK_INDEX)
    loss = tf.reduce_mean(loss)
    decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
                                                       sequence_length=[y_pred.shape[1]]*y_pred.shape[0],
                                                       merge_repeated=True)
    return decoded, loss

if __name__ == "__main__":
    train_dataloader = OCRDataLoader(args.train_annotation_paths, 
                                     args.image_height, 
                                     args.image_width, 
                                     table_path=args.table_path,
                                     blank_index=BLANK_INDEX,
                                     shuffle=True, 
                                     batch_size=args.batch_size)
    print("Num of training samples: {}".format(len(train_dataloader)))
    if args.val_annotation_paths:
        val_dataloader = OCRDataLoader(args.val_annotation_paths,
                                       args.image_height,
                                       args.image_width,
                                       table_path=args.table_path,
                                       blank_index=BLANK_INDEX,
                                       batch_size=args.batch_size)
        print("Num of val samples: {}".format(len(val_dataloader)))
    print("Num of classes: {}".format(NUM_CLASSES))
    print("Blank index is {}".format(BLANK_INDEX))
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start at {}".format(localtime))

    model = CRNN(NUM_CLASSES, args.backbone)
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                                 decay_steps=100000,
                                                                 decay_rate=0.96,
                                                                 staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if args.checkpoint:
        localtime = args.checkpoint.rstrip("/").split("/")[-1]
    manager = tf.train.CheckpointManager(checkpoint, directory="./tf_ckpts/{}".format(localtime), max_to_keep=args.max_to_keep)
    summary_writer = tf.summary.create_file_writer("./logs/{}".format(localtime))
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch")

    avg_loss = tf.keras.metrics.Mean(name="train_loss")
    val_avg_loss = tf.keras.metrics.Mean(name="val_loss")

    for epoch in range(1, args.epochs + 1):
        with summary_writer.as_default():
            for X, Y in train_dataloader():
                loss = train_one_step(model, X, Y, optimizer)
                tf.summary.scalar("train_loss", loss, step=optimizer.iterations)
                avg_loss.update_state(loss)
            print("[{} / {}] Mean train loss: {}".format(epoch, args.epochs, avg_loss.result()))
            avg_loss.reset_states()
            if (epoch - 1) % args.save_freq == 0:
                saved_path = manager.save(checkpoint_number=epoch)
                print("Model saved to {}".format(saved_path))
                if args.val_annotation_paths:
                    num_correct_samples = 0
                    for X, Y in val_dataloader():
                        decoded, loss = val_one_step(model, X, Y)
                        count = map_and_count(decoded, Y, INT_TO_CHAR)
                        val_avg_loss.update_state(loss)
                        num_correct_samples += count
                    tf.summary.scalar("val_loss", val_avg_loss.result(), step=epoch)
                    tf.summary.scalar("accuracy(line, greedy decoder)", num_correct_samples / len(val_dataloader), step=epoch)
                    print("[{} / {}] Mean val loss: {}".format(epoch, args.epochs, val_avg_loss.result()))
                    print("[{} / {}] Accuracy(line, greedy decoder): {:.2f}".format(epoch, args.epochs, num_correct_samples / len(val_dataloader)))
                    val_avg_loss.reset_states()