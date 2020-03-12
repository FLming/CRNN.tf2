import time
import os
import argparse
import functools

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from dataset import OCRDataLoader
from model import crnn
from metrics import WordAccuracy

def train_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    with tf.GradientTape() as tape:
        logits = self(x, training=True)
        logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        loss = tf.nn.ctc_loss(
            labels=y,
            logits=logits,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=-1)
        loss = tf.reduce_mean(loss)
    trainable_variables = self.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(grads, trainable_variables))

    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)

    self.compiled_metrics.update_state(y, decoded[0], sample_weight)
    return {**{m.name: m.result() for m in self.metrics}, 'loss': loss}

def test_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    logits = self(x, training=False)
    logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    loss = tf.nn.ctc_loss(
        labels=y,
        logits=logits,
        label_length=None,
        logit_length=logit_length,
        logits_time_major=False,
        blank_index=-1)
    loss = tf.reduce_mean(loss)

    decoded, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, perm=[1, 0, 2]),
        sequence_length=logit_length,
        merge_repeated=True)

    self.compiled_metrics.update_state(y, decoded[0], sample_weight)
    return {**{m.name: m.result() for m in self.metrics}, 'loss': loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ta", "--train_annotation_paths", type=str, 
                        required=True, nargs="+", 
                        help="The path of training data annnotation file.")
    parser.add_argument("-va", "--val_annotation_paths", type=str, nargs="+", 
                        help="The path of val data annotation file.")
    parser.add_argument("-tf", "--train_parse_funcs", type=str, required=True,
                        nargs="+", 
                        help="The parse functions of annotaion files.")
    parser.add_argument("-vf", "--val_parse_funcs", type=str, nargs="+", 
                        help="The parse functions of annotaion files.")
    parser.add_argument("-t", "--table_path", type=str, required=True, 
                        help="The path of table file.")
    parser.add_argument("-w", "--image_width", type=int, default=100, 
                        help="Image width(>=16).")
    parser.add_argument("-b", "--batch_size", type=int, default=256, 
                        help="Batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=5, 
                        help="Num of epochs to train.")
    args = parser.parse_args()

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
    else:
        val_dl = lambda: None
    
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print("Start at {}".format(localtime))
    os.makedirs('h5/{}'.format(localtime), exist_ok=True)

    model = crnn(train_dl.num_classes)
    model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    callbacks = [
        # In my computer, I don't know why use other format to save model is 
        # slower than h5 format, so I use h5 format to save model.
        tf.keras.callbacks.ModelCheckpoint("h5/{}/".format(localtime) + 
            "{epoch:03d}-{val_loss:.2f}-{val_word_accuracy:.2f}.h5"),
        tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(localtime), 
                                       histogram_freq=1, profile_batch=0)
    ]

    model.compile(optimizer=optimizer, metrics=[WordAccuracy()])
    model.train_step = functools.partial(train_step, model)
    model.test_step = functools.partial(test_step, model)
    model.fit(train_dl(), epochs=args.epochs, callbacks=callbacks, 
              validation_data=val_dl())