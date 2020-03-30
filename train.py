import os
import time
import argparse

import tensorflow as tf
from tensorflow import keras

from dataset import OCRDataLoader
from model import crnn
from losses import CTCLoss
from metrics import WordAccuracy


parser = argparse.ArgumentParser()
parser.add_argument("-ta", "--train_annotation_paths", type=str, 
                    required=True, nargs="+", 
                    help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, nargs="+", 
                    help="The path of val data annotation file.")
parser.add_argument("-tf", "--train_parse_funcs", type=str, required=True,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-vf", "--val_parse_funcs", type=str, nargs="+", 
                    help="The parse functions of annotaion files.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-e", "--epochs", type=int, default=20, 
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
os.makedirs("saved_models/{}".format(localtime))
saved_model_path = ("saved_models/{}/".format(localtime) + 
    "{epoch:03d}_{word_accuracy::.4f}_{val_word_accuracy:.4f}.h5")

model = crnn(train_dl.num_classes)
model.compile(optimizer=keras.optimizers.Adam(0.0001), loss=CTCLoss(),
              metrics=[WordAccuracy()])

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(saved_model_path),
    keras.callbacks.TensorBoard(log_dir="logs/{}".format(localtime), 
                                histogram_freq=1, profile_batch=0)
]

model.fit(train_dl(), epochs=args.epochs, callbacks=callbacks,
          validation_data=val_dl())