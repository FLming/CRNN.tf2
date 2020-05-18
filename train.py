import argparse
import time
import os

import tensorflow
from tensorflow import keras

from dataset import build_dataset
from model import build_model
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument("-ta", "--train_annotation_paths", type=str, 
                    required=True, nargs="+", 
                    help="The path of training data annnotation file.")
parser.add_argument("-va", "--val_annotation_paths", type=str, nargs="+", 
                    help="The path of val data annotation file.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width, this parameter will affect the output "
                         "shape of the model, default is 100, so this model "
                         "can only predict up to 24 characters.")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, 
                    help="Learning rate.")
parser.add_argument("-e", "--epochs", type=int, default=20, 
                    help="Num of epochs to train.")
parser.add_argument("--channels", type=int, default=1, help="Image channels, "
                    "0: Use the number of channels in the image, "
                    "1: Grayscale image, "
                    "3: RGB image")
args = parser.parse_args()

localtime = time.asctime()
train_ds, size, num_classes = build_dataset(
    args.train_annotation_paths,
    args.image_width,
    args.table_path,
    shuffle=True,
    batch_size=args.batch_size,
    channels=args.channels)
print("Num of training samples: {}".format(size))
saved_model_prefix = "{epoch:03d}_{word_accuracy:.4f}"
if args.val_annotation_paths:
    val_ds, size, num_classes = build_dataset(
        args.val_annotation_paths,
        args.image_width,
        args.table_path,
        batch_size=args.batch_size,
        channels=args.channels)
    print("Num of val samples: {}".format(size))
    saved_model_prefix = saved_model_prefix + "_{val_word_accuracy:.4f}"
else:
    val_ds = None
saved_model_path = ("saved_models/{}/".format(localtime) + 
                    saved_model_prefix + ".h5")
os.makedirs("saved_models/{}".format(localtime))
print("Training start at {}".format(localtime))

model = build_model(num_classes, channels=args.channels)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
              loss=CTCLoss(), metrics=[WordAccuracy()])
callbacks = [keras.callbacks.ModelCheckpoint(saved_model_path),
             keras.callbacks.TensorBoard(log_dir="logs/{}".format(localtime))]

model.fit(train_ds, epochs=args.epochs, callbacks=callbacks,
          validation_data=val_ds)