import argparse
import time
from pathlib import Path

import yaml
import tensorflow as tf
from tensorflow import keras

from dataset_factory import DatasetBuilder
from model import build_model
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='The config file path.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='The path to save the model and log')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']

localtime = time.asctime()
model_dir = Path(args.model_dir, localtime)
model_dir.mkdir()
model_prefix = '{epoch}_{word_accuracy:.4f}_{val_word_accuracy:.4f}'
saved_model_path = f'{model_dir}/{model_prefix}.h5'
strategy = tf.distribute.MirroredStrategy()
batch_size = config['batch_size_per_replica'] * strategy.num_replicas_in_sync
print('Training start at {}'.format(localtime))

dataset_builder = DatasetBuilder(**config['dataset_builder'])
train_ds = dataset_builder.build(config['train_ann_paths'], batch_size, True)
val_ds = dataset_builder.build(config['val_ann_paths'], batch_size, False)

with strategy.scope():
    model = build_model(dataset_builder.num_classes, 
                        img_channels=config['dataset_builder']['img_channels'])
    model.compile(optimizer=keras.optimizers.Adam(config['learning_rate']),
                  loss=CTCLoss(), metrics=[WordAccuracy()])

if config['restore']:
    model.load_weights(config['restore'], by_name=True, skip_mismatch=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(saved_model_path),
    keras.callbacks.ReduceLROnPlateau(monitor='val_word_accuracy', mode='max',
                                      **config['reduce_lr']),
    keras.callbacks.TensorBoard(log_dir=str(model_dir), 
                                **config['tensorboard'])]

model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks,
          validation_data=val_ds)