import argparse
import shutil
from pathlib import Path

import yaml
import tensorflow as tf
from tensorflow import keras

from dataset_factory import DatasetBuilder
from models import build_model
from losses import CTCLoss
from metrics import WordAccuracy
from callbacks import XTensorBoard

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--save_dir', type=Path, required=True,
                    help='The path to save the model, config file and logs')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['train']
print(config)

args.save_dir.mkdir(exist_ok=True)
if list(args.save_dir.iterdir()):
    raise ValueError(f'{args.save_dir} is not a empty folder')
shutil.copy(args.config, args.save_dir / args.config.name)
model_prefix = '{epoch}_{word_accuracy:.4f}_{val_word_accuracy:.4f}'
model_path = f'{args.save_dir}/{model_prefix}.h5'
strategy = tf.distribute.MirroredStrategy()
batch_size = config['batch_size_per_replica'] * strategy.num_replicas_in_sync

dataset_builder = DatasetBuilder(**config['dataset_builder'])
train_ds = dataset_builder.build(config['train_ann_paths'], batch_size, True)
val_ds = dataset_builder.build(config['val_ann_paths'], batch_size, False)

with strategy.scope():
    model = build_model(dataset_builder.num_classes, 
                        config['dataset_builder']['img_shape'])
    model.compile(optimizer=keras.optimizers.Adam(config['learning_rate']),
                  loss=CTCLoss(), metrics=[WordAccuracy()])

if config['restore']:
    model.load_weights(config['restore'], by_name=True, skip_mismatch=True)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(model_path),
    keras.callbacks.ReduceLROnPlateau(**config['reduce_lr']),
    XTensorBoard(log_dir=str(args.save_dir), **config['tensorboard'])]

model.fit(train_ds, epochs=config['epochs'], callbacks=callbacks,
          validation_data=val_ds)