import argparse
from pathlib import Path

import yaml
import tensorflow as tf
from tensorflow import keras

import decoders

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--model', type=str, required=True, default='',
                    help='The saved model.')
parser.add_argument('--post', type=str, help='Post processing.')
parser.add_argument('--output', type=str, required=True, 
                    help='The output path.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

trained_model = keras.models.load_model(args.model, compile=False)
if args.post:
    img_input = keras.Input(shape=config['img_shape'])
    x = trained_model(img_input)
    if args.post == 'softmax':
        x = tf.nn.softmax(x, axis=-1)
    elif args.post == 'greedy':
        x = decoders.CTCGreedyDecoder(config['table_path'])(x)
    elif args.post == 'beam_search':
        x = decoders.CTCBeamSearchDecoder(config['table_path'])(x)
    model = keras.Model(inputs=img_input, outputs=x)
else:
    model = trained_model

model.summary()

model.save(args.output, include_optimizer=False)