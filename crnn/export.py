import argparse
from pathlib import Path

import yaml
from tensorflow import keras

from models import build_model
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True, 
                    help='The config file path.')
parser.add_argument('--weight', type=str, required=True, default='',
                    help='The saved weight path.')
parser.add_argument('--pre', type=str, help='pre processing.')
parser.add_argument('--post', type=str, help='Post processing.')
parser.add_argument('--output', type=str, required=True, 
                    help='The output path.')
args = parser.parse_args()

with args.config.open() as f:
    config = yaml.load(f, Loader=yaml.Loader)['dataset_builder']

with open(config['table_path']) as f:
    num_classes = len(f.readlines())

if args.pre == 'rescale':
    preprocess = keras.layers.experimental.preprocessing.Rescaling(1./255)
else:
    preprocess = None

if args.post == 'softmax':
    postprocess = keras.layers.Softmax()
elif args.post == 'greedy':
    postprocess = CTCGreedyDecoder(config['table_path'])
elif args.post == 'beam_search':
    postprocess = CTCBeamSearchDecoder(config['table_path'])
else:
    postprocess = None

model = build_model(num_classes, 
                    weight=args.weight,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    img_shape=config['img_shape'])
model.summary()
model.save(args.output)
