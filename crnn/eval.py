import argparse

import yaml
from tensorflow import keras

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='The config file path.')
parser.add_argument('--model', type=str, required=True, 
                    help='The saved model path.')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)['eval']

dataset_builder = DatasetBuilder(**config['dataset_builder'])
ds = dataset_builder.build(config['ann_paths'], config['batch_size'], False)
model = keras.models.load_model(args.model, compile=False)
model.compile(loss=CTCLoss(), metrics=[WordAccuracy()])
model.evaluate(ds)