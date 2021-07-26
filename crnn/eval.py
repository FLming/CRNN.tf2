import argparse
import pprint

import yaml

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import SequenceAccuracy, EditDistance
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, 
                    help='The config file path.')
parser.add_argument('--weight', type=str, required=True, 
                    help='The saved weight path.')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.Loader)['eval']
pprint.pprint(config)

dataset_builder = DatasetBuilder(**config['dataset_builder'])
ds = dataset_builder(config['ann_paths'], config['batch_size'], False)
model = build_model(dataset_builder.num_classes,
                    weight=args.weight,
                    img_shape=config['dataset_builder']['img_shape'])
model.compile(loss=CTCLoss(), metrics=[SequenceAccuracy(), EditDistance()])
model.evaluate(ds)