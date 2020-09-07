import argparse

from tensorflow import keras

from dataset_factory import DatasetBuilder
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--ann_paths', type=str, required=True, 
                    nargs='+', help='The paths of annnotation file.')
parser.add_argument('-t', '--table_path', type=str, required=True, 
                    help='The path of table file.')
parser.add_argument('-w', '--img_width', type=int, default=100, 
                    help='Image width, this parameter will affect the output '
                         'shape of the model, default is 100, so this model '
                         'can only predict up to 24 characters.')
parser.add_argument('-b', '--batch_size', type=int, default=256, 
                    help='Batch size.')
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='The saved model.')
parser.add_argument('--img_channels', type=int, default=1, 
                    help='0: Use the number of channels in the image, '
                         '1: Grayscale image, 3: RGB image')
parser.add_argument('--ignore_case', action='store_true', 
                    help='Whether ignore case.(default false)')
args = parser.parse_args()

dataset_builder = DatasetBuilder(args.table_path, args.img_width, 
                                 args.img_channels, args.ignore_case)
eval_ds = dataset_builder.build(args.ann_paths, args.batch_size, False)
model = keras.models.load_model(args.model, compile=False)
model.compile(loss=CTCLoss(), metrics=[WordAccuracy()])
model.evaluate(eval_ds)