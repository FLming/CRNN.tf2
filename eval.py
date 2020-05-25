import argparse

from tensorflow import keras

from dataset import build_dataset
from losses import CTCLoss
from metrics import WordAccuracy

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotation_paths", type=str, required=True, 
                    nargs="+", help="The paths of annnotation file.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width, this parameter will affect the output "
                         "shape of the model, default is 100, so this model "
                         "can only predict up to 24 characters.")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-m", "--model", type=str, required=True, 
                    help="The saved model.")
parser.add_argument("--channels", type=int, default=1, help="Image channels, "
                    "0: Use the number of channels in the image, "
                    "1: Grayscale image, "
                    "3: RGB image")
parser.add_argument("--ignore_case", action="store_true", 
                    help="Whether ignore case.(default false)")
args = parser.parse_args()

eval_ds, size, num_classes = build_dataset(
    args.annotation_paths,
    args.table_path,
    args.image_width,
    args.channels,
    args.ignore_case,
    batch_size=args.batch_size)
print("Num of eval samples: {}".format(size))

model = keras.models.load_model(args.model, compile=False)
model.summary()
model.compile(loss=CTCLoss(), metrics=[WordAccuracy()])
model.evaluate(eval_ds)