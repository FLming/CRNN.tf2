import argparse

from tensorflow import keras

from dataset import OCRDataLoader
from losses import CTCLoss
from metrics import WordAccuracy


parser = argparse.ArgumentParser()
parser.add_argument("-a", "--annotation_paths", type=str, required=True, 
                    nargs="+", help="The paths of annnotation file.")
parser.add_argument("-f", "--parse_funcs", type=str, required=True,
                    nargs="+", help="The parse functions of annotaion files.")
parser.add_argument("-t", "--table_path", type=str, required=True, 
                    help="The path of table file.")
parser.add_argument("-w", "--image_width", type=int, default=100, 
                    help="Image width(>=16).")
parser.add_argument("-b", "--batch_size", type=int, default=256, 
                    help="Batch size.")
parser.add_argument("-m", "--model", type=str, required=True, 
                    help="The saved model path.")
args = parser.parse_args()

eval_dl = OCRDataLoader(
    args.annotation_paths, 
    args.parse_funcs, 
    args.image_width,
    args.table_path, 
    args.batch_size)

print("Num of val samples: {}".format(len(eval_dl)))

model = keras.models.load_model(
    args.model, 
    custom_objects={'CTCLoss': CTCLoss, 'WordAccuracy': WordAccuracy})

model.summary()

model.evaluate(eval_dl())