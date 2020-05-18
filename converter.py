import argparse

from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, required=True, 
                    help="The saved model.")
parser.add_argument("-o", "--output", type=str, required=True, 
                    help="The output path.")
parser.add_argument("--save_format", type=str, default="tf",
                    help="Either 'tf' or 'h5', indicating whether to save "
                         "the model to Tensorflow SavedModel or HDF5. ")
args = parser.parse_args()

model = keras.models.load_model(args.model, compile=False)
model.save(args.output, include_optimizer=False, save_format=args.save_format)