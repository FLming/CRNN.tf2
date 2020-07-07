import argparse

from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='The saved model.')
parser.add_argument('-o', '--output', type=str, required=True, 
                    help='The output path.')
args = parser.parse_args()

model = keras.models.load_model(args.model, compile=False)
model = keras.Sequential([model, keras.layers.Softmax()])
model.save(args.output, include_optimizer=False)