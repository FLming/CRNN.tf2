import argparse

import tensorflow as tf

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-p", "--saved_model_path", type=str, help="The SavedModel path.")
args = parser.parse_args()

def to_tflite():
    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_path)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

if __name__ == "__main__":
    to_tflite()