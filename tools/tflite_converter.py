import argparse

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, 
                    help='The saved model.')
parser.add_argument('-o', '--output', type=str, required=True, 
                    help='The output path.')
args = parser.parse_args()

model = tf.saved_model.load(args.model)
concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 32, 100, 1])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TF Lite model.
with tf.io.gfile.GFile(args.output, 'wb') as f:
    f.write(tflite_model)