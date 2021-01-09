import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing


class CTCGreedyDecoder(keras.layers.Layer):
    def __init__(self, vocabulary, merge_repeated=True, **kwargs):
        super().__init__(**kwargs)
        self.table = preprocessing.StringLookup(
            mask_token=None, vocabulary=vocabulary, invert=True)
        self.merge_repeated = merge_repeated
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        sequence_length = tf.fill([input_shape[0]], input_shape[1])
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]), 
            sequence_length,
            self.merge_repeated)
        x = self.table(decoded[0])
        x = tf.RaggedTensor.from_sparse(x)
        strings = tf.strings.reduce_join(x, axis=1)
        labels = tf.cast(decoded[0], tf.int32)
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=inputs,
            label_length=None,
            logit_length=sequence_length,
            logits_time_major=False,
            blank_index=-1)
        probability = tf.math.exp(-loss)
        return strings, probability


class CTCBeamSearchDecoder(keras.layers.Layer):
    def __init__(self, vocabulary, beam_width=100, top_paths=1, **kwargs):
        super().__init__(**kwargs)
        self.table = preprocessing.StringLookup(
            mask_token=None, vocabulary=vocabulary, invert=True)
        self.beam_width = beam_width
        self.top_paths = top_paths
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        decoded, log_probability = tf.nn.ctc_beam_search_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]), 
            tf.fill([input_shape[0]], input_shape[1]),
            self.beam_width, 
            self.top_paths)
        strings = []
        # TODO(hym) map?
        for i in range(self.top_paths):
            x = self.table(decoded[i])
            x = tf.RaggedTensor.from_sparse(x)
            x = tf.strings.reduce_join(x, axis=1, keepdims=True)
            strings.append(x)
        strings = tf.concat(strings, 1)
        probability = tf.math.exp(log_probability)
        return strings, probability