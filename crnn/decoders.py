import tensorflow as tf
from tensorflow import keras


class CTCDecoder(keras.layers.Layer):
    def __init__(self, table_path, **kwargs):
        super().__init__(**kwargs)
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, 
            tf.string, tf.lookup.TextFileIndex.WHOLE_LINE), '')

    def detokenize(self, x):
        x = tf.RaggedTensor.from_sparse(x)
        x = tf.ragged.map_flat_values(self.table.lookup, x)
        strings = tf.strings.reduce_join(x, axis=1)
        return strings


class CTCGreedyDecoder(CTCDecoder):
    def __init__(self, table_path, merge_repeated=True, **kwargs):
        super().__init__(table_path, **kwargs)
        self.merge_repeated = merge_repeated
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        sequence_length = tf.fill([input_shape[0]], input_shape[1])
        decoded, neg_sum_logits = tf.nn.ctc_greedy_decoder(
            tf.transpose(inputs, perm=[1, 0, 2]), 
            sequence_length,
            self.merge_repeated)
        strings = self.detokenize(decoded[0])
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


class CTCBeamSearchDecoder(CTCDecoder):
    def __init__(self, table_path, beam_width=100, top_paths=1, **kwargs):
        super().__init__(table_path, **kwargs)
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
        for i in range(self.top_paths):
            strings.append(self.detokenize(decoded[i]))
        strings = tf.concat(strings, 1)
        probability = tf.math.exp(log_probability)
        return strings, probability