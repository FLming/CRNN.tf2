import tensorflow as tf
from tensorflow import keras

class WordAccuracy(keras.metrics.Metric):
    """
    Calculate the word accuracy between y_true and y_pred.
    """
    def __init__(self, name='word_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, 
                                     initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, 
                                     initializer=tf.zeros_initializer())
                
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Maybe have more fast implementation.
        """
        b = tf.shape(y_true)[0]
        max_width = tf.maximum(tf.shape(y_true)[1], tf.shape(y_pred)[1])
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])        
        decoded, _ = tf.nn.ctc_greedy_decoder(
            inputs=tf.transpose(y_pred, perm=[1, 0, 2]),
            sequence_length=logit_length)
        y_true = tf.sparse.reset_shape(y_true, [b, max_width])
        y_pred = tf.sparse.reset_shape(decoded[0], [b, max_width])
        y_true = tf.sparse.to_dense(y_true, default_value=-1)
        y_pred = tf.sparse.to_dense(y_pred, default_value=-1)
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        values = tf.math.reduce_any(tf.math.not_equal(y_true, y_pred), axis=1)
        values = tf.cast(values, tf.int32)
        values = tf.reduce_sum(values)
        self.total.assign_add(b)
        self.count.assign_add(b - values)

    def result(self):
        return self.count / self.total

    def reset_states(self):
        self.count.assign(0)
        self.total.assign(0)