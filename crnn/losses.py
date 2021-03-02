import tensorflow as tf
from tensorflow import keras


class CTCLoss(keras.losses.Loss):
    """ A class that wraps the function of tf.nn.ctc_loss. 
    
    Attributes:
        logits_time_major: If False (default) , shape is [batch, time, logits], 
            If True, logits is shaped [time, batch, logits]. 
        blank_index: Set the class index to use for the blank label. default is
            -1 (num_classes - 1). 
    """

    def __init__(self, logits_time_major=False, blank_index=-1, 
                 name='ctc_loss'):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index

    def call(self, y_true, y_pred):
        """ Computes CTC (Connectionist Temporal Classification) loss. work on
        CPU, because y_true is a SparseTensor.
        """
        y_true = tf.cast(y_true, tf.int32)
        y_pred_shape = tf.shape(y_pred)
        logit_length = tf.fill([y_pred_shape[0]], y_pred_shape[1])
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index)
        return tf.math.reduce_mean(loss)