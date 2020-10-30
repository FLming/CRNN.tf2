from tensorflow import keras


class XTensorBoard(keras.callbacks.TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': self.model.optimizer.lr})
        super().on_epoch_end(epoch, logs)