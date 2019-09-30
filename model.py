import tensorflow as tf


class OriginFeatureExtractor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=[3, 3],
                                            padding="same",
                                            activation=tf.nn.relu)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])

        self.conv2 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=[3, 3],
                                            padding="same",
                                            activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2])

        self.conv3 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[3, 3],
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=[3, 3],
                                            padding="same",
                                            activation=tf.nn.relu)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv5 = tf.keras.layers.Conv2D(filters=512,
                                            kernel_size=[3, 3],
                                            padding="same")
        self.bn5 = tf.keras.layers.BatchNormalization()

        self.conv6 = tf.keras.layers.Conv2D(filters=512,
                                            kernel_size=[3, 3],
                                            padding="same",
                                            activation=tf.nn.relu)
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv7 = tf.keras.layers.Conv2D(filters=512,
                                            kernel_size=[2, 2])
        self.bn7 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv6(x)
        x = self.pool6(x)

        x = self.conv7(x)
        x = self.bn7(x)
        outputs = tf.nn.relu(x)

        return outputs


class Decoder(tf.keras.Model):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
        self.dense1 = tf.keras.layers.Dense(units=num_classes)

    def call(self, inputs): # (None, width, 512)
        x = self.lstm1(inputs) # (None, width, 512)
        x = self.lstm2(x)
        outputs = self.dense1(x)
        return outputs


class CRNN(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = OriginFeatureExtractor()
        self.decoder = Decoder(256, num_classes)

    # @tf.function(input_signature=[tf.TensorSpec([None, 32, 100, 1], tf.float32), tf.TensorSpec([1], tf.bool)])
    def call(self, inputs, training=False):
        x = self.feature_extractor(inputs, training=training)
        x = tf.squeeze(x, 1)
        outputs = self.decoder(x)
        return outputs


if __name__ == "__main__":
    model = CRNN(10)
    x = tf.zeros([100, 32, 120, 1])
    y = model(x, training=False)
    
    model.feature_extractor.summary()
    model.decoder.summary()
    model.summary()

    print("Input shape is {}, y shape is: {}".format(x.shape, y.shape))