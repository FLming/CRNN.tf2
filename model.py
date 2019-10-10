import tensorflow as tf


def CRNN(num_classes):
    """Functional API."""
    inputs = tf.keras.Input(shape=(32, None, 1))

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2])(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2])(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[2, 2])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=tf.nn.relu)(x)

    x = tf.keras.layers.Reshape((-1, 512))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True))(x)

    outputs = tf.keras.layers.Dense(units=num_classes)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CRNN")

if __name__ == "__main__":
    model = CRNN(10)
    model.summary()