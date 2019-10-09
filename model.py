import tensorflow as tf


def CRNN(num_classes):
    """Sequential version 1."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32, None, 1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),

        tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2]),

        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.nn.relu),

        tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1]),

        tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.nn.relu),
        
        tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1]),

        tf.keras.layers.Conv2D(filters=512, kernel_size=[2, 2]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(activation=tf.nn.relu),

        tf.keras.layers.Reshape((-1, 512)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True)),
        tf.keras.layers.Dense(units=num_classes)
    ])

    return model

if __name__ == "__main__":
    model = CRNN(10)
    x = tf.zeros([100, 32, 20, 1])
    y = model(x, training=False)

    model.summary()

    print("Input shape is {}, y shape is: {}".format(x.shape, y.shape))