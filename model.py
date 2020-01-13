import tensorflow as tf
from tensorflow import keras

def VGG(inputs):
    """Related paper: http://arxiv.org/abs/1507.05717"""
    x = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", 
                            activation=tf.nn.relu)(inputs)
    x = keras.layers.MaxPool2D(pool_size=[2, 2], padding="same")(x)

    x = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", 
                            activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2], padding="same")(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=tf.nn.relu)(x)

    x = keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", 
                            activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], 
                               padding="same")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=tf.nn.relu)(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", 
                            activation=tf.nn.relu)(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 1], strides=[2, 1], 
                               padding="same")(x)

    x = keras.layers.Conv2D(filters=512, kernel_size=[2, 2])(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation=tf.nn.relu)(x)

    return x

def identity_block(input_tensor, filters):
    filters1, filters2 = filters

    x = keras.layers.Conv2D(filters=filters1, kernel_size=[3, 3], 
                            padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(filters=filters2, kernel_size=[3, 3], 
                            padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation("relu")(x)
    return x

def conv_block(input_tensor, filters):
    filters1, filters2 = filters

    x = keras.layers.Conv2D(filters=filters1, kernel_size=[3, 3], 
                            padding="same")(input_tensor)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(filters=filters2, kernel_size=[3, 3], 
                            padding="same")(x)
    x = keras.layers.BatchNormalization()(x)

    shortcut = keras.layers.Conv2D(filters=filters2, 
                                   kernel_size=[1, 1])(input_tensor)
    shortcut = keras.layers.BatchNormalization()(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation("relu")(x)
    return x

def ResNet(inputs):
    """Related paper: 
    http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf
    """
    x = keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same", 
                            activation="relu")(inputs)
    x = keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", 
                            activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2])(x)

    x = conv_block(x, [128, 128])

    x = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", 
                            activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2])(x)

    x = conv_block(x, [256, 256])
    x = identity_block(x, [256, 256])

    x = keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", 
                            activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1], 
                               padding="same")(x)

    x = conv_block(x, [512, 512])
    for i in range(4):
        x = identity_block(x, [512, 512])

    x = keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", 
                            activation="relu")(x)

    x = conv_block(x, [512, 512])
    for i in range(2):
        x = identity_block(x, [512, 512])

    x = keras.layers.Conv2D(filters=512, kernel_size=[2, 2], strides=[2, 1], 
                            padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(filters=512, kernel_size=[2, 2], 
                            activation="relu")(x)

    return x

def CRNN(num_classes, backbone="VGG"):
    inputs = keras.Input(shape=(32, None, 1))

    if backbone == "VGG":
        x = VGG(inputs)
    elif backbone == "ResNet":
        x = ResNet(inputs)

    x = keras.layers.Reshape((-1, 512))(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True))(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units=256, return_sequences=True))(x)

    outputs = keras.layers.Dense(units=num_classes)(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="CRNN")

if __name__ == "__main__":
    model = CRNN(10, backbone="ResNet")
    model.summary()