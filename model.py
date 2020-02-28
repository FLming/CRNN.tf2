from tensorflow.keras import Input, layers, Model

def original(input_tensor):
    """Related paper: http://arxiv.org/abs/1507.05717"""
    x = layers.Conv2D(
        filters=64, 
        kernel_size=3, 
        padding='same',
        activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(
        filters=128, 
        kernel_size=3, 
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(
        filters=128, 
        kernel_size=3, 
        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=256, 
        kernel_size=3, 
        padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(
        filters=512, 
        kernel_size=3, 
        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=512, 
        kernel_size=3, 
        padding='same')(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(
        filters=512, 
        kernel_size=2, 
        padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def identity_block(input_tensor, filters):
    x = layers.Conv2D(
        filters=filters[0], 
        kernel_size=3, 
        padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters[1], 
        kernel_size=3, 
        padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, filters):
    x = layers.Conv2D(
        filters=filters[0], 
        kernel_size=3, 
        padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters[1], 
        kernel_size=3, 
        padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(
        filters=filters[1], 
        kernel_size=1)(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def resnet(input_tensor):
    """Related paper: 
    http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf
    """
    x = layers.Conv2D(
        filters=32, 
        kernel_size=3,
        padding='same',
        activation='relu')(input_tensor)
    x = layers.Conv2D(
        filters=64, 
        kernel_size=3,
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = conv_block(x, (128, 128))

    x = layers.Conv2D(
        filters=128, 
        kernel_size=3,
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = conv_block(x, (256, 256))
    x = identity_block(x, (256, 256))

    x = layers.Conv2D(
        filters=256, 
        kernel_size=3,
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=[2, 1], padding='same')(x)

    x = conv_block(x, (512, 512))
    for i in range(4):
        x = identity_block(x, (512, 512))

    x = layers.Conv2D(
        filters=512, 
        kernel_size=3,
        padding='same',
        activation='relu')(x)

    x = conv_block(x, (512, 512))
    for i in range(2):
        x = identity_block(x, (512, 512))

    x = layers.Conv2D(
        filters=512, 
        kernel_size=2, 
        strides=(2, 1), 
        padding='same', 
        activation='relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=2, activation='relu')(x)
    return x

def crnn(num_classes, backbone='original'):
    img_input = Input(shape=(32, None, 1))

    if backbone.lower() == 'original':
        x = original(img_input)
    elif backbone.lower() == 'resnet':
        x = resnet(img_input)

    x = layers.Reshape((-1, 512))(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)

    x = layers.Dense(units=num_classes)(x)

    return Model(inputs=img_input, outputs=x, name='CRNN')

if __name__ == "__main__":
    model = crnn(10)
    model.summary()