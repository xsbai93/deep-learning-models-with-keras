from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import add, Flatten


class ResNet50(Model):
    def __init__(self):
        super().__init__()

        input = Input(shape=(224, 224, 3))
        layer = ZeroPadding2D((3, 3))(input)
        layer = self.Conv2d_BN(layer, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

        layer = self.Conv_Block(layer, nb_filter=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), with_conv_shortcut=True)
        layer = self.Conv_Block(layer, nb_filter=[64, 64, 256], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[64, 64, 256], kernel_size=(3, 3))

        layer = self.Conv_Block(layer, nb_filter=[128, 128, 512], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        layer = self.Conv_Block(layer, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[128, 128, 512], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[128, 128, 512], kernel_size=(3, 3))

        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[256, 256, 1024], kernel_size=(3, 3))

        layer = self.Conv_Block(layer, nb_filter=[512, 512, 2048], kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        layer = self.Conv_Block(layer, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        layer = self.Conv_Block(layer, nb_filter=[512, 512, 2048], kernel_size=(3, 3))
        layer = AveragePooling2D(pool_size=(7, 7))(layer)
        layer = Flatten()(layer)
        layer = Dense(1000, activation='softmax')(layer)

        model = Model(inputs=input, outputs=layer)
        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        model.summary()

    def Conv2d_BN(self, layer, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        layer = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(layer)
        layer = BatchNormalization(axis=3, name=bn_name)(layer)
        return layer

    def Conv_Block(self, input, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        layer = self.Conv2d_BN(input, nb_filter=nb_filter[0], kernel_size=(1, 1), strides=strides, padding='same')
        layer = self.Conv2d_BN(layer, nb_filter=nb_filter[1], kernel_size=(3, 3), padding='same')
        layer = self.Conv2d_BN(layer, nb_filter=nb_filter[2], kernel_size=(1, 1), padding='same')
        if with_conv_shortcut:
            shortcut = self.Conv2d_BN(input, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size)
            layer = add([layer, shortcut])
            return layer
        else:
            layer = add([layer, input])
            return layer
