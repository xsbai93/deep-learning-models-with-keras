from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D


class GoogLeNet(Model):
    def __init__(self):
        super().__init__()

        input = Input(shape=(224, 224, 3))
        layer = self.Conv2d_BN(input, 64, (7, 7), strides=(2, 2), padding='same')
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = self.Conv2d_BN(layer, 192, (3, 3), strides=(1, 1), padding='same')
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = self.Inception(layer, 64)  # 256
        layer = self.Inception(layer, 120)  # 480
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = self.Inception(layer, 128)  # 512
        layer = self.Inception(layer, 128)
        layer = self.Inception(layer, 128)
        layer = self.Inception(layer, 132)  # 528
        layer = self.Inception(layer, 208)  # 832
        layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)
        layer = self.Inception(layer, 208)
        layer = self.Inception(layer, 256)  # 1024
        layer = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(layer)
        layer = Dropout(0.4)(layer)
        layer = Dense(1000, activation='relu')(layer)
        layer = Dense(1000, activation='softmax')(layer)

        model = Model(input, layer, name='Inception')

        model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        model.summary()

    def Conv2d_BN(self, layer, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        layer = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(
            layer)
        layer = BatchNormalization(axis=3, name=bn_name)(layer)

        return layer

    def Inception(self, layer, nb_filter):
        branch1x1 = self.Conv2d_BN(layer, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branch3x3 = self.Conv2d_BN(layer, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch3x3 = self.Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)

        branch5x5 = self.Conv2d_BN(layer, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch5x5 = self.Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        branchpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(layer)
        branchpool = self.Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)

        layer = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)

        return layer
