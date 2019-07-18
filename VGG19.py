from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D


class VGG19(Sequential):
    def __init__(self):
        super().__init__()

        self.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu'))
        self.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2), name="VGG19_Pool3"))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(2, 2)))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.add(Flatten())
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(1000, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        self.summary()
