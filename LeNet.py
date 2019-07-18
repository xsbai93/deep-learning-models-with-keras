from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,AveragePooling2D

class LeNet(Sequential):
    def __init__(self):
        super().__init__()

        self.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), input_shape=(32, 32, 1), padding="same", activation='tanh'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        self.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
        self.add(Flatten())
        self.add(Dense(84, activation='tanh'))
        self.add(Dense(10, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        self.summary()
