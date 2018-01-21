import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from os.path import exists, basename, splitext, join

class image_gender_classifier:
    # dimensions of our images.
    __IMG_WIDTH, __IMG_HEIGHT = 200, 200
    __WEIGHTS_PATH = '/data/gender.weights'
    __BATCH_SIZE = 16

    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        if K.image_data_format() == 'channels_first':
            input_shape = (3, self.__IMG_WIDTH, self.__IMG_HEIGHT)
        else:
            input_shape = (self.__IMG_WIDTH, self.__IMG_HEIGHT, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        if exists(self.__WEIGHTS_PATH):
            model.load_weights(self.__WEIGHTS_PATH)
        else:
            print('Gender model weights not found.')
            model = None
        return model

    def test(self, **kwargs):
        prediction = None
        input_dir = kwargs['input_dir']

        model = self.__get_model()
        if (model == None): return {}

        test_datagen = ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_directory(
            input_dir,
            target_size=(self.__IMG_WIDTH, self.__IMG_HEIGHT),
            class_mode=None,
            classes=['image'],
            shuffle=False,
            batch_size=self.__BATCH_SIZE)
        ids = [splitext(basename(x))[0] for x in test_generator.filenames]
        predictions = model.predict_generator(
                test_generator,
                steps=len(test_generator.filenames) // self.__BATCH_SIZE
                )
        predictions = [{'gender': ('male' if int(round(x[0])) == 1 else 'female'), 'confidence': abs(float(x) - 0.5) * 2} for x in predictions]
        results = dict(zip(ids, predictions))
        return results
