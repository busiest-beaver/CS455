import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications.xception import Xception
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from os.path import exists, basename, splitext, join

class image_gender_classifier:
    # dimensions of our images.
    __IMG_WIDTH, __IMG_HEIGHT = 200, 200
    __WEIGHTS_PATH = '/data/weights/gender_xception.hdf5'
    __BATCH_SIZE = 16

    def __init__(self):
        '''empty constructor'''

    def __get_model(self):
        xcept = Xception(include_top=False, weights='imagenet', input_shape = (200,200,3))
        model = Sequential()
        model.add(Flatten(input_shape=xcept.output_shape[1:]))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(1, activation='sigmoid'))
        model = Model(inputs=xcept.input, outputs=model(xcept.output))
        model.load_weights(self.__WEIGHTS_PATH)
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
