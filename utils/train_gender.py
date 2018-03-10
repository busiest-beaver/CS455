import os
import sys
import numpy as np
from itertools import chain
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, AlphaDropout
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.callbacks import Callback
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
from keras.optimizers import Adagrad

# dimensions of our images.
img_width, img_height = 200, 200
base_path = os.path.expanduser('~/data2')
epochs = 150
batch_size = 32

def count(dir):
	return len([name for name in list(chain(*[os.listdir(os.path.join(dir, klass)) for klass in os.listdir(dir)]))])

def get_model(shape, num_classes, activation):	
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation=activation))
    return model


def initial_predictions(class_name):
    training_save_path = "%s/weights/%s_bottleneck_training.npy" % (base_path, class_name)
    validation_save_path = "%s/weights/%s_bottleneck_validation.npy" % (base_path, class_name)
    if (os.path.isfile(training_save_path) or os.path.isfile(validation_save_path)):
	print("Initial predictions found for %s. Stopping early." % class_name)
	return
    train_data_dir = "%s/%s/train" % (base_path, class_name)
    validation_data_dir = "%s/%s/validate" % (base_path, class_name)
    nb_train_samples = count(train_data_dir)
    nb_train_samples = nb_train_samples - nb_train_samples % batch_size
    nb_validation_samples = count (validation_data_dir)
    nb_validation_samples = nb_validation_samples - nb_validation_samples % batch_size
    igen = ImageDataGenerator(rescale=1. / 255)
    model = keras.applications.xception.Xception(include_top=False, weights='imagenet')
    if (class_name == 'age'):
	class_mode = 'categorical'
    else:
        class_mode = None

    training_gen = igen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False)
    training_predictions = model.predict_generator(training_gen, nb_train_samples // batch_size)
    np.save(open(training_save_path, 'wb'), training_predictions)
    print("Saving predicted imagenet fit for %s training samples." % class_name)

    validation_gen = igen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False)
    validation_predictions = model.predict_generator(validation_gen, nb_validation_samples // batch_size)
    np.save(open(validation_save_path, 'wb'), validation_predictions)
    print("Saving predicted imagenet fit for %s validation samples." % class_name)

def train_top_model(class_name):
    training_save_path = "%s/weights/%s_bottleneck_training.npy" % (base_path, class_name)
    validation_save_path = "%s/weights/%s_bottleneck_validation.npy" % (base_path, class_name)
    weights_save_path = "%s/weights/%s_rough_weights.h5" % (base_path, class_name)
    if (os.path.isfile(weights_save_path)):
	print("Rough weights found for %s. Stopping early." % class_name)
	return
    train_data_dir = "%s/%s/train" % (base_path, class_name)
    validation_data_dir = "%s/%s/validate" % (base_path, class_name)
    nb_train_samples = count(train_data_dir)
    nb_train_samples = nb_train_samples - nb_train_samples % batch_size
    nb_validation_samples = count(validation_data_dir)
    nb_validation_samples = nb_validation_samples - nb_validation_samples % batch_size
    if (class_name == 'age'):
        loss = 'categorical_crossentropy'
	num_classes = 4
	class_mode = 'categorical'
        activation = 'softmax'
    else:
        loss = 'binary_crossentropy'
	class_mode = None
	num_classes = 1
	activation = 'sigmoid'

    train_data = np.load(open(training_save_path))
    validation_data = np.load(open(validation_save_path))

    igen = ImageDataGenerator(rescale=1. / 255)
    training_gen = igen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False)
    validation_gen = igen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False)

    if (class_name == 'gender'):
        validation_labels = validation_gen.classes[:nb_validation_samples]
	train_labels = training_gen.classes[:nb_train_samples]
    elif (class_name == 'age'):
	train_labels = training_gen.classes[:nb_train_samples]
	train_labels = to_categorical(train_labels, num_classes=num_classes)
        validation_labels = validation_gen.classes[:nb_validation_samples]
	validation_labels = to_categorical(validation_labels, num_classes=num_classes)
    
    model = get_model(train_data.shape[1:], num_classes, activation)
    optimizer = Adagrad(lr=1e-5)
    model.compile(optimizer=optimizer,
                  loss=loss, metrics=['accuracy'])
    logger = LoggerCallback()
    logger.set_class_name(class_name + '_rough')
    checkpointer = ModelCheckpoint(filepath="%s/weights/%s_rough.{epoch:02d}-{val_loss:.2f}.hdf5" % (base_path, class_name), verbose=1, save_best_only=True, period=1)
    callbacks = [logger, checkpointer]
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
    	      callbacks=callbacks)

def fine_tune(class_name, weights_path):
	train_data_dir = "%s/%s/train" % (base_path, class_name)
	validation_data_dir = "%s/%s/validate" % (base_path, class_name)
	nb_train_samples = count(train_data_dir)
	nb_train_samples = nb_train_samples - nb_train_samples % batch_size
	nb_validation_samples = count (validation_data_dir)
	nb_validation_samples = nb_validation_samples - nb_validation_samples % batch_size
	 
	if (class_name == 'age'):
	    num_classes = 4
	    activation = 'sigmoid'
	    loss = 'categorical_crossentropy'
	    class_mode = 'categorical'
	else:
	    num_classes = 1
	    activation = 'sigmoid'
	    loss = 'binary_crossentropy'
	    class_mode = 'binary'

	model = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape = (200,200,3))
	print('Model loaded.')

	n_layers = len(model.layers) - 6

	top_model = get_model(model.output_shape[1:], num_classes, activation)
	top_model.load_weights(weights_path)

	model = Model(inputs=model.input, outputs=top_model(model.output))

	for layer in model.layers[:n_layers]:
	    layer.trainable = False

	model.compile(loss=loss,
		      optimizer=optimizers.Adagrad(lr=1e-5),
		      metrics=['accuracy'])

	# prepare data augmentation configuration
	train_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
	    train_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode=class_mode)

	validation_generator = test_datagen.flow_from_directory(
	    validation_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode=class_mode)

	logger = LoggerCallback()
	logger.set_class_name(class_name + '_fine')
	checkpointer = ModelCheckpoint(filepath="%s/weights/%s_fine.{epoch:02d}-{val_loss:.2f}.hdf5" % (base_path, class_name), verbose=1, save_best_only=True, monitor='val_loss', period=1)
	callbacks = [logger, checkpointer]
	model.fit_generator(
	    train_generator,
	    samples_per_epoch=nb_train_samples,
	    epochs=epochs,
	    validation_data=validation_generator,
	    nb_val_samples=nb_validation_samples,
            callbacks=callbacks)

def test_model(class_name, weights_path):
	train_data_dir = "%s/%s/train" % (base_path, class_name)
	validation_data_dir = "%s/%s/validate" % (base_path, class_name)
	nb_train_samples = count(train_data_dir)
	nb_train_samples = nb_train_samples - nb_train_samples % batch_size
	nb_validation_samples = count (validation_data_dir)
	nb_validation_samples = nb_validation_samples - nb_validation_samples % batch_size
	 
	if (class_name == 'age'):
	    num_classes = 4
	    activation = 'sigmoid'
	    loss = 'categorical_crossentropy'
	    class_mode = 'categorical'
	else:
	    num_classes = 1
	    activation = 'sigmoid'
	    loss = 'binary_crossentropy'
	    class_mode = 'binary'

	model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (200,200,3))
	print('Model loaded.')

	top_model = Sequential()
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(num_classes, activation=activation))

	model = Model(inputs=model.input, outputs=top_model(model.output))

	model.load_weights(weights_path)

	model.compile(loss=loss,
		      optimizer=optimizers.Adagrad(lr=1e-5),
		      metrics=['accuracy'])

	train_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(
	    train_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode=class_mode)

	validation_generator = test_datagen.flow_from_directory(
	    validation_data_dir,
	    target_size=(img_height, img_width),
	    batch_size=batch_size,
	    class_mode=class_mode)

	evaluations = model.evaluate_generator(
	    validation_generator,
	    steps=nb_validation_samples // batch_size
            )

	print(model.metrics_names)
	print(evaluations)

class LoggerCallback(Callback):
    def set_class_name(self, class_name):
        self.class_name = class_name
    
    def on_train_begin(self, logs={}):
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        with open("%s/logs/%s.log" % (base_path, self.class_name), 'w') as log_file:
	    pickle.dump(self.logs, log_file)


def main():
	class_name = 'gender'
	
	if (len(sys.argv) < 3):
		exit('command path')
	command = sys.argv[1]
	path = sys.argv[2]

	if (command == 'predict'):
		print('Fitting data to image net.')
		initial_predictions(class_name)
	if (command == 'rough'):	
		print('Roughly fitting models.')
		train_top_model(class_name)
	
	if (command == 'fine'):
		print('Fine tune models.')
		fine_tune(class_name, path)
	
	if (command == 'test'):
		print('Testing model.')
		test_model(class_name, path)

main()
