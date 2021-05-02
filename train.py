##################################
'''
Model Train code for Mask Detection
'''
##################################
import warnings

import sys
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.callbacks import EarlyStopping

import config as cfg

def get_model(IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS=3):
	model = Sequential()
	# Conv layer 1
	model.add(Conv2D(16, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS)))
	model.add(MaxPooling2D())
	# Conv layer 2
	model.add(Conv2D(16, (3,3), activation='relu'))
	model.add(MaxPooling2D())
	# Conv layer 3
	model.add(Conv2D(16, (3,3), activation='relu'))
	model.add(MaxPooling2D())
	# Flatten
	model.add(Flatten())
	# Fully Connected
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	# Output
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def save_model(model, models_path):
	if not os.path.exists(models_path):
		os.makedirs(models_path)
	timestr = time.strftime("%Y%m%d_%H%M%S")
	model_file_name = 'MODEL_' + timestr
	model.save(models_path+model_file_name+'.h5')

def train_and_save_model():
	print('Training on GPU: ', tf.test.is_gpu_available())
	images_root_directory = cfg.DATASET_DIR
	train_path = 'Train/'
	val_path = 'Validation/'

	image_labels = os.listdir(images_root_directory + train_path)
	# 0 - Mask and 1 - No Mask
	IMG_WIDTH = cfg.IMG_WIDTH
	IMG_HEIGHT = cfg.IMG_HEIGHT
	batch_size = cfg.BATCH_SIZE
	epochs = cfg.EPOCHS

	full_train_path = images_root_directory + train_path
	full_val_path = images_root_directory + val_path

	# load data using ImageDataGenerator
	train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
	    full_train_path,
	    target_size=(IMG_WIDTH, IMG_HEIGHT),
	    batch_size=batch_size,
	    shuffle=True,
	    class_mode='categorical')
	validation_generator = test_datagen.flow_from_directory(
	    full_val_path,
	    target_size=(IMG_WIDTH, IMG_HEIGHT),
	    batch_size=batch_size,
	    class_mode='categorical')

	model = get_model(IMG_WIDTH, IMG_HEIGHT)
	# early-stop to prevent overfitting
	early_stop = EarlyStopping(monitor='val_loss', patience=2)
	# fit model to data generator
	model_history = model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[early_stop])
	history = pd.DataFrame(model_history.history)
	history[['loss', 'val_loss']].plot()
	plt.show()
	history[['accuracy', 'val_accuracy']].plot()
	plt.show()
	models_path = cfg.MODELS_PATH
	save_model(model, models_path)
	print('Model Saved.')

if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	train_and_save_model()