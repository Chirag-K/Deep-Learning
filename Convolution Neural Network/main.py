# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:53:35 2020

@author: Acer
"""

import tensorflow as tf

with tf.device('/gpu:0'):
    # Importing the keras libraries and packages
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense, Dropout

    # Intializing the CNN
    classifier = Sequential()

    # Convulation - Making Feature Maps
    classifier.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3),
                                 activation = 'relu'))

    # Applying MaxPooling to reduce the size of image
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding another convolution layer
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding another convolution layer
    classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding Flattening to make 2D matrix into 1D for data inputing
    classifier.add(Flatten())

    # Adding Dense Layers
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))

    # Compling
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                       metrics = ['accuracy'])


    # Preprocessing of Data
    train_datagen = ImageDataGenerator(
                    rescale = 1. / 255,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(150, 150),
                                                     batch_size = 64,
                                                     class_mode = 'binary')

    test_set = test_datagen.flow_from_directory('dataset/validation_set',
                                                target_size=(150, 150),
                                                batch_size = 64,
                                                class_mode = 'binary')

    classifier.fit_generator(training_set,
                             steps_per_epoch = 8000,
                             epochs = 10,
                             validation_data = test_set,
                             validation_steps = 2000)
    
    
# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
