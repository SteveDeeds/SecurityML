"""
This module encapsulates training, evaluation, and image processing 
for embedding models.

Example:

files = glob(os.join(my_image_path, "*.jpg"))
train_gen, validation_gen = get_generators(files)
model = get_model()
compile(model)

model.fit(
    x=train_gen,
    steps_per_epoch=20,
    validation_data=validation_gen,
    validation_steps=10,
    epochs=100)

"""

import cv2
import glob
import os
import random
import tensorflow as tf
import numpy as np


class EmbeddingGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=20):
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.on_epoch_end()

    def on_epoch_end(self):
        'Shuffle the data at the end of each epoch'
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)

    def crop(self, image):
        zoom = random.uniform(0.25, 1.0)
        image_height, image_width, image_depth = image.shape
        crop_height = int(zoom * image_height)
        crop_width = int(crop_height / 2)
        x = random.randrange(0, image_width - crop_width)
        y = random.randrange(0, image_height - crop_height)
        image_cropped = image[y:y+crop_height, x:x+crop_width]
        image_final = cv2.resize(image_cropped, dsize=(299, 599),
                                 interpolation=cv2.INTER_AREA)
        # TODO
        return image_final

    def add_noise(self, image):
        height, width, depth = image.shape
        mean = 0
        sd = 0.02
        gauss = np.random.normal(mean, sd, (height, width, depth))
        gauss = gauss.reshape(height, width, depth)
        noisy = image + gauss
        return noisy

    def __data_generation(self, batch_image_paths):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_size = len(batch_image_paths)
        x = np.empty((batch_size, 599, 299, 3), dtype='float32')
        y = np.empty((batch_size, 599, 299, 3), dtype='float32')

        # Generate data
        for i, filename in enumerate(batch_image_paths):
            image = cv2.imread(filename)
            image = image * (1. / 255.) - 0.5
            cropped_image = self.crop(image)
            y[i, ] = cropped_image
            noisy_image = self.add_noise(cropped_image)
            x[i, ] = noisy_image

        return x, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)
        indexes = self.indexes[0:self.batch_size]

        # Look up paths by selected indexes
        batch_image_paths = [self.image_paths[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(batch_image_paths)

        return x, y

    def __len__(self):
        return int(100 * np.ceil(len(self.image_paths) / self.batch_size))


def get_generators(trainPath):
    batch_size = 20

    all_files = glob.glob(os.path.join(trainPath, '*'))
    train_files = []
    validation_files = []
    for i in range(0, len(all_files)):
        if random.random() > 0.8:
            validation_files.append(all_files[i])
        else:
            train_files.append(all_files[i])

    train_generator = EmbeddingGenerator(train_files, batch_size=batch_size)
    validation_generator = EmbeddingGenerator(
        validation_files, batch_size=batch_size)

    return train_generator, validation_generator


def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=[599, 299, 3], dtype='float32'))
    model.add(tf.keras.layers.Conv2D(
        filters=3, kernel_size=(5, 1), strides=1, padding="same", activation="tanh"))
    model.add(tf.keras.layers.Conv2D(
        filters=1, kernel_size=(1, 5), strides=1, padding="same", activation="tanh"))

    # model.add(tf.keras.layers.Reshape([599 * 299]))
    # model.add(tf.keras.layers.Dense(100, activation='tanh'))

    model.add(tf.keras.layers.Conv2D(
        filters=1, kernel_size=(5, 1), strides=1, padding="same", activation="tanh"))
    model.add(tf.keras.layers.Conv2D(
        filters=3, kernel_size=(1, 5), strides=1, padding="same", activation="tanh"))

    print("Parameter count: " + str(model.count_params()))

    return model


def compile(model):
    loss_fn = tf.keras.losses.MeanSquaredError()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999),
        loss=loss_fn)
    return model
