"""
Train an embedding model based on random crops from source images.
We vary scale of the crops and add noise to the input image to increase
versitility and avoid over-fitting.
"""
import os.path
import random
import glob
import numpy as np
import cv2
import sys
import SMLsettings as s
import tensorflow as tf
import embedding


def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit(
        x=train_generator,
        steps_per_epoch=20,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def train_and_save(weights_file):
    settings = s.getSettings()
    trainPath = os.path.join(*settings["EmbeddingTrainingPath"])
    os.makedirs(trainPath, exist_ok=True)
    modelPath = os.path.join(*settings["EmbeddingModelPath"])
    os.makedirs(modelPath, exist_ok=True)
    logPath = os.path.join(*settings["EmbeddingModelPath"]+["logs"])
    os.makedirs(logPath, exist_ok=True)

    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            modelPath, "embedding.{epoch:03d}-{val_loss:.2f}.hdf5"),
        verbose=1,
        save_best_only=True)

    model = embedding.get_model()
    generators = embedding.get_generators(trainPath)

    if weights_file is None:
        print("Using uninitialized model.")
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)
    embedding.compile(model)
    model = train_model(
        model, 20, generators, [checkpointer])


if __name__ == '__main__':
    weights_file = None
    if len(sys.argv) > 1:
        weights_file = sys.argv[1]
    train_and_save(weights_file)
