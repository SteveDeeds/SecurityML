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

# Local import
import embedding


def predict(weights_file):
    settings = s.getSettings()
    trainPath = os.path.join(*settings["FrameDestinationPath"])
    os.makedirs(trainPath, exist_ok=True)
    modelPath = os.path.join(*settings["EmbeddingModelPath"])
    os.makedirs(modelPath, exist_ok=True)
    logPath = os.path.join(*settings["EmbeddingModelPath"]+["logs"])
    os.makedirs(logPath, exist_ok=True)

    model = embedding.get_model()
    train, validation = embedding.get_generators(trainPath)
    print("Loading saved model: %s." % weights_file)
    model.load_weights(weights_file)
    embedding.compile(model)

    input_images = train.__getitem__(0)[0]
    print("shape: " + str(input_images.shape))
    output_images = model.predict(input_images)
    print("shape: " + str(output_images.shape))

    batch_size = input_images.shape[0]
    print("batch size: " + str(batch_size))
    for i in range(0, batch_size):
        input_filename = "image_%d_input.jpg" % i
        output_filename = "image_%d_output.jpg" % i
        cv2.imwrite(input_filename, (input_images[i, ] + 0.5) * 255)
        cv2.imwrite(output_filename, (output_images[i, ] + 0.5) * 255)
        print("output file: " + output_filename)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Weight file required.")
        exit(1)
    weights_file = sys.argv[1]
    predict(weights_file)
