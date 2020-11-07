"""
Load a modle and use it to sort images by class.
"""
import numpy as np
import operator
import random
import glob
import os.path
import random
from processor import process_image
from keras.models import load_model
import SMLsettings

settings = SMLsettings.getSettings()

global srcPath 
srcPath = os.path.join(*settings["FrameDestinationPath"])
global dstPath
dstPath = os.path.join(*settings["MachineSortedPath"])
os.makedirs(dstPath,exist_ok=True)
os.makedirs(os.path.join(dstPath,"unsure"),exist_ok=True)
global modelPath
modelPath = os.path.join(*settings["ModelPath"])

trainPath = os.path.join(*settings["TrainingPath"])
global classes
classes = os.listdir(trainPath)
for c in classes:
    os.makedirs(os.path.join(dstPath,c),exist_ok=True)

def main(nb_images=5):
    """Spot-check `nb_images` images."""
    # Get all our test images.
    images = glob.glob(os.path.join(srcPath, '*.jpg'))
    images = images + glob.glob(os.path.join(srcPath, '**', '*.jpg'))
    random.shuffle(images)
    print("found %d images." % len(images))

    #Load the most recent model
    max_mtime =0
    files = glob.glob(os.path.join(modelPath, '*.hdf5'))
    for fname in files:
        mtime = os.stat(fname).st_mtime
        if mtime > max_mtime:
            max_mtime = mtime
            max_file = fname
    model = load_model(max_file)

    for image in images:
        print('-'*80)
        print(image)

        # Turn the image into an array.
        image_arr = process_image(image, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)

        # Predict.
        predictions = model.predict(image_arr)

        # Show how much we think it's each one.
        label_predictions = {}
        for i, label in enumerate(classes):
            label_predictions[label] = predictions[0][i]

        sorted_lps = sorted(label_predictions.items(), key=operator.itemgetter(1), reverse=True)
        
        for i, class_prediction in enumerate(sorted_lps):
            # Just get the top five.
            if i > 4:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
            i += 1
        # sort the files into folders by category
        filename = image.split('\\')[-1]
        if(sorted_lps[0][1]>0.95):
            dest = os.path.join(dstPath,sorted_lps[0][0], filename)
            if(os.path.isfile(dest)):
                os.remove(dest)
            os.rename(image, dest)
        else:
            dest = os.path.join(dstPath,"unsure", filename)
            if(os.path.isfile(dest)):
                os.remove(dest)
            os.rename(image, dest)


if __name__ == '__main__':
    main()
