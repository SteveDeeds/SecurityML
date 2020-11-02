"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
import os.path
#from data import DataSet
from processor import process_image
from keras.models import load_model

global srcPath
#srcPath = os.path.join('data', 'unsorted')
srcPath = os.path.join('data','unsorted')
global dstPath
dstPath = os.path.join('data', 'machine')
os.makedirs(dstPath,exist_ok=True)
global classes
classes = os.listdir(dstPath)
for c in classes:
    os.makedirs(os.path.join(dstPath,c),exist_ok=True)

def main(nb_images=5):
    """Spot-check `nb_images` images."""
    #data = DataSet()
    model = load_model('data/checkpoints/inception.006-0.03.hdf5')

    # Get all our test images.
    images = glob.glob(os.path.join(srcPath, '**', '*.jpg'))

    for image in images:
        print('-'*80)
        # Get a random row.
        #sample = random.randint(0, len(images) - 1)
        #image = images[sample]

        # Turn the image into an array.
        print(image)
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
        filename = image.split('\\')[-1]
        if(sorted_lps[0][1]>0.9):
            os.rename(image, os.path.join(dstPath,sorted_lps[0][0], filename))


if __name__ == '__main__':
    main()
