import csv
import glob
import os
import platform
import os.path
import time
import numpy as np
from subprocess import call
import cv2  # pip3 install opencv-python
import operator
from PIL import Image   # pip3 install Pillow
import random
import string
import math
from PIL import ImageTk, Image
import tkinter as tk
import functools as ft
import SMLsettings as s
import concurrent.futures
from shutil import rmtree
from validate import load_model
from processor import process_image
from keras.models import load_model

settings = s.getSettings()

global srcPath
srcPath = os.path.join(*settings["VideoSourcePath"])
global archivePath
archivePath = os.path.join(*settings["VideoArchivePath"])
global dstPath
dstPath = os.path.join(*settings["MachineSortedPath"])
os.makedirs(dstPath, exist_ok=True)
global modelPath
modelPath = os.path.join(*settings["ModelPath"])
global interactive
trainPath = os.path.join(*settings["TrainingPath"])
global classes
classes = os.listdir(trainPath)
for c in classes:
    os.makedirs(os.path.join(dstPath, c), exist_ok=True)

interactive = settings["Interactive"]


# def preprocess():
#     files = glob.glob('*.mp4')
#     # For all .mp4 files in the in the unsorted folder
#     for filename in files:
#         # make a jpg of each frame in a temp folder
#         extractFrames(filename)

def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        timeNumber = time.localtime(os.path.getctime(path_to_file))
    else:
        stat = os.stat(path_to_file)
        try:
            timeNumber = time.gmtime(stat.st_birthtime)
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            timeNumber = time.gmtime(stat.st_mtime)
    return time.strftime("%y%m%d_%H%M%S", timeNumber)


def extractFrames(filename):
    src = os.path.join(filename)
    filename_no_ext = creation_date(filename)
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]
    dest = os.path.join(srcPath, "temp", justName)
    #dest = os.path.join("C:\\temp", justName)
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest, filename_no_ext + '-%04d')
    #cmd = ["ffmpeg", "-i", src, "-r", "1", dest]
    #cmd = ["ffmpeg", "-i", src, dest]
    # if filename.find("timelapse")>=0:
    #     cmd = 'ffmpeg -i %s -vf "scale=1920:1080" %s.jpg' % (src, dest)
    # else:
    #     cmd = 'ffmpeg -i %s -vf "scale=1920:1080,fps=fps=1" %s.jpg' % (src, dest)
    # print(cmd)
    # call(cmd)

    if filename.find("timelapse") >= 0:
        #cmd = 'ffmpeg -i %s -vf "scale=1920:1080,fps=fps=6" %s.bmp' % (src, dest)
        cmd = 'ffmpeg -i %s -q:v 1 -vf "scale=1920:1080,fps=fps=6" %s.jpg' % (
            src, dest)
    else:
        #cmd = 'ffmpeg -i %s -vf "scale=1920:1080,fps=fps=1" %s.bmp' % (src, dest)
        cmd = 'ffmpeg -i %s -q:v 1 -vf "scale=1920:1080,fps=fps=1" %s.jpg' % (
            src, dest)
    print(cmd)
    call(cmd)


def showGUI(image, delta, filename, x, y):
    window = tk.Tk()
    image2 = image.copy()
    image2 = cv2.rectangle(image2, (x-150, y-150),
                           (x+150, y+150), (255, 0, 0), 5)
    originalwidth = image2.shape[1]
    originalheight = image2.shape[0]
    scaleFactor = 960/originalwidth
    newWidth = int(scaleFactor*originalwidth)
    newHeight = int(scaleFactor*originalheight)
    imageScaled = cv2.resize(image2, dsize=(
        newWidth, newHeight)).astype(np.uint8)
    b, g, r = cv2.split(imageScaled)
    imageScaled = cv2.merge((r, g, b))
    TkImg = Image.fromarray(imageScaled)
    TkImg = ImageTk.PhotoImage(TkImg)

    panel = tk.Label(window, image=TkImg)
    panel.grid(row=1, column=0, columnspan=10)
    window.bind("<Button 1>", lambda e: onClick(
        e, image, filename, scaleFactor))

    window.mainloop()


def onClick(e, image, filename, scaleFactor):
    x = int(e.x/scaleFactor)
    y = int(e.y/scaleFactor)
    CropAndSave(x, y, image, filename)


def maskFrames(filename):
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]
    #files = glob.glob(os.path.join(srcPath, "temp", justName, '*.bmp'))
    files = glob.glob(os.path.join(srcPath, "temp", justName, '*.jpg'))

    last_image = cv2.imread(files[0]).astype(np.int16)
    image = cv2.imread(files[1]).astype(np.int16)
    for i in range(1, len(files)-1):
        next_image = cv2.imread(files[i+1]).astype(np.int16)
        background = np.percentile([last_image, image, next_image], 50, axis=0)
        delta = abs(image - background)
        delta = (np.sum(delta, 2))
        #cv2.imwrite("delta_raw.jpg", delta)
        height, width = delta.shape
        # blank out movment of the time stamp
        delta[height-100:height, 0:width] = 0
        # low_values_flags = delta < 16  # Where values are low
        # delta[low_values_flags] = 0  # All low values set to 0
        delta_blur = imageErode(delta, 5)
        #delta_blur = cv2.blur(delta, (3, 3)).astype(np.uint8)
        #cv2.imwrite("delta_blur.jpg", delta_blur)
        #print(".", end='')
        print(np.amax(delta_blur))
        # if np.amax(delta_blur) < 48:
        tries = 0
        # while(np.amax(delta_blur) > 48):
        if(interactive):
            showGUI(image, delta, files[i], x, y)
        else:
            while(np.amax(delta_blur) > 40):
                #print( "[{:d}]".format(np.amax(delta_blur)), end ='')
                x, y, delta_blur = centroid(delta_blur)
                # print("x=%d y=%d" % (x, y))
                CropAndSave(x, y, image, files[i], delta)
                tries = tries + 1
                if tries > 3:
                    break
        last_image = image
        image = next_image


def CropAndSave(x, y, image, filename, delta=None):
    height, width, _ = image.shape
    x = max(150, min(width-150, x))
    y = max(150, min(height-150, y))
    image_cropped = image[y-149:y+150, x-149:x+150]
    image_cropped = cv2.normalize(
        image_cropped,  image_cropped, 0, 255, cv2.NORM_MINMAX)
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]

    # Turn the image into an array.
    img_arr = (image_cropped / 255.).astype(np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predict.
    global model
    predictions = model.predict(img_arr)

    # Show how much we think it's each one.
    label_predictions = {}
    for i, label in enumerate(classes):
        label_predictions[label] = predictions[0][i]

    sorted_lps = sorted(label_predictions.items(),
                        key=operator.itemgetter(1), reverse=True)

    # for i, class_prediction in enumerate(sorted_lps):
    #     # Just get the top five.
    #     if i > 4:
    #         break
    #     print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
    #     i += 1
    # sort the files into folders by category
    # if(sorted_lps[0][1] > 0.80 and not sorted_lps[0][0] == "uninteresting"):
    if(not sorted_lps[0][0] == "uninteresting"):
        dest = os.path.join(dstPath, sorted_lps[0][0], '{:2d}_'.format(
            int(sorted_lps[0][1]*100)) + justName + get_random_string(4) + '.jpg')
        cv2.imwrite(dest, image_cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        #os.setxattr(dest, 'user.score', sorted_lps[0][1])
        #os.setxattr(dest, 'Tags', sorted_lps[0][1])
        print()
        print("saved " + dest)


def get_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def centroid(delta):
    y, x = np.unravel_index(np.argmax(delta, axis=None), delta.shape)
    height, width = delta.shape

    # mask2d=np.amax(delta, 2).astype(np.int32)
    # histx = cv2.reduce(mask2d, 0, cv2.REDUCE_SUM, dtype =cv2.CV_32S)
    # histx = histx.flatten()
    # histxi = []
    # for i in range(histx.size):
    #     histxi.append([histx[i],i])
    # histxi.sort(reverse=True)
    # x = int(histxi[0][1])

    # histy = cv2.reduce(mask2d, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    # histy = histy.flatten()
    # histyi = []
    # for i in range(histy.size):
    #     histyi.append([histy[i],i])
    # histyi.sort(reverse=True)
    # y = int(histyi[0][1])

    x = max(150, min(width-150, x))
    y = max(150, min(height-150, y))

    delta[y-149:y+150, x-149:x+150] = 0

    return x, y, delta


def imageDilate(img, size=3):
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def imageErode(img, size=3):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def toGrey(img):
    retval = np.amax(img, 2)
    retval = cv2.merge((retval, retval, retval))
    return retval


def clearTemp(filename):
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]
    folder_to_delete = os.path.join(srcPath, "temp", justName)
    #folder_to_delete = os.path.join("C:\\temp",justName)
    rmtree(folder_to_delete)
    #files = glob.glob(os.path.join(srcPath, 'temp', '*'))
    # for f in files:
    #    os.remove(f)


def videoToArchive(filename):
    justName = os.path.split(filename)[1]
    destName = os.path.join(archivePath, creation_date(
        filename) + get_random_string(4)+".MP4")
    if (not os.path.isfile(destName)):
        os.rename(filename, destName)
    else:
        os.remove(filename)


def processVideo(filename):
    extractFrames(filename)
    maskFrames(filename)
    clearTemp(filename)
    videoToArchive(filename)


def main():
    os.makedirs(os.path.join(srcPath, 'frames_temp'), exist_ok=True)
    os.makedirs(archivePath, exist_ok=True)

    # Load the most recent model
    max_mtime = 0
    files = glob.glob(os.path.join(modelPath, '*.hdf5'))
    for fname in files:
        mtime = os.stat(fname).st_mtime
        if mtime > max_mtime:
            max_mtime = mtime
            max_file = fname
    global model
    model = load_model(max_file)

    files = glob.glob(os.path.join(srcPath, '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '**', '*.mp4'))
    files.sort(reverse=True)
    # 3 workers uses 75% CPU and 75% disk.  6 workers makes the computer unusable for much else
    e = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    for filename in files:
        e.submit(processVideo, filename)
    e.shutdown(wait=True)
    # for file in files:
    #  processVideo(file)


if __name__ == '__main__':
    main()
