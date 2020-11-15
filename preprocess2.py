import csv
import glob
import os
import platform
import os.path
import time
import numpy as np
from subprocess import call
import cv2  # pip3 install opencv-python

from PIL import Image   # pip3 install Pillow
import random
import string
import math
import SMLsettings as s

settings = s.getSettings()

global srcPath
srcPath = os.path.join(*settings["VideoSourcePath"])
global archivePath
archivePath = os.path.join(*settings["VideoArchivePath"])
global interactive
interactive = settings["Interactive"]


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
            timeNumber = stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            timeNumber = stat.st_mtime
    return time.strftime("%y%m%d_%H%m%S", timeNumber)


def extractFrames(filename):
    src = os.path.join(filename)
    filename_no_ext = creation_date(filename)
    dest = os.path.join(srcPath, 'temp', filename_no_ext + '-%04d.jpg')
    #cmd = ["ffmpeg", "-i", src, "-vf scale=960:540;fps=fps=1", dest]
    #cmd = ["ffmpeg", "-i", src, "-filter:v fps=fps=1", dest]
    cmd = 'ffmpeg -i %s -vf "scale=960:540,fps=fps=1" %s' % (src, dest)
    call(cmd)


def makeBackground():
    print("Making the background.  This can take a while.")
    imagefiles = glob.glob(os.path.join(srcPath, 'temp', '*.jpg'))
    random.shuffle(imagefiles)
    imagefiles = imagefiles[:24]
    images = []
    for imagefile in imagefiles:
        images.append(cv2.imread(imagefile))
    background = np.percentile(images, 50, axis=0)
    cv2.imwrite("background.png", background)
    return background


def maskFrames(background):
    files = glob.glob(os.path.join(srcPath, 'temp', '*.jpg'))
    for filename in files:
        image = cv2.imread(filename)
        b, g, r = cv2.split(image)
        delta = abs(image - background).astype(np.uint8)
        delta = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
        newImage = cv2.merge((b, g, r, delta))
        filename_no_ext = filename.split('.')[0]
        newFileName = filename_no_ext.replace(
            "temp", "frames") + get_random_string(4) + ".png"
        cv2.imwrite(newFileName, newImage)


def get_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def clearTemp():
    files = glob.glob(os.path.join(srcPath, 'temp', '*'))
    for f in files:
        # os.rename(f,f.replace('temp','frames'))
        os.remove(f)


def videoToArchive(filename):
    justName = filename.split('\\')[-1]
    destName = os.path.join(archivePath, creation_date(
        filename) + get_random_string(4)+".MP4")
    if (not os.path.isfile(destName)):
        os.rename(filename, destName)


def main():
    os.makedirs(os.path.join(srcPath, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(srcPath, 'temp'), exist_ok=True)
    os.makedirs(archivePath, exist_ok=True)
    clearTemp()
    files = glob.glob(os.path.join(srcPath, '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '**', '*.mp4'))
    random.shuffle(files)
    for filename in files:
        extractFrames(filename)
        background = makeBackground()
        maskFrames(background)
        clearTemp()
        videoToArchive(filename)


if __name__ == '__main__':
    main()
