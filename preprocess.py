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
from PIL import ImageTk, Image
import tkinter as tk
import functools as ft
import SMLsettings as s
import concurrent.futures
from shutil import rmtree

settings = s.getSettings()

global srcPath
srcPath = os.path.join(*settings["VideoSourcePath"])
global archivePath
archivePath = os.path.join(*settings["VideoArchivePath"])
global interactive
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

    if filename.find("timelapse")>=0:
        cmd = 'ffmpeg -i %s -vf "scale=1920:1080,fps=fps=4" %s.bmp' % (src, dest)
    else:
        cmd = 'ffmpeg -i %s -vf "scale=1920:1080,fps=fps=1" %s.bmp' % (src, dest)
    print(cmd)
    call(cmd)

def makeBackground(filename):
    print("Making the background.  This can take a while.")
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]
    #imagefiles = glob.glob(os.path.join(srcPath, justName, '*.jpg'))
    imagefiles = glob.glob(os.path.join(srcPath, "temp", justName, '*.bmp'))
    random.shuffle(imagefiles)
    imagefiles = imagefiles[:24]
    images = []
    for imagefile in imagefiles:
        images.append(cv2.imread(imagefile))
    background = np.percentile(images, 50, axis=0)
    cv2.imwrite("background.png", background)
    return background


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
    imageScaled = cv2.resize(image2, dsize=(newWidth, newHeight))
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


def maskFrames(background, filename):
    justName = os.path.split(filename)[1]
    justName = justName.split('.')[0]
    #files = glob.glob(os.path.join(srcPath, justName, '*.jpg'))
    files = glob.glob(os.path.join(srcPath, "temp", justName, '*.bmp'))
    for filename in files:
        image = cv2.imread(filename)
        delta = abs(image - background)
        delta = (np.sum(delta,2)/3)
        #cv2.imwrite("delta_raw.jpg", delta)
        height, width = delta.shape
        # blank out movment of the time stamp
        delta[height-100:height, 0:width] = 0
        # low_values_flags = delta < 16  # Where values are low
        # delta[low_values_flags] = 0  # All low values set to 0
        delta = imageErode(delta, 3)
        delta_blur = cv2.blur(delta, (3, 3)).astype(np.uint8)
        #cv2.imwrite("delta_blur.jpg", delta)
        print(np.amax(delta_blur))
        #if np.amax(delta_blur) < 48:
        if np.amax(delta_blur) < 64:
            continue
        tries = 0
        #while(np.amax(delta_blur) > 48):
        while(np.amax(delta_blur) > 64):
            x, y, delta_blur = centroid(delta_blur)
            print(np.amax(delta_blur))
            print("x=%d y=%d" % (x, y))
            if(interactive):
                showGUI(image, delta, filename, x, y)
            else:
                CropAndSave(x, y, image, filename, delta)
            tries = tries + 1
            if tries > 3:
                break


def CropAndSave(x, y, image, filename, delta = None):
    #print(np.amin(delta))
    #print(np.amax(delta))
    delta = (delta /2 +127).astype(np.uint8)
    #print(np.amin(delta))
    #print(np.amax(delta))
    r,g,b = cv2.split(image)
    image = cv2.merge((r,g,b,delta))
    #r,g,b,a = cv2.split(image)
    #print(np.amin(a))
    #print(np.amax(a))
    height, width, _ = image.shape
    x = max(150, min(width-150, x))
    y = max(150, min(height-150, y))
    image_cropped = image[y-149:y+150, x-149:x+150]
    #r,g,b,a = cv2.split(image_cropped)
    #print(np.amin(a))
    #print(np.amax(a))
    justName = os.path.split(filename)[1]
    justName = filename.split('.')[0]
    newFileName = os.path.join(srcPath,"frames",justName+get_random_string(4)+".jpg")
    #newFileName = filename_no_ext.replace(
        #justName, "frames") + get_random_string(4) + ".jpg"
        #"temp", "frames") + get_random_string(4) + ".jpg"
    cv2.imwrite(newFileName, image_cropped,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #newFileName = os.path.join(srcPath,"png",justName+get_random_string(4)+".png")
    #newFileName = filename_no_ext.replace(
        #justName, "png") + get_random_string(4) + ".png"
        #"temp", "png") + get_random_string(4) + ".png"
    #cv2.imwrite(newFileName, image_cropped)
    # #sometimes a head is chopped off, so crop above the target as well.
    # y=y-150
    # y = max(150, min(height-150, y))
    # image_cropped = image[y-149:y+150, x-149:x+150]
    # print("saved "+filename_no_ext)
    # newFileName = filename_no_ext.replace("temp","frames") + get_random_string(4) + ".jpg"
    # cv2.imwrite(newFileName, image_cropped)


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
    folder_to_delete = os.path.join(srcPath,"temp",justName)
    rmtree(folder_to_delete)
    #files = glob.glob(os.path.join(srcPath, 'temp', '*'))
    #for f in files:
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
    background = makeBackground(filename)
    maskFrames(background, filename)
    clearTemp(filename)
    videoToArchive(filename)

def main():
    os.makedirs(os.path.join(srcPath, 'frames'), exist_ok=True)
    os.makedirs(archivePath, exist_ok=True)
    files = glob.glob(os.path.join(srcPath, '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '*.mp4'))
    files = files + glob.glob(os.path.join(srcPath, '**', '**', '*.mp4'))
    files.sort(reverse=True)
    e = concurrent.futures.ThreadPoolExecutor(max_workers=6)
    for filename in files:
       e.submit(processVideo,filename)
    e.shutdown(wait=True)
    #processVideo("F:\\SecurityML\\data\\unsorted\\2020_1128_042452_018.MP4")


if __name__ == '__main__':
    main()
