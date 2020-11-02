import csv
import glob
import os
import os.path
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

global srcPath 
srcPath = os.path.join('data','unsorted')
global archivePath
archivePath = os.path.join('data','archive')
global interactive
interactive = True

def preprocess():
    files = glob.glob('*.mp4')
    # For all .mp4 files in the in the unsorted folder
    for filename in files:
        # make a jpg of each frame in a temp folder
        extractFrames(filename)

def extractFrames(filename):
    src = os.path.join(filename)
    filename_no_ext = filename.split('.')[0]
    filename_no_ext = os.path.split(filename_no_ext)[-1]
    dest = os.path.join(srcPath,'temp',filename_no_ext + '-%04d.jpg')
    cmd = ["ffmpeg", "-i", src, "-r", "1", dest]
    call(cmd)

def makeBackground():
    imagefiles = glob.glob(os.path.join(srcPath,'temp','*.jpg'))
    images = []
    for imagefile in imagefiles:
        images.append(cv2.imread(imagefile))
    background = np.percentile(images, 50, axis=0)
    cv2.imwrite("background.png", background)
    return background

def showGUI(image,delta,filename):
    window = tk.Tk() 
    #window.geometry("960x960")  
    originalwidth=image.shape[1]
    originalheight=image.shape[0]
    scaleFactor=960/originalwidth
    newWidth=int(scaleFactor*originalwidth)
    newHeight=int(scaleFactor*originalheight)
    imageScaled = cv2.resize(image, dsize=(newWidth,newHeight))
    TkImg = Image.fromarray(imageScaled)
    TkImg = ImageTk.PhotoImage(TkImg)

    panel = tk.Label(window, image = TkImg)
    panel.grid(row=1,column=0,columnspan=10)
    window.bind("<Button 1>", lambda e: onClick(e,image,filename,scaleFactor))

    window.mainloop()

def onClick(e,image,filename,scaleFactor):
    x=int(e.x/scaleFactor)
    y=int(e.y/scaleFactor)
    CropAndSave(x,y,image,filename)
    

def maskFrames(background):
    files = glob.glob(os.path.join(srcPath,'temp','*.jpg'))
    for filename in files:
        image = cv2.imread(filename)
        delta = abs(image - background)
        delta=imageErode(delta,20).astype(np.uint8)
        if np.amax(delta)<32 : continue
        if(interactive):
            showGUI(image,delta,filename)
        else:
            delta=cv2.blur(delta, (3,3))
            x,y = centroid(delta)
            print("x=%d y=%d" % (x, y))
            CropAndSave(x,y,image,filename)


def CropAndSave(x,y,image,filename):
    width = image.shape[1]
    height = image.shape[0]
    x = max(150, min(width-150, x))
    y = max(150, min(height-150, y))
    image_cropped = image[y-149:y+150, x-149:x+150]
    filename_no_ext = filename.split('.')[0]
    newFileName = filename_no_ext.replace("temp","frames") + get_random_string(4) + ".jpg"
    cv2.imwrite(newFileName, image_cropped)
    #sometimes a head is chopped off, so crop above the target as well.
    y=y-150
    y = max(150, min(height-150, y))
    image_cropped = image[y-149:y+150, x-149:x+150]
    print("saved "+filename_no_ext)
    newFileName = filename_no_ext.replace("temp","frames") + get_random_string(4) + ".jpg"
    cv2.imwrite(newFileName, image_cropped)

def get_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def centroid(delta):

    height,width,depth = delta.shape


    mask2d=np.amax(delta, 2)
    histx = cv2.reduce(mask2d, 0, cv2.REDUCE_SUM, dtype =cv2.CV_32S)
    histx = histx.flatten()
    histxi = []
    for i in range(histx.size):
        histxi.append([histx[i],i]) 
    histxi.sort(reverse=True)
    x = histxi[0][1]

    histy = cv2.reduce(mask2d, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    histy = histy.flatten()
    histyi = []
    for i in range(histy.size):
        histyi.append([histy[i],i]) 
    histyi.sort(reverse=True)
    y = histyi[0][1]
    return int(x),int(y)

def imageDilate(img, size=3):
    kernel = np.ones((size,size), np.uint8) 
    return cv2.dilate(img, kernel, iterations=1) 

def imageErode(img, size=3):
    kernel = np.ones((size,size), np.uint8) 
    return cv2.erode(img, kernel, iterations=1) 

def toGrey(img):
    retval = np.amax(img,2)
    retval = cv2.merge((retval,retval,retval))
    return retval

def clearTemp():
    files = glob.glob(os.path.join(srcPath,'temp','*'))
    for f in files:
        #os.rename(f,f.replace('temp','frames'))
        os.remove(f)
def videoToArchive(filename):
    justName=filename.split('\\')[-1]
    destName = os.path.join(archivePath, justName)
    if (not os.path.isfile(destName)): 
        os.rename(filename, destName)

def main():

    os.makedirs(os.path.join(srcPath,'frames'),exist_ok=True)
    os.makedirs(os.path.join(srcPath,'temp'),exist_ok=True)
    os.makedirs(archivePath, exist_ok=True)
    clearTemp()
    files = glob.glob(os.path.join(srcPath,'*.mp4'))
    random.shuffle(files)
    for filename in files:   
        extractFrames(filename)
        background = makeBackground()
        maskFrames(background)
        clearTemp()
        videoToArchive(filename)

if __name__ == '__main__':
    main()