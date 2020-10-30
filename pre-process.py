import csv
import glob
import os
import os.path
import numpy as np
from subprocess import call
import cv2
from PIL import Image 
import random

global srcPath 
srcPath = os.path.join('data','unsorted')
global archivePath
archivePath = os.path.join('data','archive')

def preprocess():
    files = glob.glob('*.mp4')
    # For all .mp4 files in the in the unsorted folder
    for filename in files:
        # make a jpg of each frame in a temp folder
        extractFrames(filename)
        # open up to 100 images for one video and calculate the background

            # remove the background from each frame
        #delete the massive number of frames
        #for imagefile in imagefiles
        #    os.remove(imagefile)

def extractFrames(filename):
    src = os.path.join(filename)
    filename_no_ext = filename.split('.')[0]
    filename_no_ext = filename_no_ext.split('\\')[-1]
    dest = os.path.join(srcPath,'temp',filename_no_ext + '-%04d.jpg')
    ##cmd = "ffmpeg -i " + src + ' -vf "select=not(mod(n\,100))" ' +  dest
    cmd = "ffmpeg -i " + src + ' -vf "scale=640:360, fps=fps=1" ' +  dest
    call(cmd)

def makeBackground():
    imagefiles = glob.glob(os.path.join(srcPath,'temp','*.jpg'))
    images = []
    for imagefile in imagefiles:
        images.append(cv2.imread(imagefile))
    background = np.percentile(images, 50, axis=0)
    cv2.imwrite("background.png", background)
    return background

def maskFrames(background):
    files = glob.glob(os.path.join(srcPath,'temp','*.jpg'))
    for filename in files:
        image = cv2.imread(filename)
        
        #mask = (imageDilate(abs(image - background)) > 45) * image
        #mask = imageDilate(abs(image - background))
        #mask = toGrey(mask)
        alpha_channel = (np.amax(abs(image - background), axis=2)).astype(np.uint8)
        b_channel, g_channel, r_channel = cv2.split(image)
        image = cv2.merge((b_channel, g_channel))
        image = cv2.merge((image, r_channel))
        image = cv2.merge((image, alpha_channel))
        filename_no_ext = filename.split('.')[0]
        newFileName = filename_no_ext.replace("temp","frames") + ".png"
        cv2.imwrite(newFileName, image)

def imageDilate(img):
    kernel = np.ones((3,3), np.uint8) 
    return cv2.dilate(img, kernel, iterations=1) 

def toGrey(img):
    retval = np.amax(img,2)
    return retval

def clearTemp():
    files = glob.glob(os.path.join(srcPath,'temp','*'))
    for f in files:
        #os.rename(f,f.replace('temp','frames'))
        os.remove(f)
def videoToArchive(filename):
    justName=filename.split('\\')[-1]
    os.rename(filename, os.path.join(archivePath, justName))


def main():

    os.makedirs(os.path.join(srcPath,'frames'),exist_ok=True)
    os.makedirs(os.path.join(srcPath,'temp'),exist_ok=True)
    os.makedirs(archivePath, exist_ok=True)
    clearTemp()
    files = glob.glob(os.path.join(srcPath,'*.mp4'))
    random.shuffle(files)
    for filename in files:   
    #filename = files[0]      
        extractFrames(filename)
        background = makeBackground()
        maskFrames(background)
        clearTemp()
        videoToArchive(filename)




if __name__ == '__main__':
    main()