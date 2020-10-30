import csv
import glob
import os
import os.path
import numpy as np
from subprocess import call
import cv2
from PIL import Image 
import random
import math

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
    #cmd = "ffmpeg -i " + src + ' -vf "scale=640:360, fps=fps=1" ' +  dest
    cmd = "ffmpeg -i " + src + ' -vf "fps=fps=1" ' +  dest
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
        mask = (imageDilate(abs(image - background)) > 45).astype(np.uint8)
        x,y = centroid(mask)
        if(x<0):continue
        mask = imageDilate(imageErode(toGrey(mask),size=10), size=20)
        image_masked = image * mask
        image_cropped = image[y-149:y+150, x-149:x+150]
        filename_no_ext = filename.split('.')[0]
        newFileName = filename_no_ext.replace("temp","frames") + ".png"
        cv2.imwrite(newFileName, image_cropped)
        newFileName = filename_no_ext.replace("temp","frames") + "_mask.png"
        cv2.imwrite(newFileName, image_masked)
        #newFileName = filename_no_ext.replace("temp","frames") + "_image.png"
        #cv2.imwrite(newFileName, image)
        print("saved "+filename_no_ext)

def centroid(mask):
    sumx=0
    sumy=0
    denominator=0
    height,width,depth = mask.shape
    mask=imageErode(mask,20)
    for y,row in enumerate(mask): #1920x1080 takes about a minute
       for x,pixel in enumerate(row):
           if pixel.any()>0:
               sumx = sumx + x
               sumy = sumy + y
               denominator + denominator + 1
    # for x in range(width):  # 1920x1080 takes about a minute
    #     for y in range(height):
    #         if(mask[y,x].any()>0):
    #            sumx = sumx + x
    #            sumy = sumy + y
    #            denominator = denominator + 1
    # mask1d=mask.reshape(-1,3) # 1920x1080 takes about a minute
    # for i,p in enumerate(mask1d):
    #        if p.any()>0:
    #            x=i%width
    #            y=math.floor(i/width)
    #            sumx = sumx + x
    #            sumy = sumy + y
    #            denominator + denominator + 1    
    # mask1d=mask.reshape(-1) # 1920x1080 took 3 minutes
    # for i,p in enumerate(mask1d):
    #        if p.any()>0:
    #            x=i%(width*3)
    #            y=math.floor(i/(width*3))
    #            sumx = sumx + x
    #            sumy = sumy + y
    #            denominator + denominator + 1

    if(denominator>0):
        x = sumx / denominator
        y = sumy / denominator
        if x<150:x=150
        if y<150:y=150
        if x>(width-150):x=width-150
        if y>(height-150):y=height-150
    else:
        x=-1
        y=-1
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
        #clearTemp()
        #videoToArchive(filename)




if __name__ == '__main__':
    main()