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

settings = s.getSettings()

global archivePath
archivePath = os.path.join(*settings["VideoArchivePath"])
#archivePath = "F:\SecurityML\example_video"



def compressVideo(filename):
    src = filename
    dest = os.path.join(os.path.split(filename)[0],"compressed",os.path.split(filename)[1])
    cmd = 'ffmpeg -i %s -vf "scale=960:540,fps=fps=4" -vcodec libx264 -crf 25 %s' % (src, dest)
    print(cmd)
    call(cmd)
    compresion = os.stat(dest).st_size / os.stat(src).st_size
    print("Reduced to %2.2f%% of original size." % (compresion * 100))
    os.remove(filename)

def main():
    os.makedirs(os.path.join(archivePath, 'compressed'), exist_ok=True)
    files = glob.glob(os.path.join(archivePath, '*.mp4'))
    #files = files + glob.glob(os.path.join(archivePath, '*.MP4'))
    files.sort()
    for filename in files:
        compressVideo(filename)


if __name__ == '__main__':
    main()
