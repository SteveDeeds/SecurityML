from tkinter import *
from PIL import Image, ImageTk
import glob
import random
import os
import SMLsettings
import cv2
import numpy as np
import string

settings = SMLsettings.getSettings()

global srcPath
srcPath = os.path.join(*settings["FrameDestinationPath"])
global dstPath
dstPath = os.path.join(*settings["TrainingPath"])


class Window(Frame):
    def nextImage(window, event=None):
        global currentFile
        currentFile = files.pop()
        #load = Image.open(currentFile)
        #render = ImageTk.PhotoImage(load)
        image = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(image)
        newImage = cv2.merge((r, g, b))
        render = ImageTk.PhotoImage(Image.fromarray(newImage))
        window.img.image = render
        window.img.config(image=render)

    def imageClicked(window, event):
        CropAndSave(event.x, event.y)
        nextImage(window, event)

    def keyPressed(event):
        print("a key was pressed.")

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        image = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(image)
        newImage = cv2.merge((r, g, b))
        render = ImageTk.PhotoImage(Image.fromarray(newImage))
        self.img = Label(self, image=render)
        self.img.image = render
        self.img.bind("<Button-1>", self.imageClicked)
        self.bind("<Key>", self.keyPressed)
        # self.img.place(x=0,y=0)
        self.bUninteresting = Button(
            self, text="Uninteresting", command=self.nextImage)
        self.img.grid(row=0, column=0, columnspan=10)
        self.bUninteresting.grid(row=1, column=0)


def get_random_string(length):
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def CropAndSave(x, y):
    global currentFile
    image = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
    #b,g,r,a = cv2.split(image)
    #image = cv2.merge((b,g,r,a))
    #image = Image.open(currentFile)
    #image = np.array(image)
    height, width, _ = image.shape
    x = max(75, min(width-75, x))
    y = max(75, min(height-225, y))
    image_cropped = image[y-75:y+225, x-75:x+75]
    #image_cropped = Image.fromarray(image_cropped)
    filename_no_ext = os.path.split(currentFile)[1]
    filename_no_ext = filename_no_ext.split('.')[0]
    newFileName = os.path.join(
        dstPath, filename_no_ext + get_random_string(4) + ".png")
    # image_cropped.save(newFileName)
    cv2.imwrite(newFileName, image_cropped)


root = Tk()

files = glob.glob(os.path.join(srcPath, '*.png'))
files = files + glob.glob(os.path.join(srcPath, '*.jpg'))
random.shuffle(files)
currentFile = files.pop()

app = Window(root)
root.wm_title("Click on a head")
# root.geometry("300x300")
root.mainloop()
