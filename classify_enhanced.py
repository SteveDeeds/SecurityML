import csv
import glob
import os
import os.path
import numpy as np
from subprocess import call
import cv2
from PIL import ImageTk, Image
import tkinter as tk
import random
import SMLsettings
from validate import load_model
from functools import partial
import threading

window = tk.Tk()
currentFile = ""
panel1 = tk.Label()
panel2 = tk.Label()

settings = SMLsettings.getSettings()

global srcPath
# srcPath = os.path.join(*settings["FrameDestinationPath"])
srcPath = 'F:\\SecurityML\\data\\machine\\person'
global dstPath
dstPath = os.path.join(*settings["TrainingPath"])
trainPath = os.path.join(*settings["TrainingPath"])
global modelPath
modelPath = os.path.join(*settings["ModelPath"])
global interactive
global classes
classes = os.listdir(trainPath)


class PictureInfo:
    def __init__(self, fname="", classification="", score=1):
        self.fname = fname
        self.classification = classification
        self.score = score

    def __lt__(self, other):
        return self.score < other.score

    def __ge__(self, other):
        return self.score >= other.score


def displayImage():
    global currentFile
    global files_with_scores
    global window
    if len(files_with_scores) > 0:
        item = files_with_scores.pop()
        currentFile = item.fname
        window.title(item.classification + str(item.score))
    else:
        currentFile = files.pop()
        window.title(currentFile)
    img1 = Image.open(currentFile)
    w, h = img1.size
    ratio = w/h
    img1 = img1.resize((int(400*ratio), 400), Image.ANTIALIAS)

    img2 = np.asarray(img1)
    img2 = img2 * 2
    img2 = Image.fromarray(img2)
    img2 = ImageTk.PhotoImage(img2)

    img1 = ImageTk.PhotoImage(img1)

    panel1.configure(image=img1)
    panel1.image = img1
    panel2.configure(image=img2)
    panel2.image = img2
    return img1, img2


def add_scores():
    from processor import process_image
    from keras.models import load_model
    global files
    global files_with_scores

    max_mtime = 0
    model_files = glob.glob(os.path.join(modelPath, '*.hdf5'))
    for fname in model_files:
        mtime = os.stat(fname).st_mtime
        if mtime > max_mtime:
            max_mtime = mtime
            max_file = fname
    model = load_model(max_file)

    while len(files):
        current_file = files.pop()

        classification = ""
        for c in classes:
            if current_file.find(c) >= 0:
                classification = c
                break

        image_arr = process_image(current_file, (299, 299, 3))
        image_arr = np.expand_dims(image_arr, axis=0)
        predictions = model.predict(image_arr)
        prediction = predictions[0]
        if not classification == "":
            score = prediction[classes.index(classification)]
        else:
            score = np.amax(prediction)
            classification = prediction.index(score)

        files_with_scores.append(PictureInfo(
            current_file, classification, score))

        files_with_scores.sort(reverse=True)


def Initilize():
    global window
    global currentFile
    global panel1
    global panel2

    window.title("Calssify the image")
    # window.geometry("800x1000")
    window.configure(background='grey')

    # files = glob.glob(os.path.join(srcPath, '*.png'))
    global files
    files = glob.glob(os.path.join(srcPath, '*.jpg'))
    global files_with_scores
    files_with_scores = []
    # for file in files:
    #     classification = ""
    #     for c in classes:
    #         if file.find(c) >= 0:
    #             classification = c
    #     files_with_scores.append(PictureInfo(file, classification, 1))

    ai = threading.Thread(target=add_scores)
    ai.start()

    img1, img2 = displayImage()
    # currentFile = files.pop()
    # img1 = ImageTk.PhotoImage(Image.open(currentFile))
    # img1 = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
    # r,g,b,a = cv2.split(img1)
    # a = ((a>32)*255).astype(np.uint8)
    # a = a.astype(np.uint8)
    # a = (a * 4 + 32).astype(np.int32)
    # maximum = np.full(a.shape, 255).astype(np.int32)
    # cv2.merge((a,maximum))
    # a = np.amin(a)
    # a = a.astype(np.uint8)
    # img1 = cv2.merge((b,g,r,a))
    # img1 = Image.fromarray(img1)
    # img1 = ImageTk.PhotoImage(img1)
    # img2 = cv2.merge((b,g,r))
    # img2 = Image.fromarray(img2)
    # img2 = ImageTk.PhotoImage(img2)
    # img2 = ImageTk.PhotoImage(Image.open(currentFile.replace('masked','frames')))
    panel1 = tk.Label(window, image=img1)
    panel2 = tk.Label(window, image=img2)
    i = 0
    buttons = [None] * len(classes)
    for c in classes:
        # buttons[i] = (tk.Button(window, text=c, command= lambda: buttonClick(c)))
        buttons[i] = (tk.Button(window, text=c,
                                command=partial(buttonClick, c)))
        buttons[i].grid(row=3, column=i)
        i = i+1

    # srcPathLabel = tk.Label(master=window,textvariable=srcPath)
    # srcButton = tk.Button(text="Browse", command=browse_src_button)

    # pack it
    # srcButton.grid(row=0,column=0)
    # srcPathLabel.grid(row=0,column=1)
    panel1.grid(row=1, column=0, columnspan=7)
    panel2.grid(row=1, column=8, columnspan=7)

    window.mainloop()


def main():
    for c in classes:
        os.makedirs(os.path.join(dstPath, c), exist_ok=True)
    Initilize()


def moveToFolder(folder):
    justName = os.path.split(currentFile)[1]
    dest = os.path.join(dstPath, folder, justName)
    # if(os.path.isfile(dest)):
    #    os.remove(currentFile)
    # else:
    os.rename(currentFile, dest)
    print("moved " + justName + " to " + dest)


def buttonClick(c):
    moveToFolder(c)
    displayImage()


if __name__ == '__main__':
    main()
