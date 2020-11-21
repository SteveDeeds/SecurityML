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

window = tk.Tk()
currentFile = ""
panel1 = tk.Label()
panel2 = tk.Label()

settings = SMLsettings.getSettings()

global srcPath
srcPath = os.path.join(*settings["FrameDestinationPath"])
srcPath = 'F:\\SecurityML\\data\\machine\\unsure'
global dstPath
dstPath = os.path.join(*settings["TrainingPath"])
trainPath = os.path.join(*settings["TrainingPath"])
global classes
classes = os.listdir(trainPath)
global files

# def browse_src_button():
#     global srcPath
#     foldername = filedialog.askdirectory()
#     srcPath.set(foldername)
# def browse_dst_button():
#     global srcPath
#     foldername = filedialog.askdirectory()
#     dstPath.set(foldername)


def displayImage():
    global currentFile
    global files
    currentFile = files.pop()
    #print(os.path.split(currentFile)[1])
    random.shuffle(files)
    currentFile = files.pop()
    img1 = ImageTk.PhotoImage(Image.open(currentFile))
    #img1 = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
    #r,g,b,a = cv2.split(img1)
    #a = ((a>32)*255).astype(np.uint8)
    #a = a.astype(np.uint8)
    #a = (a * 4 + 32).astype(np.int32)
    #maximum = np.full(a.shape, 255).astype(np.int32)
    # cv2.merge((a,maximum))
    #a = np.amin(a)
    #a = a.astype(np.uint8)
    #img1 = cv2.merge((b,g,r,a))
    #img1 = Image.fromarray(img1)
    #img1 = ImageTk.PhotoImage(img1)
    #img2 = cv2.merge((b,g,r))
    #img2 = Image.fromarray(img2)
    #img2 = ImageTk.PhotoImage(img2)
    panel1.configure(image=img1)
    panel1.image = img1
    #panel2.configure(image = img2)
    #panel2.image = img2


def Initilize():
    global window
    global currentFile
    global panel1
    global panel2
    global files
    window.title("Calssify the image")
    # window.geometry("800x1000")
    window.configure(background='grey')
    files = glob.glob(os.path.join(srcPath, '*.png'))
    files = files + glob.glob(os.path.join(srcPath, '*.jpg'))
    random.shuffle(files)
    currentFile = files.pop()
    img1 = ImageTk.PhotoImage(Image.open(currentFile))
    #img1 = cv2.imread(currentFile, cv2.IMREAD_UNCHANGED)
    #r,g,b,a = cv2.split(img1)
    #a = ((a>32)*255).astype(np.uint8)
    #a = a.astype(np.uint8)
    #a = (a * 4 + 32).astype(np.int32)
    #maximum = np.full(a.shape, 255).astype(np.int32)
    # cv2.merge((a,maximum))
    #a = np.amin(a)
    #a = a.astype(np.uint8)
    # img1 = cv2.merge((b,g,r,a))
    # img1 = Image.fromarray(img1)
    # img1 = ImageTk.PhotoImage(img1)
    # img2 = cv2.merge((b,g,r))
    # img2 = Image.fromarray(img2)
    # img2 = ImageTk.PhotoImage(img2)
    #img2 = ImageTk.PhotoImage(Image.open(currentFile.replace('masked','frames')))
    panel1 = tk.Label(window, image=img1)
    #panel2 = tk.Label(window, image = img2)
    i=0
    for c in classes:
        button = tk.Button(window, text=c, command= lambda: buttonClick(c))
        button.grid(row=3, column=i)
        i=i+1

    #srcPathLabel = tk.Label(master=window,textvariable=srcPath)
    #srcButton = tk.Button(text="Browse", command=browse_src_button)

    # pack it
    # srcButton.grid(row=0,column=0)
    # srcPathLabel.grid(row=0,column=1)
    panel1.grid(row=1, column=0, columnspan=10)
    # panel2.grid(row=2,column=0,columnspan=10)

    window.mainloop()


def main():
    for c in classes:
        os.makedirs(os.path.join(dstPath, c), exist_ok=True)
    Initilize()


def moveToFolder(folder):
    justName = os.path.split(currentFile)[1]
    dest = os.path.join(dstPath, folder, justName)
    if(os.path.isfile(dest)):
        os.remove(currentFile)
    else:
        os.rename(currentFile, dest)
    print("moved " + justName + " to " + dest)


def buttonClick(c):
    moveToFolder(c)
    displayImage()

if __name__ == '__main__':
    main()
