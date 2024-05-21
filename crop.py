import cv2 as cv
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER
import os
import csv
import random

# Quellen
#  - How to iterate over all files/folders in one directory: https://www.tutorialspoint.com/python/os_walk.htm
#  - How to add border to an image: https://www.geeksforgeeks.org/python-opencv-cv2-copymakeborder-method/

# This is the cropping of images
def crop(args):
    # TODO: Crop the full-frame images into individual crops

    #   Create the TRAIN_FOLDER and VAL_FOLDER is they are missing (os.mkdir)
    if not os.path.exists(TRAIN_FOLDER):
        os.mkdir(TRAIN_FOLDER)

    if not os.path.exists(VAL_FOLDER):
        os.mkdir(VAL_FOLDER)

    #   Clean the folders from all previous files if there are any (os.walk)
    def clean_folder(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for name in files:
                os.remove(os.path.join(root, name))

    clean_folder(TRAIN_FOLDER)
    clean_folder(VAL_FOLDER)

    #   Iterate over all object folders and for each such folder over all full-frame images 
    for person in os.listdir(ROOT_FOLDER):
        person_folder_path = os.path.join(ROOT_FOLDER, person)
        if os.path.isdir(person_folder_path):
            for root, dirs, files in os.walk(person_folder_path):
                #   Read the image (cv.imread) and the respective file with annotations you have saved earlier (e.g. CSV)
                for file in files:
                    if file.endswith(('.png')):
                        frame = cv.imread(file)
                        frame = cv.copyMakeBorder(frame, top_border, top_border, left_border, left_border, cv.BORDER_REFLECT)

    #   Attach the right amount of border to your image (cv.copyMakeBorder)
    #   Crop the face with border added and save it to either the TRAIN_FOLDER or VAL_FOLDER
    #   You can use 
    #
    #       random.uniform(0.0, 1.0) < float(args.split) 
    #
    #   to decide how to split them.



    if args.border is None:
        print("Cropping mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()


