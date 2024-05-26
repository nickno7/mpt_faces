import os
import cv2 as cv
import csv
import shutil
import random
from common import ROOT_FOLDER, TRAIN_FOLDER, VAL_FOLDER


def process_image(image_path, border_factor, csv_file_path):
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Error reading image {image_path}")
        return None

    pixels = int(border_factor * min(frame.shape[:2]))

    frame = cv.copyMakeBorder(frame, pixels, pixels, pixels, pixels, cv.BORDER_REFLECT)

    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            x, y, w, h = map(int, row)

    x += pixels
    y += pixels

    cropped_image = frame[y - pixels : y + h + pixels, x - pixels : x + w + pixels]
    return cropped_image


def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def crop(args):
    if args.border is None or not (0 <= float(args.border) <= 1):
        print("Cropping mode requires a border value between 0 and 1.")
        return

    clear_directory(TRAIN_FOLDER)
    clear_directory(VAL_FOLDER)

    for root, dirs, files in os.walk(ROOT_FOLDER):
        if not files:
            continue

        folder_name = os.path.basename(root)
        train_output_folder = os.path.join(TRAIN_FOLDER, folder_name)
        val_output_folder = os.path.join(VAL_FOLDER, folder_name)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)

        for file in files:
            if not file.endswith(".png"):
                continue

            image_path = os.path.join(root, file)
            csv_file_path = f"{os.path.splitext(image_path)[0]}.csv"
            cropped_image = process_image(image_path, float(args.border), csv_file_path)
            if cropped_image is None:
                continue

            if random.random() < float(args.split):
                cv.imwrite(os.path.join(val_output_folder, file), cropped_image)
            else:
                cv.imwrite(os.path.join(train_output_folder, file), cropped_image)
