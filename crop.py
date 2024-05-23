import os
import cv2 as cv
import argparse
import shutil
import random

def process_image(image_path, border_factor):
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Error reading image {image_path}")
        return None

    height, width = frame.shape[:3]
    top_border = int(border_factor * height)
    left_border = int(border_factor * width)
    
    frame = cv.copyMakeBorder(
        frame, top_border, top_border, left_border, left_border, cv.BORDER_REFLECT
    )
    return frame

def save_image(image, output_path):
    cv.imwrite(output_path, image)

def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def crop_images(base_folder, border, split):
    train_folder = 'train'
    val_folder = 'val'

    clear_directory(train_folder)
    clear_directory(val_folder)

    for root, dirs, files in os.walk(base_folder):
        if not files:
            continue

        folder_name = os.path.basename(root)
        train_output_folder = os.path.join(train_folder, folder_name)
        val_output_folder = os.path.join(val_folder, folder_name)
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(val_output_folder, exist_ok=True)

        for file in files:
            if not file.endswith('.png'):
                continue
            
            image_path = os.path.join(root, file)
            processed_image = process_image(image_path, border)
            if processed_image is None:
                continue

            if random.random() < split:
                output_path = os.path.join(val_output_folder, file)
            else:
                output_path = os.path.join(train_output_folder, file)

            save_image(processed_image, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop images with border and split into train/val.')
    parser.add_argument('crop', help='Crop command')
    parser.add_argument('--border', type=float, required=True, help='Border factor as a percentage of width/height')
    parser.add_argument('--split', type=float, required=True, help='Fraction of images to go into val folder')
    args = parser.parse_args()

    if args.crop == 'crop':
        base_folder = 'objects'
        crop_images(base_folder, args.border, args.split)
