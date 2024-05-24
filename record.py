import cv2 as cv
import os
import csv
from common import ROOT_FOLDER

# from cascade import create_cascade

# Quellen
#  - How to open the webcam: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
#  - How to run the detector: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
#  - How to download files from google drive: https://github.com/wkentaro/gdown
#  - How to save an image with OpenCV: https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
#  - How to read/write CSV files: https://docs.python.org/3/library/csv.html
#  - How to create new folders: https://www.geeksforgeeks.org/python-os-mkdir-method/


# This is the data recording pipeline
def record(args):
    if args.folder is None:
        print("Please specify folder for data to be recorded into")
        exit()

    # create objects folder if it doesn't exist already
    if not os.path.exists(ROOT_FOLDER):
        os.mkdir(ROOT_FOLDER)

    # define the output folder with the chosen name (args)
    output_folder = os.path.join(ROOT_FOLDER, args.folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # counter for the image numbers
    counter = 0

    # counter to wait 30 frames before saving the next image
    frame_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_count += 1

        # to only save every 30. frame
        if frame_count == 30:
            frame_count = 0

            # when a face is detected
            if len(faces) == 1:
                # save image to a png file
                cv.imwrite(os.path.join(output_folder, f"face_{counter}.png"), frame)

                # save coordinates to a csv file with the same name as the image file
                with open(
                    os.path.join(output_folder, f"face_{counter}" + ".csv"),
                    "w",
                    newline="",
                ) as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    for x, y, w, h in faces:
                        writer.writerow([x, y, w, h])

                counter += 1

        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the resulting frame
        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    # TODO: Implement the recording stage of your pipeline
    #   Create missing folders before you store data in them (os.mkdir)
    #   Open The OpenCV VideoCapture Device to retrieve live images from your webcam (cv.VideoCapture)
    #   Initialize the Haar feature cascade for face recognition from OpenCV (cv.CascadeClassifier)
    #   If the cascade file (haarcascade_frontalface_default.xml) is missing, download it from google drive
    #   Run the cascade on every image to detect possible faces (CascadeClassifier::detectMultiScale)
    #   If there is exactly one face, write the image and the face position to disk in two seperate files (cv.imwrite, csv.writer)
    #   If you have just saved, block saving for 30 consecutive frames to make sure you get good variance of images.


if __name__ == "__main__":
    record("test")
