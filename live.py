import cv2 as cv
import torch
from network import Net

# from cascade import create_cascade
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline


def live(args):
    if args.border is None:
        print("Live mode requires a border value to be set")
        exit()

    args.border = float(args.border)
    if args.border < 0 or args.border > 1:
        print("Border must be between 0 and 1")
        exit()

    checkpoint = torch.load("model.pt")
    net = Net(len(checkpoint["classes"]))
    net.load_state_dict(checkpoint["model"])
    net.eval()

    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv.VideoCapture(0)
    while True:
        _, frame = cap.read()

        border = int(min(frame.shape[:2]) * args.border)

        frame = cv.copyMakeBorder(
            frame, border, border, border, border, cv.BORDER_REFLECT
        )

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
            face = ValidationTransform(face)

            output = net(face)
            output = torch.argmax(output, dim=1).item()

            name = checkpoint["classes"][output]

            font = cv.FONT_HERSHEY_SIMPLEX

            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv.rectangle(frame, (x, y - 32), (x + w, y), (255, 0, 0), -1)
            cv.putText(
                frame, name, (x + 8, y - 6), font, 1, (255, 255, 255), 2, cv.LINE_AA
            )

        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q"):
            break

    # TODO:
    #   Load the model checkpoint from a previous training session (check code in train.py)
    #   Initialize the face recognition cascade again (reuse code if possible)
    #   Also, create a video capture device to retrieve live footage from the webcam.
    #   Attach border to the whole video frame for later cropping.
    #   Run the cascade on each image, crop all faces with border.
    #   Run each cropped face through the network to get a class prediction.
    #   Retrieve the predicted persons name from the checkpoint and display it in the image
