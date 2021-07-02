# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import tkinter
import math
from tkinter import*
import tkinter.messagebox
from PIL import ImageTk, Image
import cv2
import wget
import argparse
from tkinter import filedialog
from tkinter import messagebox
import smtplib

from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import tkinter.filedialog as tkFileDialog

from email.message import EmailMessage


def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
    # define the list of age buckets our age detector will predict
    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
                   "(38-43)", "(48-53)", "(60-100)"]

    # initialize our results list
    results = []

    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConf:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the ROI of the face
            face = frame[startY:endY, startX:endX]

            # ensure the face ROI is sufficiently large
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            # construct a blob from *just* the face ROI
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                             (78.4263377603, 87.7689143744,
                                              114.895847746),
                                             swapRB=False)

            # make predictions on the age and find the age bucket with
            # the largest corresponding probability
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
            i = preds[0].argmax()
            age = AGE_BUCKETS[i]
            ageConfidence = preds[0][i]

            # construct a dictionary consisting of both the face
            # bounding box location along with the age prediction,
            # then update our results list
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
            }
            results.append(d)

    # return our results to the calling function
    return results


def detect_and_predict_mask(frame, faceNet, MaskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = MaskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# python detect_mask_video.py --face face_detector --age age_detector
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", required=True,
                help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
                help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# load our serialized age detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    start_point = (15, 15)
    end_point = (370, 80)
    thickness = -1
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # predict the age
    results = detect_and_predict_age(
        frame, faceNet, ageNet, minConf=args["confidence"])
    for r in results:
        # draw the bounding box of the face along with the associated
        # predicted age
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
        (startX, startY, endX, endY) = r["loc"]
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutmask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text

        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutmask) * 100)
        # Defining ROI
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        label = "Mask" if mask > withoutmask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        if(label == 'No Mask'):

            image = cv2.rectangle(frame, start_point,
                                  end_point, (0, 0, 255), thickness)
            cv2.putText(image, label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            cv2.imwrite("./Output/detected.jpg", frame)
            messagebox.showwarning(
                "Warning", "Access Denied. Please wear a Face Mask")

            msg = EmailMessage()
            msg['Subject'] = 'Subject - Attention!! Someone violated our facemask policy.'
            msg['From'] = 'adityasawant979@gmail.com'
            msg['To'] = 'adityasawant979@gmail.com'
            msg.set_content(
                'A person has been detected without a face mask. Below is the attached image of that person.Please Alert the Authorities.\n'
            )

            with open("Output/detected.jpg", "rb") as f:
                fdata = f.read()
                fname = f.name
                msg.add_attachment(fdata, maintype='Image',
                                   subtype="jpg", filename=fname)

            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login('2018.aditya.sawant@ves.ac.in', 'fhsxjsxsnj')
                smtp.send_message(msg)
            print('[INFO] alert mail Sent to authorities')
        elif(label == 'Mask'):
            image = cv2.rectangle(frame, start_point,
                                  end_point, (0, 255, 0), thickness)
            cv2.putText(image, label, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
            pass
            break
        else:
            print("Invalid")
        print("[INFO] saving image...")
        cv2.imwrite("./Output/detected.jpg", frame)


# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# autopep8 -i detect_mask_video.py
# python detect_mask_video.py --face face_detector --age age_detector
