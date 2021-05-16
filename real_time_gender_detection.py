from json import load
import tensorflow as tf
import numpy as np 
import cv2 as cv
from keras.models import load_model
import argparse
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.generic_utils import default
import imutils
from tensorflow.keras.preprocessing.image import img_to_array

parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--cascade', help= 'Path to cascade', default= 'data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('-m', '--model', help= 'Path to pre-trained gender detector',default= 'TrainedModels_Serialized\gender_classifier_celeba.h5' )
parser.add_argument('-v', '--video', help= 'Path to video file(optional)')
args = vars(parser.parse_args())

model = load_model(args['model'])
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not args.get('video', False):
    camera = cv.VideoCapture(0)

else:
    camera = cv.VideoCapture(args['video'])

if not camera.isOpened():
    print("Cannot open camera")
    exit()

while True:

    (ret, frame) = camera.read()

    if args.get('video') and not ret:
        break

    frame = imutils.resize(frame, width= 300)
    frameClone = frame.copy()

    rects = face_cascade.detectMultiScale(frame, scaleFactor= 1.1, minNeighbors= 5, minSize= (30, 30), flags= cv.CASCADE_SCALE_IMAGE)

    for(fx, fy, fw, fh) in rects:

        roi = frame[fy - 20: fy + fh , fx - 20: fx + fw]
        roi= cv.resize(frame, (64, 64))
        roi = roi.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        pred_logits = model.predict(roi)
        male = tf.sigmoid(pred_logits)
        label = 'Male' if male>0.5 else 'Female'

        cv.putText(frameClone, label, (fx, fy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv.rectangle(frameClone, (fx , fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv.imshow('Face', frameClone)

        if(cv.waitKey(1) & 0xFF == ord('q')):
            break

camera.release()
cv.distroyAllWindows()