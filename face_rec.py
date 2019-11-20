import face_recognition
import cv2
import numpy as np
# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import microgear.client as microgear
import os
from os import listdir
from os.path import isfile, join
from imutils.video import VideoStream
import imutils
from sklearn.externals import joblib

#=======NETPIE=========
key = '7QtCCxcGV5ToUPF'
secret = 'z9cSiPuCDcincC0b3gKhUesZn'
app = 'FaceRecM'

microgear.create(key,secret,app,{'debugmode': True})
def connection():
    print("Now I am connected with netpie")

def subscription(topic,message):
    print(topic+" "+message)

def disconnect():
    print("disconnected")

microgear.setalias("facerec")
microgear.on_connect = connection
microgear.on_message = subscription
microgear.on_disconnect = disconnect
microgear.connect()

#=======Face_Rec=========
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
frameSize = (640, 480)
areaFrame = frameSize[0] * frameSize[1]
MinCountourArea = areaFrame * 0.0111  #Adjust ths value according to your usage
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()

#camera.crop = (0.25, 0.25, 0.5, 0.5)
camera.crop = (0.22, 0.10, 0.40, 0.40)
#camera.rotation = 180
camera.hflip = False
camera.resolution = (640, 480)
camera.framerate = 30
camera.brightness = 65
#camera.roi = (0.5,0.5,0.25,0.25)
#camera.brightness = 60
rawCapture = PiRGBArray(camera, size=(640, 480))
# Get a reference to webcam #0 (the default one)
namefile = None
if args["video"] is None :
    #camera = VideoStream(src=0, usePiCamera=True, resolution=frameSize, framerate=15).start()
    namefile = 'camera'
else :
    camera = cv2.VideoCapture(args["video"])
    namefile = args["video"]

#load model
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

# Create arrays of known face encodings and their names
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
motion_flag = False
idle_time = 0
namePre = ""
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    if args["video"] is None :
        #frame = camera.read()
        #frame = imutils.rotate(frame, 180)
        frame = image.array
    else:
        _,frame = camera.read()
        if (frame is None):
            # not connect camera
            break
        x = 300
        y = 300
        frame = frame[y:y+450, x:x+800]
        frame = imutils.resize(frame, width=frameSize[0])

    # Grab a single frame of video
    # Only process every other frame of video to save time
    if process_this_frame :
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            class_probabilities = loaded_model.predict_proba([face_encoding])
            prob = np.max(class_probabilities[0])
            if prob>=0.7:
                index = np.argmax(class_probabilities[0])                
                name = loaded_model.classes_[index]
                name = str(name)+str(prob)
            else:
                name = "Unknown"+str(prob)
            print(name)
            face_names.append(name)  
            myString = ",".join(face_names)
            microgear.publish("/namePeople",myString,{'retain':True})
    process_this_frame = False
    if idle_time%5==0:
        process_this_frame = True
        
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    idle_time += 1
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
#rawCapture.release()
cv2.destroyAllWindows()
    
