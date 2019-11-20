# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
from sklearn.externals import joblib
import os
import numpy as np

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('D:\\J\\Intania\\work\\IOT\\Final\\Belly-Blue\\face_recognition\\train_dir')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("D:\\J\\Intania\\work\\IOT\\Final\\Belly-Blue\\face_recognition\\train_dir\\" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("D:\\J\\Intania\\work\\IOT\\Final\\Belly-Blue\\face_recognition\\train_dir\\" + person + "\\" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains none or more than faces, print an error message and exit
        if len(face_bounding_boxes) != 1:
            print(person + "/" + person_img + " contains none or more than one faces and can't be used for training.")
            exit()
        else:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale',verbose=1,probability=True)
clf.fit(encodings,names)

#save model
filename = 'finalized_model.sav'
joblib.dump(clf, filename)


#====Load model and predict

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('test.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")

#load model
loaded_model = joblib.load(filename)

for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    #name = loaded_model.predict([test_image_enc])
    class_probabilities = loaded_model.predict_proba([test_image_enc])
    print(class_probabilities)
    print(loaded_model.classes_)
    if np.max(class_probabilities[0])>=0.7:
        index = np.argmax(class_probabilities[0])
        print(loaded_model.classes_[index])
    else:
        print("Unknown")
    #print(str(*name) + str(class_probabilities))