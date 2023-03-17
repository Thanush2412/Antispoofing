import cv2
import numpy as np
import dlib

# Load the face detection model
detector = dlib.get_frontal_face_detector()

# Load the facial landmarks model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the face recognition model
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the image to be checked
img = cv2.imread("image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Iterate through each detected face
for face in faces:

    # Get the facial landmarks for the face
    landmarks = predictor(gray, face)

    # Get the face descriptor for the face
    face_descriptor = face_recognition_model.compute_face_descriptor(img, landmarks)

    # Convert the face descriptor to a numpy array
    face_descriptor_array = np.array(face_descriptor)

    # Print the face descriptor array
    print(face_descriptor_array)
