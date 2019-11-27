"""
Script: /example/emoRecStream.py

This script captures video from an attached webcam at video device port 0.  
A machine learning algorithm classifies the facial expression for each detected 
face into one of 7 categories, namely, 

* "0": "angry",
* "1": "disgust",
* "2": "fear",
* "3": "happy",
* "4": "sad",
* "5": "surprise",
* "6": "neutral". 

The captured frames are displayed in a window with bounding boxes and predicted 
emotion labels around any deteced faces.

This file can also be imported as a module and contains the following
functions:

* main - the main function of the script

"""

# -------------------------------------------------------------------
# Setup
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)

# Import local packages
from model.helper import convert
from lib import preprocess

# Import standard and downloaded packages
import numpy as np
import cv2
import json
from pprint import pprint
import tensorflow
from tensorflow.keras.preprocessing import image
# -------------------------------------------------------------------

# Path for saving TensorFlow Serving
savedModelPath = ROOT+'tfserving/cnn/1/'

# Read-in labels
with open(ROOT+'lib/emonetLabels.json') as f:
	labels = json.load(f)
labels = convert.jsonKeys2int(labels)

def main():
	# Initialize video capture
	cap = cv2.VideoCapture(os.getenv("VIDEO"))

	# Read-in TensorFlow serving model
	model = tensorflow.keras.models.load_model(savedModelPath)

	# Read-in preprocessing file
	cascadeClas = ROOT+'lib/haarcascade_frontalface_default.xml'
	face_cascade = cv2.CascadeClassifier(cascadeClas)

	while(True):
		# Capture face
		ret, img = cap.read()
		# Detect faces
		faces = preprocess.detectObject(face_cascade, img)
		# Normalize faces
		normalizedFaces = preprocess.normFaces(img, faces)
				
		if len(normalizedFaces) != 0: 		
			# Predict	
			predictions = model.predict(normalizedFaces)		
			# Find maximum indices. 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions,axis=1)
			# Draw bounding boxes and print emotion labels
			for num, face in zip(max_index, faces):
				cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),(255,0,0),2)
				cv2.putText(img, labels[num], (int(face[0]), int(face[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		
		cv2.imshow('img',img)

		# Press q to quit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Close OpenCV windows		
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
