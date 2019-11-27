# This file is a poorly written version of `emoRecStream.py` with 
# unnecessary `for` loops and with poor usage of parallelization constructs.
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

# Import standard and downloaded packages
import numpy as np
import cv2
import json
from pprint import pprint
import tensorflow
from tensorflow.keras.preprocessing import image
# -------------------------------------------------------------------

# Path for saving TensorFlow Serving
savedModelPath = ROOT+'/tfserving/cnn/1/'

# Read-in labels
with open(ROOT+'/lib/emonetLabels.json') as f:
	labels = json.load(f)
labels = convert.jsonKeys2int(labels)

def main():
	# Initialize video capture
	cap = cv2.VideoCapture(0)

	# Read-in TensorFlow serving model
	model = tensorflow.keras.models.load_model(savedModelPath)

	# Read-in preprocessing file
	cascadeClas = ROOT+'/lib/haarcascade_frontalface_default.xml'
	face_cascade = cv2.CascadeClassifier(cascadeClas)

	while(True):
		# Capture face
		ret, img = cap.read()
		
		# Detect faces
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)

		for (x,y,w,h) in faces:
			# Draw rectangle to main image
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) 
			
			# Normalize face
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
			detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
			
			# Prediction probabilities
			predictions = model.predict(img_pixels)
			
			# Find maximum index. 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
			max_index = np.argmax(predictions[0])
			
			# Prediction
			emotion = labels[max_index]
			
			# Write emotion text above rectangle
			cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
			
		cv2.imshow('img',img)

		# Press q to quit
		if cv2.waitKey(1) & 0xFF == ord('q'): 
			break

	# Close OpenCV windows		
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
