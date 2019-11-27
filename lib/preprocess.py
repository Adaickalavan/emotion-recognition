import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from functools import partial 

def detectObject(object_cascade, img):
    """
    Finds the bounding boxes around objects within an image.
    
    Parameters
    ----------
    object_cascade : cv2.CascadeClassifier
        Cascade classifier to detect object.
    img : numpy.ndarray
        3 dimensional array representing a BGR image.
    
    Returns
    -------
    numpy.ndarray
        Two dimensional array of shape (n,4). Here, n represents the number of detected objects. Each object has 4 coordinates representing its bounding box.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detectMultiScale()
    # Returns a numpy array if objects found. 
    # Returns a tuple if no objects found.
    objects = object_cascade.detectMultiScale(gray, 1.3, 5) 
    # print("Objects found",len(objects))
    if len(objects)==0:
        objects = np.array([])
    return objects

def normFaces(img, faces):
    """
    Crop, convert to grayscale, resize, and normalize, faces identified by the bounding boxes in input `img`.

    Parameters
    ----------
    img : numpy.ndarray
        3 dimensional array representing a BGR image.
    faces : numpy.ndarray
        Two dimensional array of shape (n,4). Here, n represents the number of detected faces. Each face has 4 coordinates representing its bounding box.
    
    Returns
    -------
    numpy.ndarray
        3 dimensional array of shape (n,48,48). Here, n represents the number of detected faces. Each face is cropped, converted to grayscale, and normalized to 48pixel-by-48pixel.
    """

    normFacePart = partial(__normFace, img)
    pixelsIterator = map(normFacePart, faces)
    normalizedFaces = np.array(list(pixelsIterator))

    return normalizedFaces

def __normFace(img, face):
    """
    Crop, convert to grayscale, resize, and normalize, a single face identified by the bounding box in input `img`.
    
    Parameters
    ----------
    img : numpy.ndarray
        3 dimensional array representing a BGR image
    face : numpy.ndarray
        Two dimensional array of shape (1,4). It contains the bounding box coordinates of a single detected face.

    Returns
    -------
    numpy.ndarray
        2 dimensional array of shape (48,48). A single face which has been cropped, converted to grayscale, resized, and normalized.
    """
    
    detected_face = img[int(face[1]):int(face[1]+face[3]), int(face[0]):int(face[0]+face[2])] #crop detected face
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

    img_pixels = image.img_to_array(detected_face)
    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    
    return img_pixels
