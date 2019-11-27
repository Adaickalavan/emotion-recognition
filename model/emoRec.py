# -------------------------------------------------------------------
# Setup
import os
import sys
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
ROOT = os.getenv("ROOT")
sys.path.append(ROOT)

# Import local packages
from tfgraph import cnn
from model.analysis import predictions

# Import standard and downloaded packages
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------

# Variables
num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 1

# Path for saving TensorFlow Serving
savedModelPath = ROOT+'/tfserving/cnn/1/'

# Dataset
# Read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

with open(ROOT+"/dataset/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("Number of instances: ", num_of_instances)
print("Instance length: ", len(lines[1].split(",")[1].split(" ")))

# -------------------------------------------------------------------

# Main function
def main():
    # Initialize trainset and test set
    x_train, y_train, x_test, y_test = [], [], [], []

    # Transfer train and test set data
    for i in range(1, num_of_instances):
        try:
            emotion, img, usage = lines[i].split(",")

            val = img.split(" ")

            pixels = np.array(val, "float32")

            emotion = tensorflow.keras.utils.to_categorical(emotion, num_classes)

            if "Training" in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            elif "PublicTest" in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except:
            print("end")

    # Data transformation for train and test sets
    x_train = np.array(x_train, "float32")
    y_train = np.array(y_train, "float32")
    x_test = np.array(x_test, "float32")
    y_test = np.array(y_test, "float32")

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_train = x_train.astype("float32")
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
    x_test = x_test.astype("float32")

    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Build TensorFlow graph
    model = cnn.graph(num_classes)

    # Batch process
    gen = image.ImageDataGenerator()
    train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

    fit = False
    if fit == True:
        # model.fit_generator(x_train, y_train, epochs=epochs) #train for all trainset
        model.fit_generator(
            train_generator, 
            steps_per_epoch=batch_size, 
            epochs=epochs
        )  # train for randomly selected one
    else:
        model.load_weights(ROOT+"/model/checkpoint/emotion_recognition_weights.h5")  # load weights

    # Overall evaluation
    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', 100*score[1])

    monitor_testset_results = True
    if monitor_testset_results == True:
        # make predictions for test set
        predictionList = model.predict(x_test)

        index = 0
        for ii in predictionList:
            if index >= 20 and index < 30:
                # print(ii) #predicted scores
                # print(y_test[index]) #actual scores

                testing_img = np.array(x_test[index], "float32")
                testing_img = testing_img.reshape([48, 48])

                plt.gray()
                plt.imshow(testing_img)
                plt.show()

                print(ii)
                predictions.barChart(ii)
                print("----------------------------------------------")
            
            index = index + 1

            if index > 30:
                break

    return model        

# Make prediction for custom image out of test set
def customImageTest(model):
    img = image.load_img(ROOT+"/dataset/happy-1.jpg", grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x /= 255

    custom = model.predict(x)
    predictions.barChart(custom[0])

    x = np.array(x, "float32")
    x = x.reshape([48, 48])

    plt.gray()
    plt.imshow(x)
    plt.show()

def saveModel():
    #Read-in model
    modelStructPath = ROOT+'/model/checkpoint/emotion_recognition_structure.json'
    model = model_from_json(open(modelStructPath, "r").read())

    #Read-in weights
    modelWeightPath = ROOT+'/model/checkpoint/emotion_recognition_weights.h5'
    model.load_weights(modelWeightPath)

	# Export the model to a SavedModel
    print("TF version=",tensorflow.__version__)
    tensorflow.keras.models.save_model(
        model,
        savedModelPath,
        overwrite=True,
        include_optimizer=True,
        save_format='tf',
        signatures=None,
        options=None
    )


if __name__ == "__main__":
    model = main()
    customImageTest(model)
    # saveModel()
