import matplotlib.pyplot as plt
import numpy as np

# Function for drawing bar chart for emotion preditions
def barChart(emotions):
    objects = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, emotions, align="center", alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel("percentage")
    plt.title("emotion")

    plt.show()