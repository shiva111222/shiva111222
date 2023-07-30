#!/usr/bin/env python3

# Test Network Program
# part of the toy detector
# uses LeNet to detect if toys are in the image
#
# Francis X. Govers 2018
#
# references:
# https://www.pyimagesearch.com/2017/12/18/keras-deep-learning-raspberry-pi/
#
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt

def main():
    # load the image
    image = cv2.imread("toy.jpg")
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model("toy_not_toy.model")

    # classify the input image
    (not_toy, toy) = model.predict(image)[0]
    print("toy =", toy, "Not Toy =", not_toy)

    # build the label
    label = "Toy" if toy > not_toy else "Not Toy"
    probability = max(toy, not_toy)
    label = "{}: {:.2f}%".format(label, probability * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (200, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (200, 255, 40), 2)

    # show the output image using matplotlib
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Toy Detector")
    plt.axis('off')  # Hide the axis values
    plt.show()

    # save the output image to a file
    cv2.imwrite("toy_classified.jpg", output)

if __name__ == "__main__":
    main()
