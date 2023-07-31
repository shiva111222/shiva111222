''''
Title: Toy vs. Not Toy Image Classification with Convolutional Neural Network (CNN)

Description:
This Python script uses a Convolutional Neural Network (CNN) to classify images as either "Toy" or "Not Toy." The script loads a dataset of toy and non-toy images, preprocesses them, and trains the CNN using data augmentation for improved generalization. The CNN architecture consists of convolutional layers with ReLU activation, max-pooling layers, and fully connected layers with softmax activation for classification.

The script performs the following steps:

    1) Loads the toy and non-toy images from the specified directory
    2) Randomly shuffles the image dataset for training.
    3) Converts the images into arrays, normalizes pixel values to [0, 1], and separates the data into training and testing sets.
    4) Encodes the labels (toy and non-toy) as one-hot vectors.
    5) Sets up an image data generator with data augmentation to enhance the CNN's ability to generalize
    6) Constructs the CNN model using the LeNet architecture.
    7) Compiles the model with binary cross-entropy loss and the Adam optimizer.
    8) Trains the CNN on the training data using data augmentation for the specified number of epochs.
    9) Saves the trained CNN model to a file for later use.
'''

#!/usr/bin/env python3

#######################################
Run the below command if your system does not have Nvidia GPU
export CUDA_VISIBLE_DEVICES=-1
#######################################

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD, Adadelta, Adagrad
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend as K
from imutils import paths  # <-- Make sure imutils is installed
import numpy as np
import random
import cv2
import os, time


class LeNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model
		model = Sequential()
		inputShape = (height, width, depth)

		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		# First layer is a convolution with 20 functions and a kernel size of 5x5 (2 neighbor pixels on each side)
		model.add(Conv2D(20, (5, 5), padding="same",
			input_shape=inputShape))
		# our activation function is ReLU (Rectifier Linear Units)
		model.add(Activation("relu"))
		# second layer is maxpooling 2x2 that reduces our image resolution by half
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Third Layer - Convolution, twice the size of the first convoltion
		model.add(Conv2D(40, (5, 5), padding="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# Fifth Layer is Full connected flattened layer that makes our 3D images into 1D arrays
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the completed network architecture
		return model

# data files
imageDirectory = "images/"
model_file = "toy_not_toy.model"

# Set up the number of training passes (epochs), learning rate, and batch size
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# initialize the data and labels
print("Loading training images set...")
data = []
labels = []

# grab the images from the directories and randomly shuffle them
imagePaths = sorted(list(paths.list_images(imageDirectory)))
# use a random seed number from the time clock
seed = int(time.time() % 1000)
random.seed(seed)
random.shuffle(imagePaths)
print("number of images", len(imagePaths))
print(imagePaths[0])

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (128, 128))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	print(label)
	if label == "images/toys":
		label = 1
		print("Make Me a Toy")
	else:
		label = 0
	labels.append(label)

# Normalize the values of the pixels to be between 0 and 1 instead of 0 to 255
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split the data between a testing set and a training set
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, random_state=seed)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2,
	height_shift_range=0.2, shear_range=0.1, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the weights of the network
print("compiling CNN network...")
cnNetwork = LeNet.build(width=128, height=128, depth=3, classes=2)
#use adam optimizer.  tried SGD for comparison - needs more epochs (>100)
opt = Adam(learning_rate=LEARNING_RATE)
#opt = SGD(lr=LEARNING_RATE,decay=LEARNING_RATE / EPOCHS)

# if we have more than two classes, use categorical crossentropy
cnNetwork.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network - here is where the magic happens...
print("training network...")
print("length trainx", len(trainX)," length trainy ",len(trainY))

H = cnNetwork.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
	epochs=EPOCHS, verbose=1)

# save the CNN network weights to file
print("Saving Network Weights to file...")
cnNetwork.save(model_file)
