import os
import csv
import cv2
import numpy as np

#dir = '/home/jidou/Data/sim_linux/sim_linux_Data/el1'
dir = '/home/jidou/Data/udacity/data'

lines = []
with open(os.path.join(dir,'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

i = 0
max_line = 300000

n_count_left = 0
n_count_right = 0
images = []
measurements = []
for line in lines[1:]:
    image_center_path = os.path.join(dir, line[0])
    image_left_path = os.path.join(dir, str.lstrip(line[1]))
    image_right_path = os.path.join(dir, str.lstrip(line[2]))

    image_center = cv2.imread(image_center_path)
    image_left = cv2.imread(image_left_path)
    image_right = cv2.imread(image_right_path)
    if image_center is None or image_left is None or image_right is None:
        print("{}   {}  {}".format(image_center_path, image_left_path, image_right_path))
        continue
    correction = 0.2
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    assert (image_center.shape == (160, 320, 3))

    #images.append(image)
    images.append(image_center)
    measurements.append(steering_center)

    if steering_center > 0.15:
        n_count_left += 1
        images.append(image_left)
        measurements.append(steering_left)
    elif steering_center < -0.15:
        images.append(cv2.flip(image_center, 1))
        measurements.append(-steering_center)

        n_count_right += 1
        images.append(image_right)
        measurements.append(steering_right)
    else:
        images.append(cv2.flip(image_center, 1))
        measurements.append(-steering_center)

    i += 1
    if i > max_line: break

assert(len(images)==len(measurements))
print("training data length = {}".format(len(images)))
print("too left is {} and too right is {}".format(n_count_left, n_count_right))

augmented_images, augmented_measurement = [], []
for image,measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurement.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurement.append(-measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurement)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
''' #lenet
model.add(Convolution2D(6,6,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,6,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))
'''

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

print("model.save model.h5")
model.save('model.h5')