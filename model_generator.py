import os
import csv
import cv2
import numpy as np
import sklearn

#dir = '/home/jidou/Data/sim_linux/sim_linux_Data/el1'
dir = '/home/jidou/Data/udacity/data'       #the dir of the training data

samples = []
with open(os.path.join(dir,'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
def generator(samples, batch_size=32):
    print("generator samples_length={}".format(len(samples)))
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image_center_path = os.path.join(cddir, line[0])
                image_left_path = os.path.join(dir, str.lstrip(line[1]))
                image_right_path = os.path.join(dir, str.lstrip(line[2]))

                image_center = cv2.imread(image_center_path)
                image_left = cv2.imread(image_left_path)
                image_right = cv2.imread(image_right_path)
                if image_center is None or image_left is None or image_right is None: continue
                correction = 0.2
                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                assert (image_center.shape == (160, 320, 3))

                images.append(image_center)
                angles.append(steering_center)

                if steering_center > 0.15:
                    images.append(image_left)
                    angles.append(steering_left)
                elif steering_center < -0.15:
                    images.append(image_right)
                    angles.append(steering_right)
                else:
                    images.append(cv2.flip(image_center, 1))
                    angles.append(-steering_center)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


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


model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=10)


print("model.save model.h5")
model.save('model.h5')