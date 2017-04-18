#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

**My english is not very good**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* video.mp4 is my result

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
```
the autonomously drive images saved in folder run1, and then

```sh
python video.py run1
```
create the file run1.mp4, it's my result

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. 
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 11 - NVIDIA Architecture,
whish is described at [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![alt text](examples/1.png)

all the code is in model.py lines 78-89

before the convolution neural network

and the data is normalized in the model using a Keras lambda layer (code line 76 in model.py).

and the image is cut off the useless part(code line 77 in model.py)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 85 and line 87). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (normalized the image code line 76 in model.py). 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road the left camera image and the right camera image

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use lenet
![alt text](examples/lenet.png)

in the code line 91-98 in model.py which is commented.

My first step was to use a convolution neural network model similar to the Lenet, 
I thought this model might be appropriate because success use lenet at project II
(CarND-Traffic-Sign-Classifier-Project) 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that use 11 - NVIDIA Architecture

The final step was to run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track flip of the image
to improve the driving behavior in these cases, I use left camera image when the car steering left and use the right camera image when the car steering right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 78-89) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

![alt text](examples/1.png)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stear left and right at specail status. 

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would reduce bias,

but i found if all images use flip and left camera image and rigt camera image, the result is overfit, but good.

so i only use left cemera image when the car is too left, and i only use right cemera image when then car is too right, 
and i only use flip image when the car is neither too left nor too righ, so the result is much more better

After the collection process, I had X number of data points. I then preprocessed this data by split training set and validation set 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. 
The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was Z as evidenced by 10
I used an adam optimizer so that manually training the learning rate wasn't necessary.
