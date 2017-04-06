#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/demo.png "Visualization"
[image2]: ./examples/augmentation.png "Augmentation"
[image4]: ./examples/100km.jpg "Traffic Sign 1"
[image5]: ./examples/No_entry.jpg "Traffic Sign 2"
[image6]: ./examples/slippery.jpg "Traffic Sign 3"
[image7]: ./examples/stop.jpg "Traffic Sign 4"
[image8]: ./examples/yield.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

Here is a link to my [project code](https://github.com/xiaosonglu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization example of a random image picked from the data set. It is an image shown "Ahead only" (ClassID 35):

![alt text][image1]

###Design and Test a Model Architecture

####1. Pre-process the Data Set 
As a first step, I decided to convert the images to grayscale because it is easier for training with one channel instead of 3 channels and the grayscale won't affect the training of the signs.
Then I use skimage.exposure.rescale_intensity to stretch or shrink its intensity levels of the images for improving the training. 

I decided to generate additional data in order to further improve the learning performance to reach a higher accuracy rate.  

To add more data to the the data set, I used the following techniques: add two more sets to the training set and each additional data set has the same size of the original set. Then I applied rotation (random rotate in a range of (-10,10) degree) to the images of these two additional sets

Here is an example of a traffic sign image before and after grayscaling and after augmentation based on the changes on intensity and rotation.

![alt text][image2]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Flatten       		| outputs 400        							|
| Fully connected		| outputs 120     								|
| RELU					|												|
| Fully connected		| outputs 84     								|
| RELU					|												|
| Fully connected		| outputs 43     								|
| Softmax				|           									|

 

####3. Describe how you trained your model.

To train the model, I used an Adam Optimizer, EPOCHS = 50, BATCH_SIZE = 128, learning rate is set to 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 0.957 
* test set accuracy of 0.938


I choose a well known architecture LeNet. The reason is that the LeNet architecture performs well on training the network based on 32x32 images from the reformatted MNIST data and the input data in this project also contains 32x32 images. The final model shows that the accuracy of the training set reaches 0.957 and the accuracy of the test set reaches 0.938 after 50 epochs. 

 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The stop sign image might be difficult to classify because the image needs to be resized to 32x32 from 359x478 and the stop sign inside the resized image is not centered and will be stretched. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100 km/h	      		| 30 km/h					 				|
| No Entry     			| No Entry 										|
| Slippery Road			| Slippery Road      							|
| Stop Sign      		| 60 km/h   									| 
| Yield					| Yield											|



The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set. The reason is that the new images needs to be resized and this leads to stretched images. The training data does not contain stretched images as the input data.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The model does predict the "No entry", "Slippery Road" and "Yield" sign correctly. But the 100km/h and Stop sign fails to identify. The 100km/h sign prediction has the second largest softmax probability with the correct sign. The stop sign does not have the correct sign within the top five softmax probabilities due to the structure of the original image as discussed before. 

100km/h: 

	probability [ 0.83,  0.16,  0.01,  0.  ,  0.  ]

	sign [30km/h, 100km/h, 80km/h, 50km/h, End of speed limit (80km/h)]


No entry:
 
	probability [ 1.,  0.,  0.,  0.,  0.]

	sign [No entry, End of no passing,  No passing, Turn left ahead, End of all speed and passing limits]


Slippery road: 

	probability [ 0.68,  0.32,  0.  ,  0.  ,  0.  ]

	sign [Slippery road, Wild animals crossing, Go straight or right, Children crossing, Right-of-way at the next intersection]

Stop: 

	probability [ 1.,  0.,  0.,  0.,  0.]

	sign [ 60km/h,  No passing, Children crossing, Go straight or right, Dangerous curve to the right]

Yield: 

	probability [ 1.,  0.,  0.,  0.,  0.]

	sign [Yield, Ahead only,  20km/h,  30km/h,  50km/h]
 



