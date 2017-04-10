# **Traffic Sign Recognition** 

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

[image1]: ./result/bar_chart.png "Visualization"
[image2_1]: ./result/before_grayscale.png "Before Grayscaling"
[image2_2]: ./result/after_grayscale.png "After Grayscaling"
[image3_1]: ./result/before_augmentation.png "Before Augmentation"
[image3_2]: ./result/after_augmentation.png "After Augmentation"
[image4]: ./test/1.jpg "Traffic Sign 1"
[image5]: ./test/2.jpg "Traffic Sign 2"
[image6]: ./test/3.jpg "Traffic Sign 3"
[image7]: ./test/4.jpg "Traffic Sign 4"
[image8]: ./test/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/villytiger/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed between classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because Pierre Sermanet and Yann LeCun in their work mentioned that information about color doesn't help to recognize signs.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2_1] ![alt text][image2_2]

As a last step, I normalized the image data because images seems different in terms of brightness and contrast.

I decided to generate additional data because it could help to avoid overfitting by learning from general characteristics of traffic signs.

To add more data to the the data set, I used the following techniques because it can help to not rely on the exact location.

Here is an example of an original image and an augmented image:

![alt text][image3_1] ![alt text][image3_2]

The difference between the original data set and the augmented data set is the following:
1. image rotated  from -15 to +15 degrees;
2. image zoomed from 0.9 to 1.1 of original size;
3. image shifted from -2 to +2 pixels in horizontal and vertical axis.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 |     Description                               |
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 grayscale image                       | 
| Convolution 5x5       | 1x1 stride, same padding, outputs 32x32x108   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 16x16x108                |
| Convolution 5x5       | 1x1 stride, same padding, outputs 16x16x108   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 8x8x108                  |
| Fully connected       | with 50 units                                 |
| RELU                  |                                               |
| Fully connected       | with 43 units                                 |
| Softmax               |                                               |

I used dropout and L2 regularization to avoid overfitting. The best performing probability before second convolution is 0.8 and before each fully connected layer it is 0.5.


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 2 stages. On first stage I used 10 times augmented data from the training set with default parameters of AdamOptimizer. After reaching 0.055 loss on the validation set it stops learning. And then I run learning on data set with 20000 samples for each class with learning rate 0.0001. It stop learning after 0.035 loss on validation set. On each stage I used 256 for batch size.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.995
* test set accuracy of 0.984

* What was the first architecture that was tried and why was it chosen?

At first I tried LeNet architecture. With it I could achieve ~0.9 accuracy on validation set.

* What were some problems with the initial architecture?

I could achieve high accuracy on training set, but it was affected by overfitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I added dropout after each layer, with that I could improve accuracy on validation set by few percents. Removing second hidden layer and using more features on convolutions allowed to get a few more percents.

* Which parameters were tuned? How were they adjusted and why?

I tried different numbers of features on convolutions and units in hidden layer. I also tried to feed output from both convolutions to hidden layer but it performed worse in my case.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Starting with 2 convolution layers and 1-2 hidden fully connected layers one can achieve good results on the training set. The most difficult problem is to avoid overfitting. Dropout layer helps a lot with that. After adding dropout layer it is important to choose right network capacity to perform well on both training and validation sets.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because 30 and 80 looks very similar. Because of low resolution or some random noise it could be difficult to distinguish two signs with the difference in small details.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                    |     Prediction                                | 
|:------------------------:|:---------------------------------------------:| 
| Speed limit (100km/h)    | Speed limit (100km/h)                         | 
| Road work                | Road work                                     |
| Yield                    | Yield                                         |
| General caution          | General caution                               |
| Speed limit (30km/h)     | Speed limit (30km/h)                          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th cell of the Ipython notebook.

For the first image, the model is pretty sure that this is a 100km/h sign (probability of 0.98), and the image does contain a 100km/h sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:-----------------------:|:---------------------------------------------:| 
| .98                     | Speed limit (100km/h)                         | 
| .01                     | Speed limit (120km/h)                         |
| .0009                   | Speed limit (70km/h)                          |
| .0007                   | Speed limit (30km/h)                          |
| .0001                   | Speed limit (80km/h)                          |


For the second image, the model is pretty sure that this is a road work sign (probability of 0.98), and the image does contain a road work sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:-----------------------:|:---------------------------------------------:| 
| .99                     | Road work                                     | 
| .00000008               | Bumpy road                                    |
| .00000008               | Turn left ahead                               |
| .000000008              | Beware of ice/snow                            |
| .000000001              | Bicycles crossing                             |


For the third image, the model is absolutely sure that this is a yield sign (probability of 1.00), and the image does contain a yield sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:-----------------------:|:---------------------------------------------:| 
| 1.                      | Yield                                         | 
| .00000000001            | Turn right ahead                              |
| .000000000007           | End of speed limit (80km/h)                   |
| .000000000002           | End of no passing                             |
| .000000000001           | Priority road                                 |


For the fourth image, the model is absolutely sure that this is a general caution sign (probability of 1.00), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:-----------------------:|:---------------------------------------------:| 
| 1.                      | General caution                               | 
| .000000000001           | Pedestrians                                   |
| .00000000000008         | Traffic signals                               |
| .00000000000000003      | Speed limit (70km/h)                          |
| .00000000000000001      | Road narrows on the right                     |


For the first image, the model is pretty sure that this is a 30km/h sign (probability of 0.99), and the image does contain a 30km/h sign. The top five soft max probabilities were

| Probability             |     Prediction                                | 
|:-----------------------:|:---------------------------------------------:| 
| .99                     | Speed limit (30km/h)                          | 
| .0001                   | Speed limit (50km/h)                          |
| .000009                 | Speed limit (80km/h)                          |
| .0000009                | Speed limit (100km/h)                         |
| .0000002                | Speed limit (20km/h)                          |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


