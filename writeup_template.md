# Traffic Sign Recognition

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/AllSigns.png "Visualization"
[image1b]: ./examples/Histogram.png "Counts of each sign"
[image2]: ./examples/ImagePreProcessing.png "Grayscaling"
[image3]: ./examples/HistogramAugmented.png "Augmented Histogram"
[image3a]: ./examples/TrainingPlot.png "Training Loss vs. Epoch"
[image4]: ./test_img/TrafficSign01.jpg "Traffic Sign 1"
[image5]: ./test_img/TrafficSign02.jpg "Traffic Sign 2"
[image6]: ./test_img/TrafficSign03.jpg "Traffic Sign 3"
[image7]: ./test_img/TrafficSign04.jpg "Traffic Sign 4"
[image8]: ./test_img/TrafficSign05.jpg "Traffic Sign 5"
[image9]: ./examples/Sign01_Prediction.png "Traffic Sign 1 Prediction Results"
[image10]: ./examples/Sign02_Prediction.png "Traffic Sign 2 Prediction Results"
[image11]: ./examples/Sign03_Prediction.png "Traffic Sign 3 Prediction Results"
[image12]: ./examples/Sign04_Prediction.png "Traffic Sign 4 Prediction Results"
[image13]: ./examples/Sign05_Prediction.png "Traffic Sign 5 Prediction Results"

## Data Set Summary & Exploration

### Basic Summary

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34,799
* The size of the validation set is 12,630
* The size of test set is 4,410
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 4

### 2. Exploration of Dataset

Here is an exploratory visualization of the data set. It is can be seen that there are some classes that have a lot of samples for the training set whereas there are other classes that have a small set of samples by comparison.

![alt text][image1]
![alt text][image1b]

## Design and Test a Model Architecture

### 1.Image Pre-Processing
The original dataset was converted to grayscale and underwent a histogram equalization in order to balance out the variation in brightness between the images.

The dataset was augmented by applying a random rotation and translation. The intent was to create a more level histogram than the original data. The maximum value from the original histogram was determined and each class was augmented by a random percentage (30% to 100%) of the difference between the count of the class and the maximum. This data augmentation may be found in cell 9 of Traffic_Sign_Classifier.ipynb.


Here is an example of a traffic sign image before and after grayscaling, random rotation, and random translation. The code corresponding the image pre-processing steps may be found in cell 8 of Traffic_Sign_Classifier.ipynb.

![alt text][image2]

This is how the updated hisrogram looks after implementing the data augmentation.
![alt text][image3]


### 2. Final Architecture


My final model was based off of the [Sermanet and LeCun architecture](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|	 							|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 1x1x400 	|
| RELU					| 									|
| Flatten       | Flattens 1x1x400  RELU layer |
| Flatten       | Flattens 10x10x16 RELU layer |
| Concatenate   | Concatenates the flattened layers |
| Dropout       | |
| Fully connected		|       									|
| Softmax         |                             |
|||



### 3. Model Training
When training the model, the following configuration was used:

 * Optimizer: Adam
 * Batch Size: 64
 * Epochs: 60
 * Learning Rate: 0.0009
 * Keep Probability: 0.1

### 4. Architecture Development

My final model results were:
* training set accuracy of 97.0%
* validation set accuracy of 93.4%
* test set accuracy of 80.0%

An iterative approach was taken in developing the model. When developing the architecture, I have started with the raw imagery. I have started with the LeNet model and varied the sizes of the hyper parameters. I have also attempted to implement the 2-stage CovNet architecture by Sermanet and LeCun and also have played with the parameters as well. Tuning the parameters was a long process of trial and error. If I could redo the project, I would probably implement it outside of a jupyter notebook and run large batches of scripts to get a data dump to analyze later.

With the raw data (no preprocessing and no augmentation), the LeNet architecutre worked better but had a 93% training set accuracy, and 89% test set accuracy. However, when I implemented data augmentation, I achieved better results with Sermanet and LeCun - 97% training set accuracy and a 93.4% test set accuracy. In the end, the Sermanet and LeCun architecture was chose.


## Test a Model on New Images

### 1. Images from the Web
Below are five German traffic signs that I found on the web:<center>![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]</center>


The first image might be difficult to classify because there are multiple speed limit signs in the dataset (there were classes for the following speeds: 20, 30, 50, 60,70, 80, 100, 120 km/h).

The second image might be difficult to classify because the keep right sign is very similar to the keep left sign.

The third and forth images might be difficult to classify because there are a number of traffic signs in the dataset that are triangular in shape and contains a person in the graphic.

The fifth image might be difficult to classify because of the reflection on the stop sign.


### 2. Model Predictions on Test Imagery

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.

Here are the results of the prediction:

| Image			            |     Prediction	        		             			|
|:---------------------:|:---------------------------------------------:|
| Speed Limit (30km)    |  Speed Limit (30km)   				      					|
| Keep Right    			  | Keep Right  									              	|
| Pedestrians					  | Pedestrians									                	|
| Road Work	      		  |  Road Work							|
| Stop		              | Yield     							                      |




### 3. Details on Test Imagery Predictions
I am a bit surprised that the stop sign was misclassified as yield while the road work sign had some occlusions and was correctly classified.

When looking at the image of the stop sign being fed into the classifier, it could be that the cropping of the image was zoomed in too much and did not have enough pixel buffer compared to the training images for the stop sign. additionally, after the image was equalized, the test image's red section had a lot of variation whereas the training images did not. To avoid this misclassification, it could be that adding noise to the training set or adding an algorithm to eliminate the pixel buffer around the sign may help the classifier.


<center>![alt text][image9]</center>


<center>![alt text][image10]</center>


<center>![alt text][image11]</center>

<center>![alt text][image12]</center>

<center>![alt text][image13]</center>
