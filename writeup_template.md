##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

# **Vehicle Detection Project**

### This project's target is to detect and track moving vehicles in a video stream, using computer vision techniques to extract the features within every frame and a classifier to identify cars from noncars.

### The Project is defined into these major steps:
#### 1. Extracting features from the provided data set of 64x64 images of cars and noncars `extract_features()`.
#### 2. Feeding the the dataset's extracted features into a Linear Support Vector Machine Classifier `clf`. 
#### 3. Process every image, using a sliding windows technique to identify and label possible windows' clusters defining vehicles; `process_img()`.
#### 4. Process a video stream, by processing every frame individually and tracking each detected car with a high certainity while ignoring false positives; `process_vid()`

## 1. DataSet Feature Extraction: `extract_features()`
The dataset provided is composed of 17,760 images, 8,792 car images and 8,968 noncar images. Each image's dimensions is 64x64, which resembles a window in an image. For every image we generate a feature vector containing the following features concatenated.

### 1. Histogram of Oriented Gradients (HOG) Features: `hog_feature()`
We first perform the Histogram of Oriented Gradients on the 64x64 image, using the `skimage.feature` module we import `hog()` function which performs HOG on a single image channel. We provide the `hog()` function with several parameters:
* The number of orientation bins within which we distribute the gradients: `hog_bins = 9`.
* The number of pixels in each HOG cell, in which we accumulate the computed gradients: `px_pcell=8`.
* The number of HOG cells in each HOG block: `cell_pblock=2`.

*The histogram of oriented gradients technique proves that its extremely efficient in detecting the external outline of a car, as follows:*
<p align="center">
<img align="center" src="./writeup_imgs/HOG_features20.png" alt="alt text">
</p>

### 2. Colors Histogram: `color_hist()`
Second, we compute the histogram of the colors within the image which is well clustered in the **YCrCb** color space for cars and could be separated with a classifier from noncars. We change the image's color space using OpenCV's function `cv2.cvtColor()`. 

Each image is separated into three separate channels, and the colors of each channel is distributed among `color_bins = 32` in a histogram using Numpy's `np.histogram()`. The histograms of each channel are then concatenated together into a vector of 96 features.

*The `YCrCb` color space was chosen following several experiments and visualizing the scatterplot of car and non car images within different colorspaces.*

Moreover, the color histogram of random car and noncar samples were visualized in the **YCrCb** color space; A pattern reveals its self.

<p align="center">
<img align="center" src="./writeup_imgs/car_color_hist.png" alt="alt text">
</p>

Where noncar images, does not show a similar pattern to car data color histograms, specially in the **YCrCb** color space. In contrast, noncars have random different patterns of color histograms.

<p align="center">
<img align="center" src="./writeup_imgs/noncar_color_hist.png" alt="alt text">
</p>


### 3. Spatial Binning: `spatial_bin()`
Third, we want use the car image itself, increasing the number of features in the feature vector. However, including a 64x64 image would create a massive addition to the other features. Therefore, we resize the image to the specified size `spatial_size=(32,32)` to 32x32 pixels using OpenCV's `cv2.resize()`. Moreover, the resized image is unraveled, into a single vector.

*The images when resized from 64x64 to 32x32 approximately doesn't lose any features; as shown below:* 
<p align="center">
<img align="center" src="./writeup_imgs/spatial_bin.png" alt="alt text">
</p>

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

