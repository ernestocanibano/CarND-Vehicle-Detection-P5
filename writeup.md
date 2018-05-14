## Vehicle Detection
### Writeup by Ernesto Cañibano.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/color_spaces.png
[image2]: ./examples/hog_example.png
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_examples.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### All the code and the examples are included in the Jupyter Notebook [P5.ipynb](./P5.ipynb), and code file [lesson_functions.py](./lesson_functions.py).

### Color spaces exploration

The first thing I have done is experiment with different color spaces to find out which is better to distinguish between car and no car images. I have used several images with cars and without cars which are included in the folder [test_color_spaces](./test_color_spaces). The code of this is contained in the second code cell of the file `P5.ipynb`.

After several test, I think that the best perfomance can be achieved with `HVS` or `YCrCb` color spaces. In the following images it is possible compare the result in HSV color space, for three vehicle and three non-vehicle images.

![Color spaces][image1]

### Data exploration

I have used the image dataset provided in the project. The dataset is prepared in the third code of cell of the file `P5.ipynb`. There are 8968 non-vehicle images and 8792 vehicle images.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fifth code cell of the IPython notebook `P5.ipynb`.  

I started by reading in all the `vehicle` and `non-vehicle` images. After thath I tried different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog examples][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I trained several linear SVM using different combination of color spaces and HOG features. Use a larger value than `orient=9` `pixel_per_cell=(8,8)` or `cells_per_block=(2,2)` doesn't have a big effecto over the result. 

I decided to use the three channels of HSV, because the perfomance was better than using only one channel. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with a combination of the following features, using the three HSV channels:
* HOG features with the parameters explained in the previous section.
* Spatial features using `spatial_size=(32,32)`.
* Color histograms features usign `hist_bins=32`.

I divided randomly the dataset, 80% for trainning the SVM and 20% to validate the SVM. I achieved an accuracy of 98%.

The code to train the SVM is included in the sixth code cell in file `P5.ipynb`. I saved the trained SVM object in the file `my_classifier.p`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried with different non-scaled windows sizes and different overlapping values. I didn't achieve good results.

After that, I modified the funcion `slide_window()` which is in file `lesson_funcions.py`, to generate scaled windows. I tried with different overlapping and scale values. In the following image it is showed all the windows generated but with 0 overlapping to see better the windows sizes.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To improve the perfomance, I tried modifiying the values of vertical and horizontal overlapping, and modifiying the initial window size.

Here are some example images:

![alt text][image4]

The code to generate these images is in the ninth cell in file `P5.ipynb`.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to the test video](./test_video_output.mp4) and another one to [the project video](./project_video_output.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The functions used here are all included in the file `lesson_functions.py`. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Sometimes the pipeline fails, not detecting a car instantly. This is due to the threshold to remove false positives.
2. I tried usin a non linear SVM , with parameters optimized using `GridSearchCV()` the result is better, but trainning time is huge. This classifer is saved in the file `my_classifier_gridsearch.py`.