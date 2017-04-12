# Lane Finding using Computer Vision by Yasuyuki Kataoka

## The goals / steps of this project

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

[//]: # (Image References)

[image_org]: ./examples/straight_lines1.jpg "original"
[image_undist]: ./examples/straight_lines1_undistort.jpg "undistortion"
[example_calib_org]:  ./examples/calibration1.jpg "calibration"
[example_calib_undist]: ./examples/calibration1_undistort.jpg "calibration_undistort"
[image_binary]: ./examples/threshold_binary.png "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Code
* P4.py - main code
* calibration.py - dump calibration ddata for undistortion
* color.py - RGB and HLS analytics
* curve.py - find radius and quadratic representation for lane curve
* sobel.py - edge detection using derivative filter(sobel)
* warp.py - convert image coordinate and bird-eye coordinate
* window_serach.py - find lane based on bird-eye image


---
### Camera Calibration

#### How it works

I start by preparing "object points", which will be the (x, y, z)
coordinates of the chessboard corners in the world. Here I am assuming
the chessboard is fixed on the (x, y) plane at z=0, such that the
object points are the same for each calibration image.  Thus, `objp`
is just a replicated array of coordinates, and `objpoints` will be
appended with a copy of it every time I successfully detect all
chessboard corners in a test image.  `imgpoints` will be appended with
the (x, y) pixel position of each of the corners in the image plane
with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the
camera calibration and distortion coefficients using the
`cv2.calibrateCamera()` function.  The calibration parameter computed
here is saved by pickle. In the main code(P4.py), simply load pickle
file to get the parameter. I applied this distortion correction to the
test image using the `cv2.undistort()` function and obtained this
result:

The code for this step is contained in calibration.py.

Original Image              |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][example_calib_org]  |  ![alt text][example_calib_undist]


### Pipeline (single images)

#### 1. Undistortion I apply the distortion correction obtained the
calibration process.  One of the test images like this one. In this
example, the difference seems like very slight. But distortion affects
the images around corner.

Original Image              |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image_org]  |  ![alt text][image_undist]


#### 2. Thresholded binary image (Color transforms, Gradients and combnations of them)

One of the test images like this one.  The color thresholding steps
 are described in `color.py`.  The gradient thresholding steps are
 described in `sobel.py`.  The combination logics are described in
 `P4.py` line around 135.

![alt text][image_binary]

In the end, I used a combination of color and gradient thresholds to
generate a binary image. To give the more flexibility to determine the
best combination, I ensembled the 4 different combinations; consider 1
if more than 2 of combnation agrees at each pixels.


#### 3. Perspective Transform

The perspective transform function is desribed in `warp.py`.

I chose the hardcode the source and destination points by looking at
straight example image with the effort to cover the src image more. In
this way, even though the line is close right/left edge, the
transoformation can work. Then, I tweaked top 2 corner points a bit so
that two lines will be parallel in bird-eye view.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 565, 460      | 150, 30        | 
| 720, 450      | 1130, 30      |
| 1280, 720     | 1130, 690      |
| 0, 720        | 150, 690        |


![alt text][image4]

#### 4. Lane Detection

Using the bird-eye view image, I used window search techniques to find
the most probable lanes. The window size is tweaked to capture more
precise information.

Then I fittd my lane lines with a 2nd order polynomial kinda like
this:

For sanity,

* drop the detection if it is outside of the mirgin at either of two
  check points.(y = 180, 719)

* average filter using 5 past data


![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  





## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!
