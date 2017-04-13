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

[result0]: ./sample_data/result0.jpg "straight"
[result1]: ./sample_data/result1.jpg "curve"
[result2]: ./sample_data/result2.jpg "many edges"
[result3]: ./sample_data/result3.jpg "shadows"

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

In the end, I used a combination of color and gradient thresholds to
generate a binary image. My ideas follows

* comb1 : Yellow line or White line
* comb2 : direction_threshold(Yellow or White or Red)
* comb3 : blur(direction_thres or XY_thres) (detection for very slight edge even in shade) 
* comb4 : blur(direction_thres(XY_thres))

(All results are shown at the end of README.md)


comb1 and comb 2 are meant for detecting lines accurately
comb3 and comb 4 are meant for detecting edges roughly

Assuming comb1 and comb2 have higher priority to use,
first the intersection of comb1 and comb2 are used for detection.
If this detects enough data (probably lane), just use two of them.

If the intersection of comb1 and comb2 does not detect enough data, I
additionaly consider comb3 and comb4 and ensemble 4 of them.  If 2 of
4 combinations agrees, then the pixel is considered as lane. (Code around 200)

This gives more flexibility to use only color combination when the
image is clear and to use both color and edge combination when the
image is shadowed.

Of course, if the lane is labeled, we may be able to optimize the obove
combination logic depending on the image feature using machine learning.


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
| 535, 500      | 100, 0        | 
| 760, 500      | 1180, 0      |
| 1280, 720     | 1180, 720      |
| 0, 720        | 100, 720        |


#### 4. Lane Detection

Using the bird-eye view image, I used window search techniques to find
the most probable lanes. The window size is tweaked to capture more
precise information. Plus, the margin used for searching gets bigger
everytime the window search fails. In this way, we focus on the narrow
searching space when close to detected line.(and vice varsa)

Then I fittd my lane lines with a 2nd order polynomial.

For sanity,

* if the detected centroids is less than 10, expand combination(loosen
  the conditions) and find the lane images again (code around 210)

* drop the detection result if it is outside of the mirgin at two
  check points.(y = 180, 719) (code around 250)

* average filter using 5 past data (code around 280)


#### 5. Calculation of the radius of curvature of the lane and the position of the vehicle from center.

I did this in lines 68 through 97 in my code in `curve.py`

#### 6. Example image of the result

I overlaid the result on the original image.  I implemented this step
in lines 315 through 343 in my code in `P4.py`.  Here is an example of
my result on a test image:

* example0. simple straight line : The bird eye view is parallel.
![alt text][result0]

* example1. simple curve : 
![alt text][result1]

* example2. image with a lot of edge distraction : you may see the final combination is not distracted by other edges. This is because the algorithm which prioritize color mapping higher.
![alt text][result2]

* example3. image with a lot of shadow : Even in the shadow area, it captures very slight edges. When color binary does not perform well, it considers the edge information too. This may cause noisy result, but it successfully capture the correct lane.
![alt text][result3]

---

### Pipeline (video)

#### 1. video result

Here's a link to my video result for
[task1 (project_video.mp4)](./result_1.mp4)

And, here's a link to my video result for
[task2 (challenge_video.mp4)](./result_2.mp4)

---

###Discussion

#### 1. Where to fail my pipeline

* My algorithm may result in noisy performance when color is not
  completely detected and rely on only edge information. As I
  mentioned above, the first priority is color. If color is not
  accurately detected, then the edge information is considered.  This
  combination works fine. When color is not detected at all, the
  algorithm completely rely on edge information, whcih tends to be noisy.

* When high latency is required. As I use the average filter, there is
  a time-delay to respond to the measuremnt. When control is
  considered, this time-delay will be critical, making the system 'unstable'.


#### 2. Possible solution

* If each line is marked as the label, we can take machine learning
  approach.  The feature is the combinations of the binary images
  using color or gradient.  Machine leanring enables to find the best
  weights to each combination.  Maybe, we need to clustering approach
  to categorize the image type (shadowed, too bright, clear) before
  the above machine learning. The model should be wisely chosen
  depending on the situation.