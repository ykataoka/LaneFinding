# 1. Compute the camera calibration matrix and distortion coefficients
#    given chessboard images.
# 2. Apply a distortion correction to raw images.
# 3. Use color transforms, gradients, etc., to create a thresholded binary image.
# 4. Apply a perspective transform to rectify binary image ("birds-eye view").
# 5. Detect lane pixels and fit to find the lane boundary.
# 6. Determine the curvature of the lane and vehicle position with respect to center.
# 7. Warp the detected lane boundaries back onto the original image.
# 8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# importing some useful packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pl
import glob
from moviepy.editor import VideoFileClip

# gradient transoformation functions
from sobel import abs_sobel_thresh
from sobel import mag_thresh
from sobel import dir_threshold

# color transformation functions
from color import RGB_thresh
from color import HLS_thresh

# perspective tranformation functions
from warp import corners_unwarp

# lane detection
from window_search import find_lanes
from window_search import find_window_centroids
from curve import find_quadcoeff_lane
from curve import convert_radius
# from curve import get_two_lines

# global variables
j = 0


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, num):
        # store past n samples
        self.N = num

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients of the last n fits (for the most recent fit)
#        self.current_fit = [np.array([False])]
        self.current_fit = []

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # radius of curvature of the line in some units
        self.radius_of_curvature = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

    def update_recent_x(self, x_new):
        if len(self.recent_xfitted) > self.N:
            self.recent_xfitted.pop(0)

        # update average x
        self.bestx = np.average(self.recent_xfitted)

    def update_recent_fit(self, fit):
        if len(self.current_fit) > self.N:
            self.current_fit.pop(0)

        # update average x
        self.best_fit = np.sum(self.current_fit, axis=0) \
                        / len(self.current_fit)

# Define the break function
class BreakIt(Exception):
    pass


def lane_preprocess(img):
    """
    @desc : full preprocess pipe line for line detection
    @param : original img (3 channel)
    @return : Lane overlayed img
    """
    global j
    try:
        """
        2. "Distortion Correction"
        """
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        """
        3-a. Color Transformation
        """
        colorH_bin = HLS_thresh(undist, color='H',
                                thresh=(10, 40))
        colorS_bin = HLS_thresh(undist, color='S',
                                thresh=(50, 255))
        colorL_bin = HLS_thresh(undist, color='L',
                                thresh=(200, 255))
        colorR_bin = RGB_thresh(undist, color='R',
                                thresh=(180, 255))
        colorY_bin = np.zeros_like(colorR_bin)
        colorY_bin[((colorS_bin == 1) & (colorH_bin == 1))] = 1
        colorY_bin[((colorS_bin == 1) & (colorH_bin == 1))] = 1

        """
        3-b. Gradient Transformation
        """
        gradX_bin = abs_sobel_thresh(undist, orient='x',
                                     thresh_min=20, thresh_max=100)
        gradY_bin = abs_sobel_thresh(undist, orient='y',
                                     thresh_min=20, thresh_max=100)
        magXY_bin = mag_thresh(undist, sobel_kernel=3,
                               mag_thresh=(15, 200))
        dir_bin = dir_threshold(undist, sobel_kernel=5,
                                thresh=(0.8, 1.2))
        
        """
        3-c. Combination
        """
        comb1 = np.zeros_like(magXY_bin)
        comb1[((colorY_bin == 1) | (colorL_bin == 1))] = 1
        comb2 = np.zeros_like(dir_bin)
        comb2[(dir_bin == 1) & (magXY_bin == 1)] = 1
#        comb2[((colorY_bin == 1) | (colorL_bin == 1) | (colorR_bin == 1)) &
#              (dir_XY_bin)] = 1
#        comb4 = np.zeros_like(magXY_bin)
#        comb4[((colorS_bin == 1) & (colorR_bin == 1))
#              | ((colorH_bin == 1) & (magXY_bin == 1))] = 1
        comb3 = np.zeros_like(magXY_bin)
        comb3[((colorY_bin == 1) | (colorL_bin == 1) | (colorR_bin == 1))] = 1
        comb4 = np.zeros_like(magXY_bin)
        comb4[((colorS_bin == 1) | (colorR_bin == 1))
              | ((colorH_bin == 1) & (magXY_bin == 1))] = 1

        """
        4. "Perspective Transform" (bird-eye view)
        """
        # decide which combination to be used
        combs = [comb1, comb2, comb3, comb4]
        diffs = []
        white_v = 5000000  # value of the white range (set by experiment)
        for comb in combs:
            top_down, birdM, birdMinv = corners_unwarp(comb, 2, 2, mtx, dist)
            diffs.append(abs(np.sum(top_down) - white_v))

        print('chosen combination')
        print(np.argmin(np.array(diffs)) + 1)
        comb = combs[np.argmin(np.array(diffs))]

        # perspective transformation to the chosen one
        top_down, birdM, birdMinv = corners_unwarp(comb, 2, 2, mtx, dist)

        """
        5. "Lane Detection"
        """
        # window settings
        window_width = 50
        window_height = 40  # Break image into 9 vertical layers (height 720)
        margin = 80  # slide window width for searching

        # Find the centroids for each window
        window_centroids, l_center, r_center = find_window_centroids(top_down,
                                                                     window_width,
                                                                     window_height,
                                                                     margin)

        # find lane path
        img_lane, img_both = find_lanes(top_down, window_centroids,
                                        window_width, window_height)

        """
        6. "Curvature Detection"
        """
        # Compute Quadratic Lines
        l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(img_lane)

        # if fitting did not work, let's skip unreliable data
        if l_fit is None:
            print("error in find_quadcoeff_lane")
            cv2.imwrite('test_preprocess/bug_undistort'+str(j)+'.jpg',
                        undist[..., ::-1])
            j = j + 1
            raise BreakIt

        # if the fitting line is not inside the boundary
        elif :
            L_Line.

        # Update data if the fitting goes well
        else:
            L_Line.recent_xfitted.append(l_x[-1])
            L_Line.allx = l_x
            L_Line.ally = l_y
            L_Line.current_fit.append(l_fit)
            L_Line.diffs = (L_Line.diffs - l_fit)

            R_Line.allx = r_x
            R_Line.ally = r_y
            R_Line.current_fit.append(r_fit)
            R_Line.diffs = (R_Line.diffs - r_fit)

        # Sanitization : average position (x, 720) using the past N data
        L_Line.update_recent_x(l_x[-1])  # update L_Line.bestx
        R_Line.update_recent_x(r_x[-1])  # update R_Line.bestx

        # Sanitization : average fit parameter using the past N data
        L_Line.update_recent_fit(L_Line.current_fit)  # update L_Line.best_fit
        R_Line.update_recent_fit(R_Line.current_fit)  # update R_Line.best_fit

        # find filtered best parameter
        r_l_y = np.array(range(0, 720, 1))
        l_fit_fil = L_Line.best_fit
        r_fit_fil = R_Line.best_fit
        l_x_fil = l_fit_fil[0]*r_l_y*r_l_y + l_fit_fil[1]*r_l_y + l_fit_fil[2]
        r_x_fil = r_fit_fil[0]*r_l_y*r_l_y + r_fit_fil[1]*r_l_y + r_fit_fil[2]

        # find car location
        offset = img_lane.shape[1]/2
        x_mid = (((L_Line.bestx + R_Line.bestx)/2.) - offset) * 3.7 / 700
        L_Line.line_base_pos = x_mid
        R_Line.line_base_pos = x_mid

        # convert radius to real-world coordinate (? need filter?)
        l_rad_real, r_rad_real = convert_radius(l_x_fil, r_x_fil,
                                                r_l_y, r_l_y)
        L_Line.radius_of_curvature = l_rad_real
        R_Line.radius_of_curvature = r_rad_real

        """
        7. Create an image to draw the lines on undistorted image
        """
        warp_zero = np.zeros_like(comb).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([l_x_fil,
                                                     r_l_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x_fil,
                                                                r_l_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using Minv
        newwarp = cv2.warpPerspective(color_warp, birdMinv,
                                      (undist.shape[1], undist.shape[0]))
        newwarp[:360, :] = 0  # mask some part

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Add Data
        rad_txt = "Radius[m] = (" + "{0:.2f}".format(l_rad_real) + ", " \
                  + "{0:.2f}".format(r_rad_real) + ")"
        dst_txt = "Difference[m] = " + "{0:.2f}".format(x_mid)
        cv2.putText(result, rad_txt, (50,  80),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)
        cv2.putText(result, dst_txt, (50, 150),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)

        # Plot
        if exp_mode == 'image':
            f, ((ax1, ax2, ax3, ax4),
                (ax5, ax6, ax7, ax8),
                (ax9, ax10, ax11, ax12),
                (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4,
                                                         figsize=(14, 8))
            f.tight_layout()

            # Original & Color Transformation
            ax1.imshow(img[..., ::-1])
            ax1.set_title('Original Image', fontsize=10)
            ax2.imshow(colorY_bin, cmap='gray')
            ax2.set_title('Color Y (yellow)', fontsize=10)
            #        ax2.imshow(colorS_bin, cmap='gray')
            #        ax2.set_title('Color S (saturation)', fontsize=10)
            ax3.imshow(colorR_bin, cmap='gray')
            ax3.set_title('Color R (Red)', fontsize=10)
            #        ax4.imshow(colorH_bin, cmap='gray')
            #        ax4.set_title('Color H (Hue)', fontsize=10)
            ax4.imshow(colorL_bin, cmap='gray')
            ax4.set_title('Color White (Lightness is big)', fontsize=10)

            # Gradient Transformation
            ax5.imshow(gradY_bin, cmap='gray')
            ax5.set_title('Gradient x binary', fontsize=10)
            ax6.imshow(gradX_bin, cmap='gray')
            ax6.set_title('Gradient y binary', fontsize=10)
            ax7.imshow(magXY_bin, cmap='gray')
            ax7.set_title('Gradient x&y magnitude', fontsize=10)
            ax8.imshow(dir_bin, cmap='gray')
            ax8.set_title('Gradient direction arctan(y/x)', fontsize=10)

            # Combination
            ax9.imshow(comb1, cmap='gray')
            ax9.set_title('combination 1', fontsize=10)
            ax10.imshow(comb2, cmap='gray')
            ax10.set_title('combination 2', fontsize=10)
            ax11.imshow(comb3, cmap='gray')
            ax11.set_title('combination 3', fontsize=10)
            ax12.imshow(comb4, cmap='gray')
            ax12.set_title('combination 4', fontsize=10)

            # TopDown
            ax13.imshow(top_down, cmap='gray')
            ax13.set_title('top_down', fontsize=10)
            ax14.imshow(img_both)
            ax14.set_title('lane window and line', fontsize=10)

            # Final result
            ax15.imshow(result[..., ::-1])
            ax15.set_title('Final Result', fontsize=10)

            # Show
            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
            plt.show()
#            plt.clf()
            plt.close()

        return result

    except BreakIt:
        return img


# Experiment Mode 'video' : output video, 'image' : proces test_images/*
exp_mode = 'video'

# create Line instance
L_Line = Line(10)
R_Line = Line(10)

# 1. "Camera Calibration" for undistortion (see the calibration.py)
calibration = pl.load(open("camera_cal/calibration.p", "rb"))
objpoints = calibration["objpoints"]
imgpoints = calibration["imgpoints"]
img_size = (1280, 720)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   img_size,
                                                   None, None)

"""
test on image data
"""
if exp_mode == 'image':

    # read image directory depending on challenge
    # img_files = glob.glob('test_images/*')      # challenge 1
    img_files = glob.glob('test_challenge2/*')  # challenge 2
    for img_file in img_files:
        # write the file name
        print(img_file)

        # init the line class to avoid filter
        L_Line = Line(5)
        R_Line = Line(5)

        # Read Test Image
        img = cv2.imread(img_file)

        # Preprocess
        out = lane_preprocess(img)

"""
output to video
"""
if exp_mode == ('video' | 'movie'):
    white_output = 'test2.mp4'
    # clip1 = VideoFileClip("project_video2.mp4")
    clip1 = VideoFileClip("challenge_video2.mp4")
    # clip1 = VideoFileClip("harder_challenge_video.mp4")
    white_clip = clip1.fl_image(lane_preprocess)
    white_clip.write_videofile(white_output, audio=False)
