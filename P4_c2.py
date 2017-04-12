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
from window_search import line_and_margin
from curve import find_quadcoeff_lane
from curve import convert_radius
# from curve import get_two_lines

# global variables
j = 0
prev_result = None
prev_warp = None
skip_cnt = 0

# print everything
#np.set_printoptions(threshold=np.inf)

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
        self.allx = [0] * 720

        # y values for detected line pixels
        self.ally = [0] * 720

    def update_recent_x(self):
        if len(self.recent_xfitted) > self.N:
            self.recent_xfitted.pop(0)

        # update average x
        self.bestx = np.average(self.recent_xfitted)
#        print(self.recent_xfitted)

    def update_recent_fit(self):
        if len(self.current_fit) > self.N:
            self.current_fit.pop(0)

        # update average fit
        self.best_fit = np.sum(self.current_fit, axis=0) \
                        / len(self.current_fit)
#        print(self.current_fit)


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
    global prev_result
    global prev_warp
    global skip_cnt
    try:
        """
        2. "Distortion Correction"
        """
        if (exp_mode == 'movie') | (exp_mode == 'video'):
            img = img[..., ::-1]
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        """
        3-a. Color Transformation
        """
        colorH_bin = HLS_thresh(undist, color='H', thresh=(10, 40))  # yellow
        colorS_bin = HLS_thresh(undist, color='S', thresh=(0, 30))  # white
        colorS_bin2 = HLS_thresh(undist, color='S', thresh=(100, 255))
        colorL_bin = HLS_thresh(undist, color='L', thresh=(150, 255))
        colorL_bin2 = HLS_thresh(undist, color='L', thresh=(1, 255))
        colorR_bin = RGB_thresh(undist, color='R', thresh=(180, 255))


        """
        3-b. Gradient Transformation
        """
#        gradX_bin = abs_sobel_thresh(undist, orient='x',
#                                     thresh_min=20, thresh_max=100)
#        gradY_bin = abs_sobel_thresh(undist, orient='y',
#                                     thresh_min=20, thresh_max=100)
        magXY_bin = mag_thresh(undist, sobel_kernel=3, mag_thresh=(5, 100))
        dir_bin = dir_threshold(undist, sobel_kernel=5, thresh=(0.8, 1.2))

        """
        3-c. Combination
        """
        # find color yellow
        colorY_bin = np.zeros_like(colorR_bin)
        colorY_bin[(colorH_bin == 1) & (colorS_bin2 == 1)] = 1
#        colorY_bin = dir_threshold(colorY_bin, sobel_kernel=5,
#                                   thresh=(0.8, 1.2))

        # refine white line
        colorW_bin = np.zeros_like(colorR_bin)
        colorW_bin[(colorL_bin == 1) &
                   (colorR_bin == 1) &
                   (colorS_bin == 1) &
                   (dir_bin == 1)] = 1

        # comb1 : color is yellow or white
        comb1 = np.zeros_like(colorR_bin)
        comb1[((colorY_bin == 1) | (colorW_bin == 1))] = 1

        # comb2 : Color(Y | W | R) + direction filter
        comb2 = np.zeros_like(colorR_bin)
        comb2[((colorY_bin == 1) | (colorW_bin == 1) | (colorR_bin == 1))] = 1
        comb2 = dir_threshold(comb2, sobel_kernel=5, thresh=(0.7, 1.3))

        # comb3 : gradient based detection
        comb3 = np.zeros_like(colorR_bin)
        comb3[(dir_bin == 1) & (magXY_bin == 1)] = 1
        comb3 = cv2.blur(comb3, (7, 7))

        # comb4 : edge detection
        comb4 = np.zeros_like(colorR_bin)
        comb4 = dir_threshold(magXY_bin, sobel_kernel=5, thresh=(0.7, 1.3))
        comb4 = cv2.blur(comb4, (7, 7))

        # decide which combination to be used

        # a. threshold base
#        combs = [comb1, comb2, comb3, comb4]
#        diffs = []
#        white_v = 5000000  # value of the white range (set by experiment)
#        for comb in combs:
#            top_down, birdM, birdMinv = corners_unwarp(comb, 2, 2, mtx, dist)
#            diffs.append(abs(np.sum(top_down) - white_v))
#        print('chosen combination')
#        print(np.argmin(np.array(diffs)) + 1)
#        comb = combs[np.argmin(np.array(diffs))]

        # b. majority base
#        agree_num = 2
#        comb = np.zeros_like(comb1)
#        comb = comb1 + comb2 + comb3 + comb4
#        comb[comb < agree_num] = 0
#        comb[comb >= agree_num] = 1

        # c. priority base
        agree_num = 2
        combA = comb1 + comb2
        combA[combA < agree_num] = 0
        combA[combA >= agree_num] = 1

        combB = comb1 + comb2 + comb3 + comb4
        combB[combB < agree_num] = 0
        combB[combB >= agree_num] = 1

#        """
#        4. "Perspective Transform" (bird-eye view)
#        """
#        # perspective transformation to the chosen one
#        top_down, birdM, birdMinv = corners_unwarp(comb, 2, 2, mtx, dist)
#
#        """
#        5. "Lane Detection"
#        """
#        # window settings
#        window_width = 30
#        window_height = 20  # Break image into 9 vertical layers (height 720)
#        margin = 15  # slide window width for searching
#
#        # Find the centroids for each window
#        window_centroids, l_center, r_center = find_window_centroids(top_down,
#                                                                     window_width,
#                                                                     window_height,
#                                                                     margin)

        """
        4. "Perspective Transform" (bird-eye view)
        """
        # perspective transformation to the chosen one
        top_downA, birdMA, birdMinvA = corners_unwarp(combA, 2, 2, mtx, dist)
        top_downB, birdMB, birdMinvB = corners_unwarp(combB, 2, 2, mtx, dist)

        """
        5. "Lane Detection"
        """
        # window settings
        window_w = 30
        window_h = 20  # Break image into 9 vertical layers (height 720)
        margin = 15  # slide window w for searching

        # Find the centroids for each window
#        plt.imshow(top_downA)
#        plt.show()
        centroids, l_center, r_center = find_window_centroids(top_downA,
                                                              window_w,
                                                              window_h,
                                                              margin)
        img_lane, img_both, num_l, num_r = find_lanes(top_downA, centroids, window_w, window_h)

        C = np.array(centroids)
        if (num_l < 10) | (num_r < 10):
            centroids, l_center, r_center = find_window_centroids(top_downB,
                                                                  window_w,
                                                                  window_h,
                                                                  margin)
            img_lane, img_both, num_l, num_r = find_lanes(top_downB, centroids, window_w, window_h)
            birdMinv = birdMinvB
            comb = combB
        else:
            birdMinv = birdMinvA
            comb = combA

        # debug
        birdMinv = birdMinvB
        comb = combB

        # find lane path
#        img_lane, img_both = find_lanes(top_down, centroids, window_w, window_h)

        """
        6. "Curvature Detection"
        """
        # Compute Quadratic Lines
        l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(img_lane)

        # if fitting did not work as all, let's skip unreliable data
        if l_fit is None:
            print("error in find_quadcoeff_lane")
            raise BreakIt

        # if the fitting line is not inside of the boundary
        margin1, margin2 = 100*skip_cnt, 50*skip_cnt
        if j < L_Line.N:
            pass

#        elif skip_cnt > 8:
#            L_Line.recent_xfitted = []
#            L_Line.current_fit = []
#            R_Line.recent_xfitted = []
#            R_Line.current_fit = []
#            j = 0
#
#        else:
#            check_point_y = int(undist.shape[0] * 1. / 4.)
#            l_dif = np.abs(L_Line.allx[check_point_y] - l_x[check_point_y])
#            r_dif = np.abs(R_Line.allx[check_point_y] - r_x[check_point_y])
#            print('l_dif, r_dif @(1/4) = ' + str(l_dif) + ', ' + str(r_dif))
#            if (l_dif > margin1) | (r_dif > margin1):
#                print("out of the margin for new fitting line")
#                skip_cnt += 1
#                raise BreakIt
#
#            check_point_y = int(undist.shape[0] * 3.9 / 4.)
#            l_dif = np.abs(L_Line.allx[check_point_y] - l_x[check_point_y])
#            r_dif = np.abs(R_Line.allx[check_point_y] - r_x[check_point_y])
#            print('l_dif, r_dif @(4/4)= ' + str(l_dif) + ', ' + str(r_dif))
#            if (l_dif > margin2) | (r_dif > margin2):
#                print("out of the margin for new fitting line")
#                skip_cnt += 1
#                raise BreakIt

        # reset skip_cnt, if successfully detected
        skip_cnt = 0

        # Update data if the fitting goes well
        L_Line.recent_xfitted.append(l_x[-1])
        L_Line.current_fit.append(l_fit)
        L_Line.diffs = (L_Line.diffs - l_fit)

        R_Line.recent_xfitted.append(r_x[-1])
        R_Line.current_fit.append(r_fit)
        R_Line.diffs = (R_Line.diffs - r_fit)

        # filter : average position (x, 720) using the past N data
        L_Line.update_recent_x()  # update L_Line.bestx
        R_Line.update_recent_x()  # update R_Line.bestx

        # filter : average fit parameter using the past N data
        L_Line.update_recent_fit()  # update L_Line.best_fit
        R_Line.update_recent_fit()  # update R_Line.best_fit

        # find filtered best parameter
        r_l_y = np.array(range(0, 720, 1))
        l_fit_fil = L_Line.best_fit
        r_fit_fil = R_Line.best_fit
        l_x_fil = l_fit_fil[0]*r_l_y*r_l_y + l_fit_fil[1]*r_l_y + l_fit_fil[2]
        r_x_fil = r_fit_fil[0]*r_l_y*r_l_y + r_fit_fil[1]*r_l_y + r_fit_fil[2]

        L_Line.allx = l_x_fil
        R_Line.allx = r_x_fil
        L_Line.ally = r_l_y
        R_Line.ally = r_l_y

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
        # analytics result in the bird-eye view
        overlay = line_and_margin(l_x_fil, r_l_y, r_x_fil, r_l_y,
                                  margin1, 'red', undist)

        # perspective transformation
        newwarp = cv2.warpPerspective(overlay, birdMinv,
                                      (undist.shape[1], undist.shape[0]))

#        warp_zero = np.zeros_like(comb).astype(np.uint8)
#        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#        # Recast the x and y points into usable format for cv2.fillPoly()
#        pts_left = np.array([np.transpose(np.vstack([l_x_fil,
#                                                     r_l_y]))])
#        pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x_fil,
#                                                                r_l_y])))])
#        pts = np.hstack((pts_left, pts_right))
#
#        # Draw the lane onto the warped blank image
#        color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
#
#        # Warp the blank back to original image space using Minv
#        newwarp = cv2.warpPerspective(color_warp, birdMinv,
#                                      (undist.shape[1], undist.shape[0]))

        # masking some part
        newwarp[:360, :] = 0  # mask some part

        # masking some part
        prev_warp = newwarp

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        # Add Data
#        rad_txt = "Radius[m] = (" + "{0:.2f}".format(l_rad_real) + ", " \
#                  + "{0:.2f}".format(r_rad_real) + ")"
#        dst_txt = "Difference[m] = " + "{0:.2f}".format(x_mid)
#        cv2.putText(result, rad_txt, (50,  80),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)
#        cv2.putText(result, dst_txt, (50, 150),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)

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
            ax3.imshow(colorW_bin, cmap='gray')
            ax3.set_title('Color White', fontsize=10)
            ax4.imshow(colorR_bin, cmap='gray')
            ax4.set_title('Color R (Red)', fontsize=10)

            # Gradient Transformation
            ax5.imshow(colorS_bin, cmap='gray')
            ax5.set_title('Saturation', fontsize=10)
            ax6.imshow(colorL_bin, cmap='gray')
            ax6.set_title('Lightness', fontsize=10)
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
            ax13.imshow(comb, cmap='gray')
            ax13.set_title('Final Combination', fontsize=10)
            ax14.imshow(img_both)
            ax14.set_title('lane window and line', fontsize=10)
            ax15.imshow(overlay)
            ax15.set_title('detected line', fontsize=10)

            # Final result
            ax16.imshow(result[..., ::-1])
            ax16.set_title('Final Result', fontsize=10)
            cv2.imwrite('test_challenge2/test' + str(j) + '.jpg', result)

            # Show
            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
            plt.show()
            plt.close()

        prev_result = result
        j += 1

        if (exp_mode == 'movie') | (exp_mode == 'video'):
            return result[..., ::-1]
        return result

    except BreakIt:
        if prev_result is None:
            return undist
        else:
            result = cv2.addWeighted(undist, 1, prev_warp, 0.3, 0)
            return result


# Experiment Mode 'video' : output video, 'image' : proces test_images/*
# exp_mode = 'image'
# L_Line = Line(1)
# R_Line = Line(1)

exp_mode = 'video'
L_Line = Line(5)
R_Line = Line(5)

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
    img_files = glob.glob('test_challenge3/*')  # challenge 2
    for img_file in img_files:
        # write the file name
        print(img_file)

        # init the line class to avoid filter
        L_Line = Line(1)
        R_Line = Line(1)

        # Read Test Image
        img = cv2.imread(img_file)

        # Preprocess and show result
        out = lane_preprocess(img)

"""
output to video
"""
if (exp_mode == 'video') | (exp_mode == 'movie'):
    white_output = 'test.mp4'
    # clip1 = VideoFileClip("project_video2.mp4")
#    clip1 = VideoFileClip("challenge_video2.mp4")
    # clip1 = VideoFileClip("challenge_video3.mp4")
    clip1 = VideoFileClip("challenge_video.mp4")

    # data from clip1
#    A1 = clip1.get_frame(0) #RGB!!

    # data from the viode dump
#    A2 = cv2.imread('test_challenge3/0.jpg', )  # BGR
#    plt.imshow(A1)
#    plt.show()
#    plt.imshow(A2[..., ::-1])
#    plt.show()

#    input()
    # clip1 = VideoFileClip("harder_challenge_video.mp4")
    white_clip = clip1.fl_image(lane_preprocess)
    white_clip.write_videofile(white_output, audio=False)
