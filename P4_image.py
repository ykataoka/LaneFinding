"""
Advanced Lane Detection using computer vision technique
"""
# 0. camera calibration
# 1. distortion correction
# 2. thresholded binary image (color, gradients transforms)
# 3. perspective transform
# 4. lane detection
# 5. curvature & vehicle position detection
# 6. Warp the detected lane boundaries
# 7. Output visual display of the lane boundaries

# importing some useful packages
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pl
import glob
from moviepy.editor import VideoFileClip

# gradient transoformation functions
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
j = 0  # image counter for the movie
prev_result = None
prev_warp = None
skip_cnt = 0

# print complete array for debugging purpose
# np.set_printoptions(threshold=np.inf)


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

    def update_recent_fit(self):
        if len(self.current_fit) > self.N:
            self.current_fit.pop(0)

        # update average fit
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
    global prev_result
    global prev_warp
    global skip_cnt

    try:
        """
        1. "Distortion Correction"
        """
        if (exp_mode == 'movie') | (exp_mode == 'video'):
            img = img[..., ::-1]
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        """
        2-a. Color Transformation
        """
        colorH_bin = HLS_thresh(undist, color='H', thresh=(10, 40))  # yellow
        colorS_bin = HLS_thresh(undist, color='S', thresh=(0, 30))  # white
        colorS_bin2 = HLS_thresh(undist, color='S', thresh=(100, 255))
        colorL_bin = HLS_thresh(undist, color='L', thresh=(150, 255))
        colorR_bin = RGB_thresh(undist, color='R', thresh=(180, 255))

        """
        2-b. Gradient Transformation
        """
        magXY_bin = mag_thresh(undist, sobel_kernel=3, mag_thresh=(5, 100))
        dir_bin = dir_threshold(undist, sobel_kernel=5, thresh=(0.8, 1.2))

        """
        2-c. Combination
        """
        # find color yellow
        colorY_bin = np.zeros_like(colorR_bin)
        colorY_bin[(colorH_bin == 1) & (colorS_bin2 == 1)] = 1

        # refine white line
        colorW_bin = np.zeros_like(colorR_bin)
        colorW_bin[(colorL_bin == 1) &
                   (colorR_bin == 1) &
                   (colorS_bin == 1) &
                   (dir_bin == 1)] = 1

        # comb1 : Color 1
        comb1 = np.zeros_like(colorR_bin)
        comb1[((colorY_bin == 1) | (colorW_bin == 1))] = 1

        # comb2 : Color 2
        comb2 = np.zeros_like(colorR_bin)
        comb2[((colorY_bin == 1) | (colorW_bin == 1) | (colorR_bin == 1))] = 1
        comb2 = dir_threshold(comb2, sobel_kernel=5, thresh=(0.7, 1.3))

        # comb3 : Gradient 1
        comb3 = np.zeros_like(colorR_bin)
        comb3[(dir_bin == 1) & (magXY_bin == 1)] = 1
        comb3 = cv2.blur(comb3, (7, 7))

        # comb4 : Gradient 2
        comb4 = np.zeros_like(colorR_bin)
        comb4 = dir_threshold(magXY_bin, sobel_kernel=5, thresh=(0.7, 1.3))
        comb4 = cv2.blur(comb4, (7, 7))

        # High prioritiy : Color combination
        agree_num1 = 1
        agree_num2 = 2
        combA = comb1 + comb2
        combA[combA < agree_num1] = 0
        combA[combA >= agree_num1] = 1

        # Low prioritiy : All four combination
        combB = comb1 + comb2 + comb3 + comb4
        combB[combB < agree_num2] = 0
        combB[combB >= agree_num2] = 1

        """
        3. "Perspective Transform" (bird-eye view)
        """
        top_downA, birdMA, birdMinvA = corners_unwarp(combA, 2, 2, mtx, dist)
        top_downB, birdMB, birdMinvB = corners_unwarp(combB, 2, 2, mtx, dist)

        """
        4. "Lane Detection" + "Determination of Combination"
        """
        # window settings
        window_w = 30
        window_h = 20
        margin = 15

        # Find visualize the centroids of the lanes
        centroids, l_center, r_center = find_window_centroids(top_downA,
                                                              window_w,
                                                              window_h,
                                                              margin)
        img_lane, img_both, num_l, num_r = find_lanes(top_downA,
                                                      centroids,
                                                      window_w,
                                                      window_h)

        # if # of the centroids is too small, consider more combinations
#        plt.imshow(img_both, cmap='gray')
#        plt.show()

        print(num_l, num_r)
        if (num_l < 5) | (num_r < 5):
            centroids, l_center, r_center = find_window_centroids(top_downB,
                                                                  window_w,
                                                                  window_h,
                                                                  margin)
            img_lane, img_both, num_l, num_r = find_lanes(top_downB,
                                                          centroids,
                                                          window_w,
                                                          window_h)
            birdMinv = birdMinvB
            comb = combB
        else:
            birdMinv = birdMinvA
            comb = combA

#        # debug
#        birdMinv = birdMinvB
#        comb = combB

        """
        5. "Curvature Detection"
        """
        # Compute Quadratic Lines
        l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(img_lane)

        # if fitting did not work as all, let's skip unreliable data
        if l_fit is None:
            print("error in find_quadcoeff_lane")
            raise BreakIt

        # if the fitting line is not inside of the boundary
        margin1, margin2 = 150*(skip_cnt + 1), 50*(skip_cnt + 1)
        if j < L_Line.N:
            pass

        elif skip_cnt > 5:
            L_Line.recent_xfitted = []
            L_Line.current_fit = []
            R_Line.recent_xfitted = []
            R_Line.current_fit = []
            j = 0

        else:
            check_point_y = int(undist.shape[0] * 1. / 4.)
            l_dif = np.abs(L_Line.allx[check_point_y] - l_x[check_point_y])
            r_dif = np.abs(R_Line.allx[check_point_y] - r_x[check_point_y])
            print('margin1 :' + str(margin1) + ' , l_dif, r_dif @(1/4) = ' + str(l_dif) + ', ' + str(r_dif))
            if (l_dif > margin1) | (r_dif > margin1):
                print("out of the margin for new fitting line@1/4")
                skip_cnt += 1
                raise BreakIt

            check_point_y = int(undist.shape[0] * 3.9 / 4.)
            l_dif = np.abs(L_Line.allx[check_point_y] - l_x[check_point_y])
            r_dif = np.abs(R_Line.allx[check_point_y] - r_x[check_point_y])
            print('margin2 :' + str(margin2) + ' ,l_dif, r_dif @(4/4)= ' + str(l_dif) + ', ' + str(r_dif))
            if (l_dif > margin2) | (r_dif > margin2):
                print("out of the margin for new fitting line@4/4")
                skip_cnt += 1
                raise BreakIt

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
        6. Create an image to draw the lines on undistorted image
        """
        # analytics result in the bird-eye view
        overlay = line_and_margin(l_x_fil, r_l_y, r_x_fil, r_l_y,
                                  margin1, 'red', undist)

        # perspective transformation
        newwarp = cv2.warpPerspective(overlay, birdMinv,
                                      (undist.shape[1], undist.shape[0]))

        # masking some part of overlay
        newwarp[:360, :] = 0

        # masking some part
        prev_warp = newwarp

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

#        # Add Data
#        rad_txt = "Radius[m] = (" + \
#                  "{0:.2f}".format(L_Line.radius_of_curvature) + ", " + \
#                  "{0:.2f}".format(R_Line.radius_of_curvature) + ")"
#        dst_txt = "Difference[m] = " + "{0:.2f}".format(L_Line.line_base_pos)
#        cv2.putText(result, rad_txt, (50,  80),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)
#        cv2.putText(result, dst_txt, (50, 150),
#                    cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)

        """
        Plot for debugging
        """
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
            ax13.imshow(comb, cmap='gray')
            ax13.set_title('Final Combination', fontsize=10)

            # TopDown
            ax14.imshow(img_both)
            ax14.set_title('lane window and line', fontsize=10)
            ax15.imshow(overlay)
            ax15.set_title('detected line', fontsize=10)

            # Final result
            ax16.imshow(result[..., ::-1])
            ax16.set_title('Final Result', fontsize=10)

            # Show
            plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
            plt.savefig('sample_data/hoge.jpg', dpi=300)
            plt.show()
            plt.close()

        prev_result = result
        # debug
        # j += 1

        if (exp_mode == 'movie') | (exp_mode == 'video'):
            return result[..., ::-1]
        return result

    except BreakIt:
        if prev_result is None:
            return undist
        else:
            result = cv2.addWeighted(undist[..., ::-1], 1, prev_warp, 0.3, 0)

            rad_txt = "Radius[m] = (" + \
                      "{0:.2f}".format(L_Line.radius_of_curvature) + ", " + \
                      "{0:.2f}".format(R_Line.radius_of_curvature) + ")"
            dst_txt = "x_c diff[m] = " + "{0:.2f}".format(L_Line.line_base_pos)
            cv2.putText(result, rad_txt, (50,  80),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)
            cv2.putText(result, dst_txt, (50, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, 1, thickness=4)
            return result


"""
Choose Experiment Mode
  'video' : output video
  'image' : process test images for debugging
"""
# exp_mode = 'video'
# L_Line = Line(5)
# R_Line = Line(5)

exp_mode = 'image'
L_Line = Line(1)
R_Line = Line(1)

"""
0. "Camera Calibration" for undistortion (see the calibration.py)
"""
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
    img_files = glob.glob('sample_data/*')  # challenge 2
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
    # preprocess movie
    clip1 = VideoFileClip("project_video.mp4")
#    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(lane_preprocess)

    # output
    white_output = 'challenge_out1.mp4'
    white_clip.write_videofile(white_output, audio=False)
