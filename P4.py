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
from curve import find_radius
from curve import convert_radius
from curve import get_two_lines

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
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


# create Line instance
l_Line = Line()
r_Line = Line()

# 1. "Camera Calibration" for undistortion (see the calibration.py)
calibration = pl.load(open("camera_cal/calibration.p", "rb"))
objpoints = calibration["objpoints"]
imgpoints = calibration["imgpoints"]
img_size = (1280, 720)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   img_size,
                                                   None, None)
img_files = glob.glob('test_images/*')
img_files = img_files[5:]
for img_file in img_files:
    # write the file name
    print(img_file)

    # *. Read Test Image
    img = cv2.imread(img_file)  # cv2 : BGR, matplotlib : RGB

    """
    2. "Distortion Correction"
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
#    cv2.imwrite('test_preprocess/straight2_undistort.jpg', undist)
#    plt.imshow(undist[...,::-1], cmap='gray')
#    plt.show()

    """
    3-a. Color Transformation
    """
    colorS_bin = HLS_thresh(undist, color='S',
                            thresh=(150, 255))
    colorR_bin = RGB_thresh(undist, color='R',
                            thresh=(200, 255))
    colorH_bin = HLS_thresh(undist, color='H',
                            thresh=(15, 100))

    """
    3-b. Gradient Transformation
    """
    gradX_bin = abs_sobel_thresh(undist, orient='x',
                                 thresh_min=20, thresh_max=100)
    gradY_bin = abs_sobel_thresh(undist, orient='y',
                                 thresh_min=20, thresh_max=100)
    magXY_bin = mag_thresh(undist, sobel_kernel=3,
                           mag_thresh=(30, 100))
    dir_bin = dir_threshold(undist, sobel_kernel=11,
                            thresh=(0.7, 1.3))

    """
    3-c. Combination
    """
    comb1 = np.zeros_like(dir_bin)
    comb1[((gradX_bin == 1) & (gradY_bin == 1)) |
          ((magXY_bin == 1) & (dir_bin == 1))] = 1
    comb2 = np.zeros_like(dir_bin)
    comb2[((colorS_bin == 1) | (colorR_bin == 1) | (colorH_bin == 1)) &
          (dir_bin == 1)] = 1
    comb3 = np.zeros_like(dir_bin)
    comb3[((colorS_bin == 1) | (colorR_bin == 1)) &
          ((colorH_bin == 1) | (dir_bin == 1))] = 1
    comb4 = np.zeros_like(dir_bin)
    comb4[((colorS_bin == 1) & (colorR_bin == 1))
          | ((colorH_bin == 1) & (magXY_bin == 1))] = 1

    # 3-debug. Plot preprocessing
    f, ((ax1, ax2, ax3, ax4),
        (ax5, ax6, ax7, ax8),
        (ax9, ax10, ax11, ax12),
        (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, figsize=(14, 8))

    f.tight_layout()

    # Original & Color Transformation
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=10)
    ax2.imshow(colorS_bin, cmap='gray')
    ax2.set_title('Color S (saturation)', fontsize=10)
    ax3.imshow(colorR_bin, cmap='gray')
    ax3.set_title('Color R (Red)', fontsize=10)
    ax4.imshow(colorH_bin, cmap='gray')
    ax4.set_title('Color H (Hue)', fontsize=10)

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

    # Plot
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.draw()

    """
    4. "Perspective Transform" (bird-eye view)
    """
    top_down, birdM, birdMinv = corners_unwarp(comb4, 2, 2, mtx, dist)
    ax13.imshow(top_down, cmap='gray')
    ax13.set_title('top_down', fontsize=10)
    plt.draw()

#    # Debug : Test Plot
#    print(type(top_down))
#    plt.imshow(top_down, cmap='gray')
#    # plt.savefig('test_images/top_down_sample.png', dpi=300)
#    # cv2.imwrite('test_images/top_down_sample.jpg', top_down)
#    plt.show()

    """
    5. "Lane Detection"
    """
    # window settings
    window_width = 50
    window_height = 40  # Break image into 9 vertical layers (height 720)
    margin = 80  # slide window width for searching

    # Find the centroids for each window
    window_centroids = find_window_centroids(top_down,
                                             window_width,
                                             window_height,
                                             margin)
    print('debug window centroids : ')
    print(window_centroids)

    # find lane path
    img_lane, img_both = find_lanes(top_down, window_centroids,
                                    window_width, window_height)
    ax14.imshow(img_both)
    ax14.set_title('lane window and line', fontsize=10)
    plt.draw()

    """
    6. "Curvature Detection"
    """
    # find quadratic line
    l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(img_lane)

    # find car location
    x_mid = (((l_x[0] + r_x[0]) / 2) - (img_lane.shape[1] / 2)) * 3.7 / 700

    # find radius
#    l_rad, r_rad = find_radius(l_fit, r_fit, l_y)

    # convert radius to real-world coordinate
    l_rad_real, r_rad_real = convert_radius(l_x, r_x, l_y, r_y)

    # Test Plot
    line_img = get_two_lines(l_x, l_y, r_x, r_y, img_both)

    """
    7. Create an image to draw the lines on undistorted image
    """
    warp_zero = np.zeros_like(comb4).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([l_x, l_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([r_x, r_y])))])
    pts = np.hstack((pts_left, pts_right))
    print(pts_left)
    print(pts_right)
#    print(np.int_([pts]))
#    print(np.int_([pts]).shape)

    # Draw the lane onto the warped blank image
    color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using Minv
    newwarp = cv2.warpPerspective(color_warp, birdMinv,
                                  (undist.shape[1], undist.shape[0]))
    newwarp[:360, :] = 0 # mask some part

    # Combine the result with the original image
    result = cv2.addWeighted(undist[..., ::-1], 1, newwarp, 0.3, 0)

    # Add Data
    rad_txt = "Radius[m] = (" + "{0:.2f}".format(l_rad_real) + ", " \
              + "{0:.2f}".format(r_rad_real) + ")"
    dst_txt = "Difference[m] = " + "{0:.2f}".format(x_mid)
    cv2.putText(result, rad_txt, (50,  80), cv2.FONT_HERSHEY_DUPLEX, 1.5, 1,
                thickness=4)
    cv2.putText(result, dst_txt, (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, 1,
                thickness=4)

    # Plot
    ax15.imshow(result)
    ax15.set_title('Final Result', fontsize=10)
    plt.show()
