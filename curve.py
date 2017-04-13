import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_average_index(arr):
    val_sum = arr.sum()
    tmp = 0
    for i, val in enumerate(arr):
        tmp += val
        if tmp > val_sum/2.0:
            return i
    print('debug in find_average_index : ' + np.argmax(arr))
    return np.argmax(arr)


def find_quadcoeff_lane(img):

    # convert img to gray scale data
    img_arr_BGR = np.array(img)
    l_img_arr = img_arr_BGR[:, :, 1]  # extract only green
    r_img_arr = img_arr_BGR[:, :, 2]  # extract only green

    lenY = img.shape[0]

    # find left average x for each
    l_plotX = []
    l_plotY = []
    for y in range(lenY):
        if sum(l_img_arr[y, :]) != 0:
            ave_x = find_average_index(l_img_arr[y, :])
            l_plotX.append(ave_x)
            l_plotY.append(y)
        else:
            pass

    # find right average x for each y
    r_plotX = []
    r_plotY = []
    for y in range(lenY):
        if sum(r_img_arr[y, :]) != 0:
            ave_x = find_average_index(r_img_arr[y, :])
            r_plotX.append(ave_x)
            r_plotY.append(y)
        else:
            pass

    # Check the availability
    if (len(l_plotX) == 0) | (len(r_plotX) == 0):
        return [None]*6

    # Fit a second order polynomial to pixel positions in each fake lane line
    plotY = np.array(range(0, 720, 1))

    l_plotX = np.array(l_plotX)
    l_plotY = np.array(l_plotY)
    l_fit = np.polyfit(l_plotY, l_plotX, 2)
    l_fitx = l_fit[0]*plotY*plotY + l_fit[1]*plotY + l_fit[2]

    r_plotX = np.array(r_plotX)
    r_plotY = np.array(r_plotY)
    r_fit = np.polyfit(r_plotY, r_plotX, 2)
    r_fitx = r_fit[0]*plotY*plotY + r_fit[1]*plotY + r_fit[2]

    return [l_fit, l_fitx, plotY, r_fit, r_fitx, plotY]


def find_radius(l_fit, r_fit, y_axis):
    y_eval = np.max(y_axis)  # np.max(l_y) or np.max(r_y)

    left_curverad = ((1 + (2*l_fit[0]*y_eval + l_fit[1])**2)**1.5) \
                    / np.absolute(2*l_fit[0])

    right_curverad = ((1 + (2*r_fit[0]*y_eval + r_fit[1])**2)**1.5) \
                     / np.absolute(2*r_fit[0])

    return [left_curverad, right_curverad]


def convert_radius(l_x, r_x, l_y, r_y):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    l_y_eval = np.max(l_y)  # np.max(l_y) or np.max(r_y)
    r_y_eval = np.max(l_y)  # np.max(l_y) or np.max(r_y)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(l_y*ym_per_pix, l_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(r_y*ym_per_pix, r_x*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*l_y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*r_y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_curverad, right_curverad)


def get_two_lines(l_x, l_y, r_x, r_y, image):
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.imshow(image)
    plt.plot(l_x, l_y, color='red', linewidth=3)
    plt.plot(r_x, r_y, color='red', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images
    plt.draw()


if __name__ == '__main__':

    # Read sample image
    lanes = cv2.imread('test_images/detected_line_sample.jpg')

    # find quadratic line
    l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(lanes)

    # find car location
    x_mid = ((l_x[0] + r_x[0]) / 2) - (lanes.shape[1] / 2)

    # find radius
    l_rad, r_rad = find_radius(l_fit, r_fit, l_y)

    # convert radius to real-world coordinate
    l_rad_real, r_rad_real = convert_radius(l_x, r_x, l_y, r_y)
