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
    img_arr = img_arr_BGR.sum(axis=2)  # because R:0, G:0-255, B:0
#    plt.imshow(img_arr, cmap='gray')
#    plt.show()

    # bug? why over 255??
    img_arr[img_arr > 255] = 255
    lenX = img_arr.shape[1]
    lenY = img_arr.shape[0]
    thres = int(lenX / 2)

    # find left average x for each
    l_arr = img_arr[:, :thres]
    l_plotX = []
    l_plotY = []
    for y in range(lenY):
        if(sum(l_arr[y, :]) != 0):
            ave_x = find_average_index(l_arr[y, :])
            l_plotX.append(ave_x)
            l_plotY.append(y)

    # find right average x for each y
    r_arr = img_arr[:, thres:]
    r_plotX = []
    r_plotY = []
    for y in range(lenY):
        if(sum(r_arr[y, :]) != 0):
            ave_x = find_average_index(r_arr[y, :])
            r_plotX.append(ave_x)
            r_plotY.append(y)

    # Fit a second order polynomial to pixel positions in each fake lane line
    l_plotX = np.array(l_plotX)
    l_plotY = np.array(l_plotY)
    l_fit = np.polyfit(l_plotY, l_plotX, 2)
    l_fitx = l_fit[0]*l_plotY*l_plotY + l_fit[1]*l_plotY + l_fit[2]

    r_plotX = np.array(r_plotX)
    r_plotY = np.array(r_plotY)
    r_fit = np.polyfit(r_plotY, r_plotX, 2)
    r_fitx = r_fit[0]*r_plotY*r_plotY + r_fit[1]*r_plotY + r_fit[2] + thres

    return [l_fit, l_fitx, l_plotY, r_fit, r_fitx, r_plotY]


def find_radius(l_fit, r_fit, y_axis):
    y_eval = np.max(y_axis)  # np.max(l_y) or np.max(r_y)

    left_curverad = ((1 + (2*l_fit[0]*y_eval + l_fit[1])**2)**1.5) \
                    / np.absolute(2*l_fit[0])

    right_curverad = ((1 + (2*r_fit[0]*y_eval + r_fit[1])**2)**1.5) \
                     / np.absolute(2*r_fit[0])

    return [left_curverad, right_curverad]


def convert_radius(l_x, r_x, y_axis):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(y_axis)  # np.max(l_y) or np.max(r_y)

    # Fit new polynomials to x,y in world space
    print(len(l_x))
    print(len(r_x))
    print(len(y_axis))
    left_fit_cr = np.polyfit(y_axis*ym_per_pix, l_x*xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_axis*ym_per_pix, r_x*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_curverad, right_curverad)


def get_two_lines(l_x, l_y, r_x, r_y, image):
    #    mark_size = 3
    #    mixed = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.imshow(image)
    plt.plot(l_x, l_y, color='red', linewidth=3)
    plt.plot(r_x, r_y, color='red', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.show()


if __name__ == '__main__':

    # Read sample image
    lanes = cv2.imread('test_images/detected_line_sample.jpg')

    # find quadratic line
    l_fit, l_x, l_y, r_fit, r_x, r_y = find_quadcoeff_lane(lanes)

    # find car location
    x_mid = ((l_x[0] + r_x[0]) / 2) - (lanes.shape[1] / 2)
    print("current location = ", x_mid)

    # find radius
    l_rad, r_rad = find_radius(l_fit, r_fit, l_y)
    print(l_rad, r_rad)

    # convert radius to real-world coordinate
    l_rad_real, r_rad_real = convert_radius(l_x, r_x, l_y)
    print(l_rad_real, r_rad_real)

#    # Test Plot
#    line_img = get_two_lines(l_x, l_y, r_x, r_y, lanes)
