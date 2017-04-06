import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def RGB_thresh(img, color='R', thresh=(200, 255)):
    """
    @desc : Define a function that extracts selected color channel
    @param : img - [BGR]
    """
    if color == 'B':
        target = img[:, :, 0]
    elif color == 'G':
        target = img[:, :, 1]
    elif color == 'R':
        target = img[:, :, 2]

    binary = np.zeros_like(target)
    binary[(target > thresh[0]) & (target <= thresh[1])] = 1

    return binary


def HLS_thresh(img, color='S', thresh=(90, 255)):
    """
    @desc : Define a function that extracts selected color channel
    @param : img - [BGR] 
    @thresh : lower_bound, upper_bound e.g., H : (15, 100)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if color == 'H':
        target = hls[:, :, 0]
    elif color == 'L':
        target = hls[:, :, 1]
    elif color == 'S':
        target = hls[:, :, 2]

    binary = np.zeros_like(target)
    binary[(target >= thresh[0]) & (target <= thresh[1])] = 1

    return binary


def HSV_thresh(img, color='S', thresh=(90, 255)):
    """
    @desc : Define a function that extracts selected color channel
    @param : img - [BGR] 
    @thresh : lower_bound, upper_bound e.g., H : (15, 100)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == 'H':
        target = hls[:, :, 0]  # [0,179]
    elif color == 'S':
        target = hls[:, :, 1]  # [0,255]
    elif color == 'V':
        target = hls[:, :, 2]  # [0,255]

    binary = np.zeros_like(target)
    binary[(target > thresh[0]) & (target <= thresh[1])] = 1

    return binary



if __name__ == '__main__':

    # Read in an image and grayscale it
    image = mpimg.imread('test_images/straight_lines1.jpg')

    # Run the function
    grad_binary = abs_sobel_thresh(image, orient='x',
                                   thresh_min=20,
                                   thresh_max=100)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
