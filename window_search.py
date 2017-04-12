import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
#np.set_printoptions(threshold=np.inf)


def window_mask(width, height, img_ref, center, level):
    """
    @desc : masking the rarget window
    """
    # bit 1 for target zone
    startX = int(img_ref.shape[0] - (level+1) * height)
    endX = int(img_ref.shape[0] - level*height)
    startY = max(0, int(center - (width/2)))
    endY = min(int(center + (width/2)), img_ref.shape[1])

    # activate the target zone
    output = np.zeros_like(img_ref)
    output[startX:endX, startY:endY] = 1

    return output


def find_window_centroids(image, window_width, window_height, margin):
    """
    @desc : find the centroids at each level
    """
    # Store the (left,right) window centroid positions per level
    window_centroids = []  # [(l_center, r_center), ...]
    out_centroids = []  # [(y, l_center, r_center), ...] can be None

    # Create our window template that we will use for convolutions
    window = np.ones(window_width)

    # find the two starting positions for the left & right lane and 
    # then np.convolve the vertical image slice with the window template

    # apply blur filter
#    plt.imshow(image)
#    plt.show()
#    image = cv2.blur(image, (11, 11))
#    image[image < 150] = 0
#    plt.imshow(image)
#    plt.show()

    # Sum quarter bottom of image to get slice (shape = (Y-720, X-1280))
    upperY = int(2*image.shape[0]/4)  # look at quarter bottom image
    thresX = int(image.shape[1]/2)

    l_sum = np.sum(image[upperY:, :thresX], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2

    r_sum = np.sum(image[upperY:, thresX:], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 \
               + int(image.shape[1]/2)

    # Add what we found for the first layer
#    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    num_level = (int)(image.shape[0]/window_height)
    l_center_init = None
    r_center_init = None
    l_miss_cnt = 1
    r_miss_cnt = 1
    for level in range(0, num_level):

        # convolve the window into the vertical slice of the image
        lowerX = int(image.shape[0]-(level+1)*window_height)
        upperX = int(image.shape[0]-level*window_height)
        image_layer = np.sum(image[lowerX:upperX, :], axis=0)
        conv_signal = np.convolve(window, image_layer)

        l_margin = margin * l_miss_cnt
        r_margin = margin * r_miss_cnt
        
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset
        # because convolution signal reference is at right side of window,
        # not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - l_margin, 0))
        l_max_index = int(min(l_center + offset + l_margin, image.shape[1]))

        # if no line is detected, simply pass the previous data
        if sum(conv_signal[l_min_index:l_max_index]) == 0:
            l_miss_cnt += 1
            pass  # l_center = l_center
        else:
            l_miss_cnt = 1
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) \
                       + l_min_index - offset

        # Find the best right centroid by using past right center
        r_min_index = int(max(r_center + offset - r_margin, 0))
        r_max_index = int(min(r_center + offset + r_margin, image.shape[1]))

        if sum(conv_signal[r_min_index:r_max_index]) == 0:
            r_miss_cnt += 1
            pass  # r_center = r_center
        else:
            r_miss_cnt = 1
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) \
                       + r_min_index - offset

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))
        # save init center data
        if l_center_init is None:
            l_center_init = l_center
        if r_center_init is None:
            r_center_init = r_center

#    # if the data is equivalent to *_center, remove the data
#    out_centroids = []
#    for center in window_centroids:
#        if (center[0] == l_center) | (center[1] == r_center):
#            pass
#        else:
#            out_centroids.append(center)
#    return out_centroids

    return [window_centroids, l_center_init, r_center_init]


def find_lanes(image, centroids, window_width, window_height):
    # If we found any window centers
    if len(centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0, len(centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,
                                 window_height,
                                 image,
                                 centroids[level][0],
                                 level)

            r_mask = window_mask(window_width,
                                 window_height,
                                 image,
                                 centroids[level][1],
                                 level)

            # If the masked place has no value, simply skip
            if (image*l_mask).sum() > 5000:
                l_points[(l_points == 255) | ((l_mask == 1))] = 255

            if (image*r_mask).sum() > 5000:
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Add graphic points from window mask here to total pixels found

        # Draw the results
        # add both left and right window pixels together
        template = np.array(r_points + l_points, np.uint8)

#        # create a zero color channel
#        zero_channel = np.zeros_like(template)

#        # make window pixels green
#        template = np.array(cv2.merge((zero_channel, template, zero_channel)),
#                            np.uint8)

        l_template = np.array(l_points, np.uint8)
        r_template = np.array(r_points, np.uint8)

        # create a zero color channel
        zero_channel = np.zeros_like(l_template)

        # make window pixels green
        template = np.array(cv2.merge((zero_channel, l_template, r_template)),
                            np.uint8)

        # making the original road pixels 3 color channels
        warpage = np.array(cv2.merge((image, image, image)), np.uint8)

        # overlay the orignal road image with window results
        mixed = cv2.addWeighted(warpage, 0.7, template, 0.5, 0.0)

    # If no window centers found, just display orginal road image
    else:
        template = np.array(cv2.merge((image, image, image)), np.uint8)
        mixed = np.array(cv2.merge((image, image, image)), np.uint8)

    num_left = np.sum(l_template) / (255 * window_width * window_height)
    num_right = np.sum(r_template) / (255 * window_width * window_height)
    print(num_left, num_right)
    
    return [template, mixed, num_left, num_right]


def line_and_margin(l_x, l_y, r_x, r_y, margin, color, img_sample):
    """
    simple visualization of the line and margin with color
    """
    overlay = np.zeros_like(img_sample).astype(np.uint8)

    # left line
    l_pts = np.array([[x, y] for x, y in zip(l_x, l_y)], np.int32)
    l_pts_plt = l_pts.reshape((-1, 1, 2))

    # right line
    r_pts = np.array([[x, y] for x, y in zip(r_x, r_y)], np.int32)
    r_pts_plt = r_pts.reshape((-1, 1, 2))

    # fill image (with margin)
    tmp = np.zeros(l_pts.shape)
    tmp[:, 0] = margin
    l_l_pts = l_pts - tmp
    r_l_pts = l_pts + tmp
    l_r_pts = r_pts - tmp
    r_r_pts = r_pts + tmp
    l_margin_pts = np.concatenate((l_l_pts, np.flipud(r_l_pts)), axis=0)
    l_margin_pts = l_margin_pts.reshape((1, 1440, 2))
    r_margin_pts = np.concatenate((l_r_pts, np.flipud(r_r_pts)), axis=0)
    r_margin_pts = r_margin_pts.reshape((1, 1440, 2))
    m_pts = np.concatenate((r_l_pts, np.flipud(l_r_pts)), axis=0)
    m_pts = m_pts.reshape((1, 1440, 2))

    # fill image (without margin)
    m_pts = np.concatenate((l_pts, np.flipud(r_pts)), axis=0)
    m_pts = m_pts.reshape((1, 1440, 2))

#    cv2.fillPoly(overlay, np.int_([l_margin_pts]), (255, 255, 0))
#    cv2.fillPoly(overlay, np.int_([r_margin_pts]), (255, 255, 0))
    cv2.fillPoly(overlay, np.int_([m_pts]), (0, 255, 0))
    cv2.polylines(overlay, [l_pts_plt], False, (255, 0, 0), thickness=10)
    cv2.polylines(overlay, [r_pts_plt], False, (0, 0, 255), thickness=10)

#    plt.imshow(overlay)
#    plt.show()

    return overlay

if __name__ == '__main__':

    # Read in a thresholded image
    warped = mpimg.imread('test_preprocess/top_down_sample.jpg')

    # histogram1 = np.sum(warped[int(warped.shape[0]/2):, :], axis=0)
    # histogram2 = np.sum(warped[:, :], axis=0)  # all
    # plt.plot(histogram2)
    # plt.show()

    # window settings
    window_width = 50
    window_height = 40  # Break image into 9 vertical layers (height 720)
    margin = 100  # slide window width for searching

    # Find the centroids for each window
    window_centroids = find_window_centroids(warped,
                                             window_width,
                                             window_height,
                                             margin)

    print(window_centroids)
    # find lane path
    img_lane, img_both = find_lanes(warped, window_centroids,
                                    window_width, window_height)

    # Display the final results
    plt.imshow(img_both)
    plt.title('window fitting results')
    plt.show()
