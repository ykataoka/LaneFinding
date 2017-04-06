import cv2
import numpy as np

def corners_unwarp(img, nx, ny, mtx, dist):
    """
    @desc : warp the image
    @param img : original image
    @param nx, ny : number of the data
    """
    # Convert undistorted image to grayscale
    if img.shape[-1] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 4 corners in the image coordinate
#    src = np.float32([[600, 450],
#                      [685, 450],
#                      [1050, 690],
#                      [250, 690]])

    # 4 corners in the image coordinate (image 2 next lane)
    offset_tmp = -40
    src = np.float32([[535 - offset_tmp, 450],
                      [760 + offset_tmp, 450],
                      [1280, 720],
                      [0, 720]])

    # 4 corners in the warped coordinate
#    dst = np.float32([[275, 671],
#                      [1033, 671],
#                      [1033, 488],
#                      [275, 488]])

    # 4 corners in the warped coordinate (big)
    offset_x = 150
    offset_y = 30
    img_size = (gray.shape[1], gray.shape[0])
    dst = np.float32([[offset_x, offset_y],
                      [img_size[0]-offset_x, offset_y],
                      [img_size[0]-offset_x, img_size[1]-offset_y],
                      [offset_x, img_size[1]-offset_y]])

    # Perspective Transform Matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    scaled_img = np.uint8(255*img)  # necessary for warpPerspective
    warped = cv2.warpPerspective(scaled_img, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv


if __name__ == '__main__':
    print('hoge')
