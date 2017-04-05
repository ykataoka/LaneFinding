import numpy as np
import cv2
import glob
import pickle as pl
import matplotlib.pyplot as plt

flg_cal = False  # True : needs corner detection, False : no need

nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y

"""
Create New Calibration Data
"""
if flg_cal is True:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    print('start calibration for each images ...')
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            idx = fname[fname.find('calibration')+11:].split('.')[0]
            write_name = 'camera_cal/corners_found'+idx+'.jpg'
            cv2.imwrite(write_name, img)

    # Save the camera calibration result for later use
    out = {"objpoints": objpoints, "imgpoints": imgpoints}
    pl.dump(out, open("camera_cal/calibration.p", "wb"))



"""
Read Past Calibration Data
"""
if flg_cal is False:
    calibration = pl.load(open("camera_cal/calibration.p", "rb"))
    objpoints = calibration["objpoints"]
    imgpoints = calibration["imgpoints"]
print('done calibration data reading!')

# Do camera calibration given object points and image points
img_size = (1280, 720)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   img_size,
                                                   None, None)

# Undistort Image and plot them
print('Undistort Image')
for fname in images:
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    idx = fname[fname.find('calibration')+11:].split('.')[0]
    cv2.imwrite('camera_cal/undistort' + idx + '.jpg', dst)

    # plot comparison
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    f.savefig('camera_cal/comparison' + idx + '.png', dpi=300)
