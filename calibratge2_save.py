import numpy as np
import cv2
import yaml

## termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Calibratge de la camera en temps real, agafant 10 frames

cap = cv2.VideoCapture(0)
found = 0
while(found < 10): # Here, 10 can be changed to whatever number you like to choose
	ret, img = cap.read() # Capture frame-by-frame
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp) # Certainly, every loop objp is the same, in 3D.
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)
		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
		found += 1
	cv2.imshow('img', img)
	cv2.waitKey(500)
# When everything done, release the capture

cv2.destroyAllWindows()
cap.release()

# Convertim totes les dades de la matriu de calibratge en un llistat per poder-los guardar

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# It's very important to transform the matrix to list.
#data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
#with open("calibration.yaml", "w") as f:
	#yaml.dump(data, f)

	
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# It's very important to transform the matrix to list.
data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff':np.asarray(dist).tolist()}
np.savez('calibration.z', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
	
	
# Si necessitem carregar la calibració fem el seguent:

#with open('calibration.yaml') as f:
#	loadeddict = yaml.load(f)

#mtxloaded = loadeddict.get('camera_matrix')
#distloaded = loadeddict.get('dist_coeff')
		