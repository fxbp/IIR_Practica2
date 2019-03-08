import numpy as np
import cv2

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + CV2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...., (6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp(:,:2) = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image pints from all the images.

objpoints = [] # 3d pint in real world space
imgpoints = [] # 2d points in image plane

# Calibratge de la camera en temps real, agafant 10 frames

cap = cv2.VideoCapture(0)
found = 0
while(found < 10): # en aquest cas es 10 frames pero es pot canviar per el nombre que es consideri necesari
	ret, img = cap.read() # Capture frame y frame
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
	#If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp) #Certainly, every loop objp is the same, in 3D.
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)
		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (9,6), corners2, ret)
		found += 1
	cv2.imshow('img',img)
	cv2.waitkey(10)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

		