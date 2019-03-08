import cv2
import numpy as np
import glob
import yaml



# A partir de les cantonades i dels punts dels eixos
# retorna una imatge resultat de dibuixar els un eix 3D a img
def draw(img, corners, imgpts):
	imgpts = np.int32(imgpts).reshape(-1,2)

	# draw ground floor in green
	img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

	# draw pillars in blue color
	for i,j in zip(range(4),range(4,8)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

	# draw top layer in red color
	img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

	return img
	
	
	
# Primer de tot carregem la configuracio de calibratge
# en el tutorial fan servir aquest metode, pero nosaltres l'hem guardat amb el yaml

# Load previously saved data
with np.load('calibration.z.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
	
# Els dos ultims parametres rvecs i tvecs no son necessaris, de fet amb el yaml no els hem guardat

#with open('calibration.yaml') as f:
#	loadeddict = yaml.load(f)

#mtx = loadeddict.get('camera_matrix')
#dist = loadeddict.get('dist_coeff')	
	
# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Creem un eix 3D de 3 unitats en cada direccio
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])

#carregem totes les imatges i busquem un grid 7*6
#si es troba es calcula la rotacio i la traslacio i les utilitzem per a la nostre projeccio

for fname in glob.glob('left*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0)
        if k == 's':
            cv2.imwrite(fname+'_cube.png', img)

cv2.destroyAllWindows()