import cv2

camera = cv2.VideoCapture(0)

for i in range(10):
	return_value, image = camera.read()
	cv2.imwrite('left'+str(i)+'.jpg', image)
	cv2.waitKey(500)
del(camera)