import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('D:\پایان نامه\نتایج\centrifuge setup\60um 1400rpm\20231205_201455.jpg' )
plt.imshow(img)
# cv.imshow("Image", img)
# cv.waitKey(0)
# cv.destroyAllWindows() # destroy all the windows
gray_img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
img2	= cv.medianBlur(gray_img,	5)
circles	= cv.HoughCircles(img2,cv.HOUGH_GRADIENT,1,200,param1=60,param2=30,minRadius=100,maxRadius=200)
circles	= np.uint16(np.around(circles))
for	i in circles[0,:]:
				#	draw	the	outer	circle
				cv.circle(img,(i[0],i[1]),i[2],(0,255,0),6)
				#	draw	the	center	of	the	circle
				cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
plt.imshow(img)
lst=[]
for (x, y, r) in circles[0, :]:
  lst.append(r)

cv = np.std(lst) / np.mean(lst) * 100
print('coefficient of varriation of droplets : ' , cv)