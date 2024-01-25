import cv2
# import numpy as np
img = cv2.imread('/home/crrcdt123/Downloads/1.bmp')
cv2.imshow('src',img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mean = cv2.medianBlur(gray,201)
cv2.imshow('mean',mean)
diff = gray - mean
cv2.imshow('diff',diff)
cv2.imwrite('diff.jpg',mean)
cv2.waitKey(0)
_,thres_low = cv2.threshold(diff,150,255,cv2.THRESH_BINARY)#二值化
_,thres_high = cv2.threshold(diff,220,255,cv2.THRESH_BINARY)#二值化
thres = thres_low - thres_high
cv2.imshow('thres',thres)
k1 = np.zeros((18,18,1), np.uint8)
cv2.circle(k1,(8,8),9,(1,1,1),-1, cv2.LINE_AA)
k2 = np.zeros((20,20,1), np.uint8)
cv2.circle(k2,(10,10),10,(1,1,1),-1, cv2.LINE_AA)
opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, k1)
cv2.imshow('opening',opening)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k2)
cv2.imshow('closing',closing)
contours,hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)