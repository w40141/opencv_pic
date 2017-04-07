import cv2, matplotlib
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('./images/osushi.jpeg')
# cv2.imshow('img', img)
# cv2.destroyAllWindows()
# k = cv2.waitKey(200)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('forsale.jpg', gray)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
