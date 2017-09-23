import cv2
import numpy
import matplotlib.pyplot as plt

# cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
while True:
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, binaryImage = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.imshow('img', cv2.flip(img, 1))
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
