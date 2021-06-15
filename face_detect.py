import cv2 as cv

# https://www.youtube.com/watch?v=oXlwWbU8l2o

img = cv.imread('gapofitzy.jpg')
# cv.imshow('Person', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# reads the haar cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
# minNeighbours refer to the number of sides of a rectangle - 1 which is 3

print(f'Number of faces = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)
