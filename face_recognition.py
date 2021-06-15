import numpy as np
import cv2 as cv
import urllib

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Yeji', 'Yuna']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

req = urllib.urlopen('https://qph.fs.quoracdn.net/main-qimg-5fde0c2a6f07897a592a42484aab12e9')
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img_url = cv.imdecode(arr, -1)
# 'Load it as it is'
cv.imshow('lalala', img_url)

img = cv.imread(r'C:\Users\iance\PycharmProjects\datasciencepy\Test\tumblr_4f6762465401b3b88a3b51da88a44783_532234cd_540.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    confidence_rounded = round(confidence, None)
    print(f'Label = {people[label]} with a confidence of {confidence_rounded}%')

    cv.putText(img, str(people[label] + f" = {confidence_rounded}%"), (x-5,y-5), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), thickness=2)

cv.imshow(f"Face Detection of {people}", img)

cv.waitKey(0)