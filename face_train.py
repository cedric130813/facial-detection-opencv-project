# https://youtu.be/oXlwWbU8l2o?t=10172
import os
import cv2 as cv
import numpy as np

people = ['Yeji', 'Yuna']
# p = []
DIR = r'C:\Users\iance\PycharmProjects\datasciencepy\Training_Folder'
# for i in os.listdir(r'C:\Users\iance\PycharmProjects\datasciencepy\Training_Folder'):
#     p.append(i)
features = []
labels = []

haar_cascade = cv.CascadeClassifier('haar_face.xml')


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y + h, x:x + w]
                features.append(face_roi)
                labels.append(label)

for i in range(1,20):
    create_train()
    print(f"{i}/20 Training Done")
# print(f'Length of the features list = {len(features)}')
# print(f'Length of the labels list = {len(labels)}')
features = np.array(features, dtype='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on features and label list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)