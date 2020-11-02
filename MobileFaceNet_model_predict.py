# -*- coding: utf-8 -*-
import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import join, exists, isdir, isfile
from os import listdir
import numpy as np
import dlib
import cv2
import utils as utils
from tensorflow.python.keras.models import Model
from Model_Structures.MobileFaceNet import mobile_face_net_train, mobile_face_net
from sklearn.preprocessing import normalize

def showImg(img):
    cv2.imshow('Face', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def detectFaceKeyPoints5(image):
    # gray image
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # face rects
    rects = detector(img_gray, 0)
    if len(rects) < 1:
        return None, None
    landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])
    return rects[0], landmarks
    # for i in range(len(rects)):
    #    landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
    #    for idx, point in enumerate(landmarks):
    #        pos = (point[0, 0], point[0, 1])
    #        cv2.circle(image, pos, 2, color=(0, 255, 0))

def popFace(image, width=112, height=112):
    rect, landmarks = detectFaceKeyPoints5(image)
    if rect is None:
        return None, None
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    landmarks = (landmarks - np.array([left, top]))/(bottom-top)*width
    landmarks = np.array(landmarks, dtype=np.int32)
    #cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    crop_img = image[top:bottom, left:right]
    crop_img = cv2.resize(crop_img, (width, height))
    new_img, _ = utils.Alignment_1(crop_img, landmarks)
    #new_img = new_img - 127.5
    #new_img = new_img * 0.0078125
    return new_img, rect

def feature_compare(feature1, feature2, threshold):
    dist = np.sum(np.square(feature1 - feature2))
    sim = np.dot(feature1, feature2.T)
    if sim > threshold:
        return True, sim
    else:
        return False, sim

NUM_LABELS = 5255
LOSS_TYPE = 'softmax'

# Loading the training model
model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE)
model.load_weights('./Models/mobilefaceNet_keras.h5')
#model.summary()

# Build predict model
pred_model = mobile_face_net()
#pred_model.summary()

# Extracting the weights & transfering to the prediction model
temp_weights_list = []
for layer in model.layers:
    if 'dropout' in layer.name:
        continue
    temp_layer = model.get_layer(layer.name)
    temp_weights = temp_layer.get_weights()
    temp_weights_list.append(temp_weights)

for i in range(len(pred_model.layers)):
    pred_model.get_layer(pred_model.layers[i].name).set_weights(temp_weights_list[i])


'''Verifying the results''' 
#x = np.random.rand(1, 112, 112, 3)
#dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)
#y1 = dense1_layer_model.predict(x)[0]
#y2 = pred_model.predict(x)[0]
#for i in range(128):
#    assert y1[i] == y2[i]

# Predict face
known_face_encodings = []  # save face coder
known_face_names = []  # save face label

dbpath = './face_dataset'
shapefile = './Models/shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapefile)
for f in listdir(dbpath):
    if not isfile(join(dbpath, f)):
        continue
    image = cv2.imread(join(dbpath, f))
    facename = f.split(".")[0]
    faceimg, _ = popFace(image)
    dim_img = np.expand_dims(faceimg, axis=0)
    #print(dim_img.shape)
    face_encoding = utils.calc_128_vec1(pred_model, dim_img)

    known_face_encodings.append(face_encoding)
    known_face_names.append(facename)

#print(known_face_encodings)

#showImg(img)
# predict
image = cv2.imread('C:/tmp/img/6.jpg')
faceimg, _ = popFace(image)
dim_img = np.expand_dims(faceimg, axis=0)
face_encoding = utils.calc_128_vec1(pred_model, dim_img)

# compare
possiblies = []
possiblies = np.dot(known_face_encodings, face_encoding.T)
print(possiblies)
best_match_index = np.argmax(possiblies)
predictname = known_face_names[best_match_index]
print(predictname)