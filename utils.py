import sys
from operator import itemgetter
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def rect2square(rectangles):
    #-----------------------------#
    #   将长方形调整为正方形
    #-----------------------------#
    w = rectangles[:, 2] - rectangles[:, 0]
    h = rectangles[:, 3] - rectangles[:, 1]
    l = np.maximum(w, h).T
    rectangles[:, 0] = rectangles[:, 0] + w*0.5 - l*0.5
    rectangles[:, 1] = rectangles[:, 1] + h*0.5 - l*0.5
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
    return rectangles


def Alignment_1(img, landmark):
    #-------------------------------------#
    #   人脸对齐
    #-------------------------------------#
    if landmark.shape[0] == 68:
        x = landmark[36, 0] - landmark[45, 0]
        y = landmark[36, 1] - landmark[45, 1]
    elif landmark.shape[0] == 5:
        x = landmark[0, 0] - landmark[1, 0]
        y = landmark[0, 1] - landmark[1, 1]
        # print('x=',x,'y=',y)

    if x == 0:
        angle = 0
    else:
        angle = math.atan(y/x)*180/math.pi

    center = (img.shape[1]//2, img.shape[0]//2)

    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    new_img = cv2.warpAffine(img, RotationMatrix, (img.shape[1], img.shape[0]))

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(RotationMatrix[0, 0]*landmark[i, 0] +
                   RotationMatrix[0, 1]*landmark[i, 1]+RotationMatrix[0, 2])
        pts.append(RotationMatrix[1, 0]*landmark[i, 0] +
                   RotationMatrix[1, 1]*landmark[i, 1]+RotationMatrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def Alignment_2(img, std_landmark, landmark):
    def Transformation(std_landmark, landmark):
        std_landmark = np.matrix(std_landmark).astype(np.float64)
        landmark = np.matrix(landmark).astype(np.float64)

        c1 = np.mean(std_landmark, axis=0)
        c2 = np.mean(landmark, axis=0)
        std_landmark -= c1
        landmark -= c2

        s1 = np.std(std_landmark)
        s2 = np.std(landmark)
        std_landmark /= s1
        landmark /= s2

        U, S, Vt = np.linalg.svd(std_landmark.T * landmark)
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    Trans_Matrix = Transformation(std_landmark, landmark)  # Shape: 3 * 3
    Trans_Matrix = Trans_Matrix[:2]
    Trans_Matrix = cv2.invertAffineTransform(Trans_Matrix)
    new_img = cv2.warpAffine(img, Trans_Matrix, (img.shape[1], img.shape[0]))

    Trans_Matrix = np.array(Trans_Matrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []
        pts.append(Trans_Matrix[0, 0]*landmark[i, 0] +
                   Trans_Matrix[0, 1]*landmark[i, 1]+Trans_Matrix[0, 2])
        pts.append(Trans_Matrix[1, 0]*landmark[i, 0] +
                   Trans_Matrix[1, 1]*landmark[i, 1]+Trans_Matrix[1, 2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark


def pre_process(x):
    #---------------------------------#
    #   图片预处理
    #   高斯归一化
    #---------------------------------#
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    #---------------------------------#
    #   l2标准化
    #---------------------------------#
    output = x / np.sqrt(np.maximum(np.sum(np.square(x),
                                           axis=axis, keepdims=True), epsilon))
    return output


def calc_128_vec(model, img):
    #---------------------------------#
    #   计算128特征值
    #---------------------------------#
    face_img = pre_process(img)
    pre = model.predict(face_img)
    pre = l2_normalize(np.concatenate(pre))
    pre = np.reshape(pre, [128])
    return pre


def calc_128_vec1(model, img):
    #---------------------------------#
    #   计算128特征值
    #---------------------------------#
    face_img = img
    pre = model.predict(face_img)
    #pre = pre / np.expand_dims(np.sqrt(np.sum(np.power(pre, 2), 1)), 1)
    pre = pre / np.linalg.norm(pre, 2, -1, keepdims=True)  # normalization
    pre = np.reshape(pre, [128])
    return pre


def face_distance(face_encodings, face_to_compare):
    #---------------------------------#
    #   计算人脸距离
    #---------------------------------#
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    #---------------------------------#
    # 比较人脸
    # known_face_encodings: array to save face of database
    # face_encoding_to_check: vector of face to be check
    #---------------------------------#
    dis = face_distance(known_face_encodings, face_encoding_to_check)
    # print(dis)
    return list(dis <= tolerance)
