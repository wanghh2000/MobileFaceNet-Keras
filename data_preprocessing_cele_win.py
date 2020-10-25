# -*- coding: utf-8 -*-
import os
# Set log level before import, 0-debug(default) 1-info 2-warnning 3-error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from skimage import transform
from mtcnn.mtcnn import MTCNN
import numpy as np
import random
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IMG_SHAPE = (112, 112)  # in HW form
detector = MTCNN()
src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)

if IMG_SHAPE == (112, 112):
    src[:, 0] = src[:, 0] + 8.0

"""Building helper functions"""


def face_detection(img, detector):

    info = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img)
    if len(results) == 0:
        return []
    elif len(results) > 0:
        for i in range(len(results)):
            result = results[i]
            confidence = result['confidence']
            box = np.array(result['box'], np.float32)
            keypoints_dict = result['keypoints']
            keypoints = []
            for key in keypoints_dict:
                keypoints.append(keypoints_dict[key])
            keypoints = np.array(keypoints, dtype=np.float32)

            info.append([confidence, box, keypoints])

        return info


def face_alignment(img, detector):
    info = face_detection(img, detector)
    if len(info) <= 0 or len(info) > 1:
        return None
    else:
        face_info = info[0]
        assert(len(face_info) == 3)
        keypoints = face_info[2]

        transformer = transform.SimilarityTransform()
        transformer.estimate(keypoints, src)
        M = transformer.params[0: 2, :]
        warped_img = cv2.warpAffine(
            img, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue=0.0)

        return warped_img


"""Data clean & augmentation"""


def data_clean_and_augmentation(input_path, output_path):

    #nums_of_imgs = []
    #paths_dir = []
    #kernel = np.ones((5, 5), dtype=np.uint8)

    for directory in os.listdir(input_path):
        # directory: label folder name
        # print(directory)
        if os.path.isdir(input_path + directory) and not os.path.exists(output_path + directory):
            # The name list of the images in each label folder
            path_dir = os.listdir(input_path + directory)
            # paths_dir.append(path_dir)

            # The number of images in each label folder
            #num_of_imgs = len(path_dir)
            # A list of the numbers of images in each label folder
            # nums_of_imgs.append(num_of_imgs)

            # if num_of_imgs > 30:
            
            # For each label folder:
            for name in path_dir:
                img = cv2.imread(input_path + directory + '/' + name)
                warped_img = face_alignment(img, detector)
                if warped_img is not None:
                    if not os.path.exists(output_path + directory):
                        os.makedirs(output_path + directory)
                    saving_path = output_path + directory + '/' + name
                    #print(saving_path)
                    success = cv2.imwrite(saving_path, warped_img)
                    if not success:
                        raise Exception("img " + name + " saving failed. ")
                #else:
                #    print("img " + input_path + directory + '/' + name + " no found face")


if __name__ == '__main__':
    #input_path = 'C:/bd_ai/dli/celeba/img_celeba_raw/'
    input_path = 'C:/bd_ai/dli/lfw/'
    output_path = 'C:/bd_ai/dli/celeba/test/'
    #if not os.path.exists(output_path):
    #    os.makedirs(output_path)
    data_clean_and_augmentation(input_path, output_path)

'''
if __name__ == '__main__':
	input_path = "./source/" # Source image folder path
	output_path = './result/' # Destination image folder path
	data_preprocessing_adjust_folders(input_path, output_path)
'''
