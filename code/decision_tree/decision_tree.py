import numpy as np 
import pandas as pd ;
import cv2 
import tree


class_num = 10;
feature_num = 784;

def binaryzation(img):
    '''
    将所有的值归一到0和1
    '''
    cv_img = img.astype(np.uint8)
    _,cv_img = cv2.threshold(cv_img,127,1,cv2.THRESH_BINARY_INV);
    return cv_img;

def binaryzation_feature(inputs):
    '''
    将所有的样本二值化
    '''
    features = [];
    for img in inputs:
        img = np.reshape(img,(28,28));
        cv_img = binaryzation(img);
        features.append(cv_img);
    
    features = np.array(features);
    print(features.shape)
    features = np.reshape(features,(-1,784))
    print(features.shape)
    return features;

def 

if __name__ == "__main__":

    origin_train_data = pd.read_csv("./data/digit-recognizer/train.csv",header=0)
    train_data = origin_train_data.values;
    train_imgs = train_data[0:,1:]
    train_labels = train_data[:,0]

    origin_test_data = pd.read_csv("./data/digit-recognizer/test.csv",header=0)
    test_data = origin_test_data.values
    test_imgs = test_data[0:,0:]
    test_labels = test_data[:,0]

    train_imgs = binaryzation_feature(train_imgs);






    