import pandas as pd;
import numpy as np ;
import cv2 as cv;
import random 
import time 
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


class model():

    def __init__(self):
        return ;
    
    def train(self,inputs,labels,num_epoch=100,lr = 0.001,target_num = 0, noChange_limit_num = 10000):
        self.w = np.zeros((len(inputs[1]),1))
        self.b = 0;

        for epoch in range(num_epoch):
            correct_num = 0;

            for index in range(len(inputs)):
                x = inputs[index];
                x = np.reshape(x,(784,1))
                y = int(labels[index] == target_num)*2-1;
                re = y*(np.dot(self.w.T,x)+self.b)

                if(re<=0):
                    self.w = self.w+y*x*lr;
                    self.b = self.b+y*lr;
                    correct_num = 0;
                elif(re>0):
                    correct_num += 1;
                    if(correct_num>noChange_limit_num):
                        break;
            
            print("epoch:{}".format(epoch));

    def predict(self,inputs,target_num = 0):
        pre = [];
        for x in inputs:
            re = (np.dot(self.w.T,x) + self.b);
            result = int(re>0);
            pre.append(result);
        return pre;

    def makeVision(self,inputs,out):
        for i in range(100):
            r = random.randint(0,20000)
            img = inputs[r];
            img = np.reshape(img,(28,28))
            print(img.shape)
            plt.imshow(img);
            plt.title(out[r])
            plt.show()


if __name__ == "__main__":
    
    origin_train_data = pd.read_csv("./data/digit-recognizer/train.csv",header=0)
    train_data = origin_train_data.values;
    train_imgs = train_data[0:,1:]
    train_labels = train_data[:,0]
    origin_test_data = pd.read_csv("/Users/wzk1998/Desktop/AllCode/My_machineLearning/binary_recognition/digit-recognizer/test.csv",header=0)
    test_data = origin_test_data.values
    print(test_data)
    test_imgs = test_data[0:,0:]
    test_labels = test_data[:,0]


    m = model();
    m.train(train_imgs,train_labels)
    out = m.predict(test_imgs)

    m.makeVision(test_imgs,out);