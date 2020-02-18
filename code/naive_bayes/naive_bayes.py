import pandas as pd 
import numpy as np 
import cv2
import random 
import time 
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


class naive_bayes_model():

    def __init__(self):
        self.class_num = 10;
        return ;
    def binaryzation(self,img):
        cv_img = img.astype(np.uint8)
        _,cv_img = cv2.threshold(cv_img,127,1,cv2.THRESH_BINARY_INV);
        return cv_img;
    def Train(self,inputs,labels):

        self.feature_len = inputs.shape[1];
        self.prior_p = np.zeros(self.class_num)
        self.condition_p = np.zeros((self.feature_len,2,self.class_num))
        print(self.condition_p.shape)

        #计算先验概率以及条件概率
        for i in range(len(labels)):
            print("The {}th img".format(i));
            x = inputs[i];
            x = self.binaryzation(x);
            self.prior_p[labels[i]]+=1;

            for j in range(self.feature_len):
                self.condition_p[j][x[j][0]][labels[i]]+=1;
        
        #对概率进行进一步计算然后归一化
        for i in range(self.class_num):
            for j in range(self.feature_len):
                p_0 = self.condition_p[j][0][i];
                p_1 = self.condition_p[j][1][i];

                rp_0 = (float(p_0)/float(p_0+p_1))*10000+1;
                rp_1 = (float(p_1)/float(p_0+p_1))*10000+1;

                self.condition_p[j][0][i] = rp_0;
                self.condition_p[j][1][i] = rp_1;
        print("train end")

    def cal_p(self,img,lable):
        re = int(self.prior_p[lable]);
        for i in range(self.feature_len):
            re *= int(self.condition_p[i][img[i][0]][lable]);
        return re;

    def predict(self,inputs):
        output = [];
        for i in range(len(inputs)):
            print("predict {}th img".format(i));
            x = inputs[i];
            x=self.binaryzation(x);
            mmax_p = 0;
            mmax_label = 0;
            for y in range(self.class_num):
                p = self.cal_p(x,y);
                
                if(p>mmax_p):
                    mmax_p = p;
                    mmax_label = y;

            output.append(mmax_label);
        return output;

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


    origin_train_data = pd.read_csv('./data/digit-recognizer/train.csv',header=0)
    train_data = origin_train_data.values;
    train_imgs = train_data[0:,1:]
    train_labels = train_data[:,0]

    origin_test_data = pd.read_csv("./data/digit-recognizer/test.csv",header=0)
    test_img = origin_test_data.values;

    nb = naive_bayes_model();
    nb.Train(train_imgs,train_labels);
    output = nb.predict(test_img);
    nb.makeVision(test_img,output)


    
    
        