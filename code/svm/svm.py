import numpy as np;
from data_generator import *
import pandas as pd 

import time
import random
import logging





#代码中的所有量都应该是矩阵形式，
#也就是np.array形式
#一维数组全都转换为(n,1)的矩阵
class SVM_model(object):
    def __init__(self,feature,label,kernel = "linear"):

        super(SVM_model,self).__init__();

        '''
        feature以及label都是矩阵
        '''

        self.n = feature.shape[1]               #特征长度
        self.featureNum = feature.shape[0]        #特征数量
        self.C = 1000
        self.X = feature
        self.Y = label
        self.b = 0.0
        self.alpha = np.zeros((feature.shape[0],1));
        print("featureNum = ",self.featureNum);
        print("alpha shape:",self.alpha.shape)
        self.kernel = kernel
        self.eps = 1e-5;
        self.E = np.array([self.getE(i) for i in range(len(feature))]).reshape(len(feature),1)

        

    def judge_KKT(self,index):

        yGx = self.Y[index,0]*self.getG(index);

        if(abs(self.alpha[index,0])<self.eps):
            return  yGx>1 or abs(yGx-1)<self.eps;
        
        elif(abs(self.alpha[index,0]-self.C)<self.eps):
            return yGx<1 or abs(yGx-self.C)<self.eps
        else:
            return abs(yGx-1)<self.eps;

    def isStop(self):
        '''
        判断alpha的选取是否结束，返回boolean类型的值
        true为结束
        false为不结束
        '''

        for index in range(self.featureNum):
            if(not self.judge_KKT(index)):
                return False;
        return True;


    def select_alpha(self):

        '''

        选取出两个最为合适的alpha变量
        '''

        index_list = [i for i in range(self.featureNum)];

        list_0toC = list(filter(lambda i: self.alpha[i,0]>0 and self.alpha[i,0]<self.C, index_list))
        list_other = list(set(index_list)-set(list_0toC))
        list_0toC.extend(list_other);
        list_re = list_0toC;



        for index in list_re:
            
            if(self.judge_KKT(index) == True):
                continue;
            
            mmax = 0.0;
            maxindex = 0;

            for j in list_re:
                if(index == j) :
                    continue;
                temp = abs(self.getE(j)-self.getE(index))
                if(temp>mmax):
                    mmax = temp;
                    maxindex = j;

            return (index,maxindex)



    def k_kernel(self,x1,x2):
        '''
        x1,x2均为向量
        '''

        if(self.kernel == "linear"):
            return np.dot(x1,x2);
        elif(self.kernel == "poly"):
            return (np.dot(x1,x2.T)+1)**3;

    def getG(self,index):

        result = 0.0;

        for j in range(self.featureNum):

            result += self.alpha[j,0]*self.Y[j,0]*self.k_kernel(self.X[index],self.X[j]);
        
        result += self.b;
        return result;

    def getE(self,index):
        '''
        g(Xi)-Yi
        '''
        return self.getG(index) - self.Y[index,0];
    
    def train(self,num_epoch = 1000):
        
        for epoch in range(num_epoch):

            print("epoch = {}".format(epoch));

            if(self.isStop()):
                return;
            
            i_1,i_2 = self.select_alpha();

            L = max(0,self.alpha[i_2,0]-self.alpha[i_1,0])
            H = min(self.C,self.C + self.alpha[i_2,0] - self.alpha[i_1,0])
            K11 = self.k_kernel(self.X[i_1],self.X[i_1])
            K22 = self.k_kernel(self.X[i_1],self.X[i_2])
            K12 = self.k_kernel(self.X[i_1],self.X[i_2])

            beta = K11 + K22 - 2*K12

            a2New =  self.alpha[i_2,0] + self.Y[i_2,0]*(self.E[i_2,0]-self.E[i_1,0])/beta;

            if(a2New > H):
                a2New = H;
            elif(a2New < L):
                a2New = L;
            else:
                a2New = a2New;

            a1New = self.alpha[i_1,0] + self.Y[i_1,0]*self.Y[i_2,0]*(self.alpha[i_2,0] - a2New);

            b1New = -self.E[i_1,0]-self.Y[i_1,0]*K11*(a1New - self.alpha[i_1,0]) - self.Y[i_2,0]*K12*(a2New-self.alpha[i_2,0]) + self.b;
            b2New = -self.E[i_2,0]-self.Y[i_2,0]*K22*(a2New - self.alpha[i_2,0]) - self.Y[i_1,0]*K12*(a1New-self.alpha[i_1,0]) + self.b;

            if(a1New>0 and a1New<self.C and a2New>0 and a2New<self.C):
                bNew = b1New;
            else:
                bNew = b1New + b2New
            

            self.alpha[i_1,0] = a1New;
            self.alpha[i_2,0] = a2New;
            self.b = bNew;

            self.E[i_1,0] = self.getE(i_1);
            self.E[i_2,0] = self.getE(i_2);

    







if __name__ == "__main__":
    
    train_set,train_label,test_set,test_label = generate_dataset(2000,visualization = False);

    train_set = np.array(train_set)
    train_label = np.array(train_label).reshape(len(train_label),1)
    test_set = np.array(test_set)
    test_label = np.array(test_label).reshape(len(test_label),1)
    print(train_set.shape)
    print(train_label.shape)
    svm = SVM_model(train_set,train_label);
    svm.train(num_epoch=10)
    
    





