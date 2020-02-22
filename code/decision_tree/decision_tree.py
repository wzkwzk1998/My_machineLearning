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
    return features

def cal_ent(labels):
    '''
    此函数可以优化，计算经验熵
    '''
    labels_set = set([labels[i] for i in range(len(labels))])
    ent = 0.0;
    for x_label in labels_set:
        p = float(labels[labels==x_label].shape[0])/labels.shape[0];
        p *= np.log2(p);
        ent -= p;
    
    return ent;
def cal_condition_ent(x,y):
    '''
    x是特征，y是标签
    计算条件经验熵 H(D|A)
    '''
    x_value_set = set([x[i] for i in range(len(x))])
    ent=0.0;

    for x_value in x_value_set:
        yy = y[x == x_value];
        temp_ent = cal_ent(yy);
        ent += (float(yy.shape[0])/float(y.shape[0]))*temp_ent;

    return ent;


def create_Tree(inputs,labels,features,step,eps = 0.1):
    '''
    递归构建决策树
    '''
    print(step);
    print("len==",len(features))
    LEAF = "leaf"
    INTERNAL = "internal"
    labels_set = set([labels[i] for i in range(len(labels))]);
    

    #全部属于一个类
    if(len(labels_set)==1):               
        return tree.TreeNode("leaf",labels_set.pop());


    #特征为空
    max_class,max_num = max([(i,len(list(filter(lambda x: x==i, labels)))) for i in range(class_num)],key = lambda x: x[1])
    if(len(features)==0):                
        return tree.TreeNode(LEAF,Class = max_num);

    #计算信息增益
    print("start cal");
    max_feature = 0;
    max_gda = 0;

    HD = cal_ent(labels)
    labels = np.array(labels);
    inputs = np.array(inputs);
    for fea in features:
        A = np.array(inputs[:,fea]);
        HDA = cal_condition_ent(A,labels);
        G = HD - HDA;
        if(G > max_gda):
            max_gda = G;
            max_feature = fea;
    print("stop cal");
    
    #信息增益小于阀值
    if(max_gda<eps):
        return tree.TreeNode(LEAF,Class = max_class, feature = max_feature)
    
    sub_features = filter(lambda x: x!=max_feature,features)
    treeNode = tree.TreeNode(INTERNAL,feature=max_feature)
    
    #子树个数,就是这个特征有多少个取值
    features = np.array(features);
    feature_num = inputs[:,max_feature]
    feature_set = set([feature_num[i] for i in range(feature_num.shape[0])]);

    #构建子树
    print(inputs.shape)
    print("start create")
    for feature in feature_set:
        sub_input_list = [];
        sub_label_list = [];
        for i in range(len(labels)):
            if(inputs[i,max_feature] == feature):
                sub_input_list.append(inputs[i])
                sub_label_list.append(labels[i])
        sub_input = np.array(sub_input_list)
        sub_label = np.array(sub_label_list) 

        sub_tree = create_Tree(sub_input,sub_label,sub_features,step+1,eps)
        treeNode.add_child(feature,sub_tree);
    return treeNode;
    print("stop create");

def train(inputs,labels,eps = 0.1):
    return create_Tree(inputs,labels,[i for i in range(feature_num)],1,eps);

def predict(inputs,root):
    output = [];
    for x in inputs:
        temp_predict = root.predict(x);
        output.append(temp_predict);
    return output;


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
    print(train_labels.shape)
    treeRoot = train(train_imgs,train_labels)
    print("stop")






    