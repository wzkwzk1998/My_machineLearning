import numpy as np 



#保存树的结点结构
class TreeNode(object):
    def __init__(self,node_type,Class=None,feature=None):
        '''
            note_type : 记录叶节点的类型
            child : 记录每一个特征的两个分类。
            feature : 记录这个叶节点基于哪一个特征进行分类
            Class : 最终分类
            -------------------------
        '''
        self.node_type = node_type;          #记录叶节点的类型
        self.childs = {};                   #记录每一个特征的两个分类。
        self.feature = feature;             #记录这个叶节点基于哪一个特征进行分类
        self.Class = Class;                #最终分类

    def add_child(self,index,TreeNode):
        self.childs[index] = TreeNode;

    def predict(self,inputs):
        if self.node_type == "leaf":
            return self.Class;
        treeNode = self.child[inputs[self.feature]];
        return treeNode.predict(inputs);


