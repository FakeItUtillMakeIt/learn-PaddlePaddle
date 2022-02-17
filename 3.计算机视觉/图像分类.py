# 图像分类是根据图像的语义信息对不同类别图像进行区分，是计算机视觉的核心，是物体检测、图像分割、
# 物体跟踪、行为分析、人脸识别等其他高层次视觉任务的基础。图像分类在许多领域都有着广泛的应用，如：
# 安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册
# 自动归类，医学领域的图像识别等。

#构件卷积神经网络并应用于眼疾图像分类

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random

import paddle
from paddle.fluid.dygraph.nn import Linear
from paddle.fluid.layers.io import data
from paddle.fluid.layers.nn import pad
import paddle.nn.functional as F
from paddle.nn import Conv2D,MaxPool2D,Layer,Dropout


#对读入的图片进行预处理
def transform_img(img):
    img=cv2.resize(img,(224,224))
    #读入图像的数据格式为[h,w,c] 
    #转换为[c,h,w]
    img=np.transpose(img,(2,0,1))
    img=img.astype('float32')
    #数据归一化处理
    img=img/255.0
    img=img*2.0-1
    return img

#定义训练集数据读取器
def traindata_loader(datadir,batch_size,mode="train"):
    #将datadir目录下文件列出来
    filenames=os.listdir(datadir)
    def reader():
        if mode=="train":
            random.shuffle(filenames)
        batch_imgs,batch_labels=[],[]
        for name in filenames:
            filepath=os.path.join(datadir,name)
            img=cv2.imread(filepath)

            img=transform_img(img)

            if name[0]=='H' or name[0]=='N':
                #非病理性
                label=0
            elif name[0]=='P':#病理性
                label=1
            else:
                raise("not excepted file name")
            #放入数据列表
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_labels)==batch_size:
                yield np.array(batch_imgs).astype('float32'),np.array(batch_labels).astype('float32').reshape(-1,1)
                batch_imgs,batch_labels=[],[]
        if len(batch_labels)>0:
            yield np.array(batch_imgs).astype('float32'),np.array(batch_labels).astype('float32').reshape(-1,1)
    
    return reader

#定义验证集数据读取器
def validdata_loader(datadir,csvfile,batch_size,mode="valid"):
    #将datadir目录下文件列出来
    filenames=open(csvfile).readlines()
    def reader():
        batch_imgs,batch_labels=[],[]
        for line in filenames[1:]:
            line=line.strip().split(',')
            try:
                name=line[1]
                label=int(line[2])
            except:
                pass
            filepath=os.path.join(datadir,name)
            img=cv2.imread(filepath)
            img=transform_img(img)
            
            #放入数据列表
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_labels)==batch_size:
                yield np.array(batch_imgs).astype('float32'),np.array(batch_labels).astype('float32').reshape(-1,1)
                batch_imgs,batch_labels=[],[]
        if len(batch_labels)>0:
            yield np.array(batch_imgs).astype('float32'),np.array(batch_labels).astype('float32').reshape(-1,1)
    
    return reader

CURRENT_DIR=os.path.abspath(os.curdir)
DATADIR = os.path.join(CURRENT_DIR,'PALM-Training400/PALM-Training400')
DATADIR2 =  os.path.join(CURRENT_DIR,'PALM-Validation400')
CSVFILE = os.path.join(CURRENT_DIR,'PALM-Validation-GT/labels.csv')
# 设置迭代轮数
EPOCH_NUM = 5

#定义训练和验证过程
def train_pm(model,optimizer):

    print('start training .......')
    use_gpu=True
    try:
        paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    except:
        pass
    
    train_loader=traindata_loader(datadir=DATADIR,batch_size=10,mode="train")
    valid_loader=validdata_loader(datadir=DATADIR2,csvfile=CSVFILE,batch_size=10,mode="valid")
    for epoch in range(EPOCH_NUM):
        model.train()
        for batch_id,data in enumerate(train_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)
        
            predicts=model.forward(images)
            loss=F.binary_cross_entropy_with_logits(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%10==0:
                print("epoch:{},batch:{},loss:{}".format(epoch,batch_id,avg_loss.numpy()))
            avg_loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        model.eval()
        acces=[]
        losses=[]
        for batch_id,data in enumerate(valid_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)
            predicts=model.forward(images)
            #
            pred1=F.sigmoid(predicts)
            #计算预测概率小于0.5的类别n
            pred2=predicts*(-1.0)+1.0
            #得到两个类别的预测概率，沿第一个维度级联
            pred=paddle.concat([pred2,pred1],axis=1)
            loss=F.binary_cross_entropy_with_logits(predicts,labels)
            losses.append(loss.numpy())
            acc=paddle.metric.accuracy(pred,paddle.cast(labels,dtype='int64'))
            acces.append(acc.numpy())
        print("validation accuracy:{},loss:{}".format(np.mean(acces),np.mean(losses)))
        paddle.save(model.state_dict(),'palm.pdparams')
        paddle.save(optimizer.state_dict(),'palm.pdopts')
    
#定义评估过程
def evaluation(model,params_file_path):
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')

    print('start evaluation .......')
    model_state_dict=paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    eval_loader=eval_loader(DATADIR,batch_size=10,model=eval)

class LeNet(paddle.nn.Layer):
    def __init__(self,num_class=1):
        super(LeNet,self).__init__()
        self.conv1=Conv2D(in_channels=3,out_channels=6,kernel_size=5)
        self.max_pool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(in_channels=6,out_channels=16,kernel_size=5)
        self.max_pool2=MaxPool2D(kernel_size=2,stride=2)
        self.conv3=Conv2D(in_channels=16,out_channels=120,kernel_size=4)
        self.fc1=Linear(input_dim=300000,output_dim=64)
        self.fc2=Linear(input_dim=64,output_dim=num_class)

    def forward(self,input,label=None):
        x=self.conv1(input)
        x=F.sigmoid(x)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        x=self.max_pool2(x)
        x=self.conv3(x)
        x=F.sigmoid(x)
        x=paddle.reshape(x,[x.shape[0],-1])

        x=self.fc1(x)
        x=F.sigmoid(x)
        x=self.fc2(x)
        if label is not None:
            acc=paddle.metric.accuracy(input=x,label=label)
            return x,acc
        else:
            return x


class AlexNet(paddle.nn.Layer):
    def __init__(self,num_class=1) -> None:
        super(AlexNet,self).__init__()
        self.conv1=Conv2D(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=5)
        self.maxpool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.maxpool2=MaxPool2D(kernel_size=2,stride=2)
        self.conv3=Conv2D(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.conv4=Conv2D(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.conv5=Conv2D(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1)
        self.maxpool5=MaxPool2D(kernel_size=2,stride=2)

        self.fc1=Linear(input_dim=18816,output_dim=4096)
        self.drop_ratio1=0.5
        self.drop1=Dropout(self.drop_ratio1)
        self.fc2=Linear(input_dim=4096,output_dim=4096)
        self.drop_ratio2=0.5
        self.drop2=Dropout(self.drop_ratio2)
        self.fc3=Linear(input_dim=4096,output_dim=num_class)

    def forward(self,input,label=None):
        x=self.conv1(input)
        x=F.relu(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.maxpool2(x)
        x=self.conv3(x)
        x=F.relu(x)
        x=self.conv4(x)
        x=F.relu(x)
        x=self.conv5(x)
        x=F.relu(x)
        x=self.maxpool5(x)
        x=paddle.reshape(x,[x.shape[0],-1])
        x=self.fc1(x)
        x=F.relu(x)
        x=self.drop1(x)
        x=self.fc2(x)
        x=F.relu(x)
        x=self.drop2(x)
        x=self.fc3(x)
        return x


from VGG import *
from GoogLeNet import *
from ResNet50 import *

model=VGG()
opt=paddle.optimizer.Momentum(learning_rate=0.001,momentum=0.9,parameters=model.parameters(),weight_decay=0.001)

train_pm(model,opt)

