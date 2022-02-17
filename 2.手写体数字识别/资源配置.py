import os
import random
from numpy.core.defchararray import index
from numpy.lib.arraypad import pad
import paddle
from paddle.fluid.layers import learning_rate_scheduler
from paddle.nn import Conv2D,MaxPool2D,Linear
import paddle.nn.functional as F
import numpy as np
from PIL import Image
import gzip
import json

from paddle.tensor.random import rand

#定义数据集读取器
def load_data(mode="train"):
    datafile="./data/mnist.json.gz"
    data=json.load(gzip.open(datafile))
    #读取数据集中的训练集，测试机和验证集
    train_set,val_set,eval_set=data
    #数据集相关参数
    IMG_ROWS=28
    IMG_COLS=28
    #根据输入mode决定使用数据集
    if mode=="train":
        imgs=train_set[0]
        labels=train_set[1]
    elif mode=="valid":
        imgs=val_set[0]
        labels=val_set[1]
    elif mode=="eval":
        imgs=eval_set[0]
        labels=eval_set[1]
    
    imgs_length=len(imgs)
    #验证图像数量和标签数量是否一致
    assert imgs_length==len(labels),"length of imgs:{} should be equal to length of labels:{}"

    index_list=list(range(imgs_length))
    #定义读取数据时的batchsize
    BATCH_SIZE=100
    #定义数据生成器
    def data_generator():
        if mode=="train":
            random.shuffle(index_list)
        imgs_list=[]
        labels_list=[]
        for i in index_list:
            #读取图像和标签转换尺寸和类型
            img=np.reshape(imgs[i],[1,IMG_ROWS,IMG_COLS]).astype("float32")
            label=np.reshape(labels[i],[1]).astype("int64")
            imgs_list.append(img)
            labels_list.append(label)
            #如果当前数据缓存达到了batchsize,返回一个批次数据
            if len(imgs_list)%BATCH_SIZE==0:
                yield np.array(imgs_list),np.array(labels_list)
                imgs_list=[]
                labels_list=[]
        #最后一个batch
        if len(imgs_list)>0:
            yield np.array(imgs_list),np.array(labels_list)
    return data_generator

#定义模型结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST,self).__init__()
        #定义卷积层
        self.conv1=Conv2D(in_channels=1,out_channels=20,kernel_size=5,stride=1,padding=2)
        #定义池化层
        self.max_pool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(in_channels=20,out_channels=20,kernel_size=5,stride=1,padding=2)
        self.max_pool2=MaxPool2D(kernel_size=2,stride=2)
        #定义全连接层
        self.fc=Linear(in_features=980,out_features=10)

    #定义前向训练
    def forword(self,input):
        x=self.conv1(input)
        x=F.relu(x)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.max_pool2(x)
        x=paddle.flatten(x)
        x=self.fc(x)
        return x

#单GPU训练
#通过设置paddle.set_device(device) API 设置在GPU还是CPU上训练
#仅优化算法的设置有所差别
def train(model):
    #开启GPU
    use_gpu=True
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
    model.train()
    #
    train_loader=load_data("train")
    #设置不同学习率
    opt=paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters())
    EPOCH_NUM=5
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)
            #前向计算过程
            predicts=model.forword(images)
            #计算loss
            loss=F.cross_entropy(predicts,labels)
            avg_loss=paddle.mean(loss)
            #训练100 batchsize打印loss
            if batch_id%100==0:
                print("epochs:{},batchs:{},loss:{}".format(epoch,batch_id,avg_loss.numpy()[0]))
            #后向传播
            loss.backward()
            #
            opt.step()
            opt.clear_grad()
    #保存模型参数
    paddle.save(model.state_dict(),"./mnist.pdparams")


model=MNIST()
#启动训练
train(model)

#分布式训练
#1.模型并行

#2.数据并行
#需要梯度同步机制
#梯度同步机制：PRC（CPU）  NCCL2（GPU）
import paddle.distributed as dist
def train_multi_gpu(model):
    #修改1.初始化并行环境
    dist.init_parallel_env()
    #修改2.增加paddle.DataParallel封装
    model=paddle.DataParallel(model)
    model.train()

    train_loader=load_data("train")
    opt=paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters())
    EPOCH_NUM=5
    for i in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)
            predicts=model.forward(images)
            loss=F.cross_entropy(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%100==0:
                print("epoch:{},batch:{},loss:{}".format(i,batch_id,avg_loss.numpy()[0]))
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
    paddle.save(model.state_dict(),"./mnist_multi_gpu.pdparams")

model=MNIST()
train_multi_gpu(model)
#1.基于launch方式启动
#设置单机多卡启动时  分布式运行
#python -m paddle.distribute.launch train.py 默认使用所有GPU
#python -m paddle.distribute.launch --gpus '0,1' train.py 使用0，1号gpu

#2.基于spawn方式启动
#dist.spawn(train) #使用所有卡
#dist.spawn(train,nprocs=2,selected_gpus='1,2')#指定1，2张gpu




