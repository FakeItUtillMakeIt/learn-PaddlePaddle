
import enum
import paddle
from paddle.fluid.data_feeder import check_shape
from paddle.fluid.layers import learning_rate_scheduler
from paddle.hapi import model
from paddle.nn import Layer,Conv2D,MaxPool2D
import paddle.nn.functional as F
import os
import random
import numpy as np
from PIL import Image
import gzip
import json

from paddle.nn.layer.common import Linear


def load_data(mode="train"):
    data_path="./mnist.json.gz"
    data=json.load(gzip.open(data_path))
    train_set,val_set,eval_set=data
    IMG_ROW,IMG_COLS=28,28
    if mode=="train":
        imgs,labels=train_set[0],train_set[1]
    elif mode=="valid":
        imgs,labels=val_set[0],val_set[1]
    elif mode=="eval":
        imgs,labels=eval_set[0],eval_set[1]
    imgs_length=len(imgs)
    assert imgs_length==len(labels),"imgs length:{},labels length:{}".format(imgs_length,len(labels))
    index_list=list(range(imgs_length))
    BATCH_SIZE=100
    def data_generator():
        if mode=="train":
            random.shuffle(index_list)
        imgs_list,labels_list=[],[]
        for i in index_list:
            img=np.reshape(imgs[i],[1,IMG_ROW,IMG_COLS]).astype('float32')
            label=np.reshape(labels[i],[1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            if len(imgs_list)==BATCH_SIZE:
                yield np.array(imgs_list),np.array(labels_list)
                imgs_list=[]
                labels_list=[]
        if len(imgs_list)>0:
            yield np.array(imgs_list),np.array(labels_list)
    return data_generator

#定义网络结构
class MNIST(paddle.nn.Layer):
    def __init__(self) -> None:
        super(MNIST,self).__init__()
        self.conv1=Conv2D(in_channels=1,out_channels=20,kernel_size=5,padding=2)
        self.max_pool1=MaxPool2D(kernel_size=2,stride=2)
        self.conv2=Conv2D(in_channels=20,out_channels=20,kernel_size=5,padding=2)
        self.max_pool2=MaxPool2D(kernel_size=2,stride=2)
        self.fc=Linear(in_features= 980,out_features= 10)
    def forword(self,inputs,label=None,check_shape=False,check_content=False):
        outputs1=self.conv1(inputs)
        outputs2=F.relu(outputs1)
        outputs3=self.max_pool1(outputs2)
        outputs4=self.conv2(outputs3)
        outputs5=F.relu(outputs4)
        outputs6=self.max_pool2(outputs5)
        outputs6=paddle.reshape(outputs6,shape=[outputs6.shape[0],980])
        outputs7=self.fc(outputs6)
        if check_shape:
            # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
             print("\n########## print network layer's superparams ##############")
             print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
             print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
             #print("max_pool1-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool1.pool_size, self.max_pool1.pool_stride, self.max_pool1._stride))
             #print("max_pool2-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool2.weight.shape, self.max_pool2._padding, self.max_pool2._stride))
             print("fc-- weight_size:{}, bias_size_{}".format(self.fc.weight.shape, self.fc.bias.shape))
             
             # 打印每层的输出尺寸
             print("\n########## print shape of features of every layer ###############")
             print("inputs_shape: {}".format(inputs.shape))
             print("outputs1_shape: {}".format(outputs1.shape))
             print("outputs2_shape: {}".format(outputs2.shape))
             print("outputs3_shape: {}".format(outputs3.shape))
             print("outputs4_shape: {}".format(outputs4.shape))
             print("outputs5_shape: {}".format(outputs5.shape))
             print("outputs6_shape: {}".format(outputs6.shape))
             print("outputs7_shape: {}".format(outputs7.shape))
             # print("outputs8_shape: {}".format(outputs8.shape))
             
        # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
        if check_content:
        # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
            print("\n########## print convolution layer's kernel ###############")
            print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
            print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

            # 创建随机数，随机打印某一个通道的输出值
            idx1 = np.random.randint(0, outputs1.shape[1])
            idx2 = np.random.randint(0, outputs4.shape[1])
            # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
            print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
            print("The {}th channel of conv2 layer: ".format(idx2), outputs4[0][idx2])
            print("The output of last layer:", outputs7[0], '\n')

        if label is not None:
            acc=paddle.metric.accuracy(input=outputs7,label=label)
            return outputs7,acc
        else:
            return outputs7

#评价函数
def evaluation(model):
    print("start evaluation")
    #定义预测过程
    param_file_path="./mnist"
    param_dict=paddle.load(param_file_path)
    model.load_dict(param_dict)

    #模型预测
    model.eval()
    eval_loader=load_data('eval')
    acc_set,avg_loss_set=[],[]
    for batch_id,data in enumerate(eval_loader()):
        images,labels=paddle.to_tensor(data[0]),paddle.to_tensor(data[1])
        predicts,acc=model.forword(images,labels)
        loss=F.cross_entropy(predicts,labels)
        avg_loss=paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    #计算多个batch的平均损失和准确率
    acc_val_mean=np.array(acc_set).mean()
    avg_val_loss_mean=np.array(avg_loss_set).mean()
    print("eval loss:{},eval acc:{}".format(avg_val_loss_mean,acc_val_mean))



#1.计算模型的分类准确率

train_loader=load_data(mode="train")
use_gpu=True
try:
    paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
except:
    pass

def train(model):
    
    model.train()
    #四种优化算法
    # opt=paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
    # opt=paddle.optimizer.Momentum(learning_rate=0.01,momentum=0.9,parameters=model.parameters())
    # opt=paddle.optimizer.Adagrad(learning_rate=0.01,parameters=model.parameters())
    opt=paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters())

    EPOCH_NUM=5
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)

            #前向计算
            predicts,acc=model.forword(images,labels,check_shape=False)
            #计算损失
            loss=F.cross_entropy(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%100==0:
                print("epoch:{},batch:{},loss:{}:acc:{}".format(epoch,batch_id,avg_loss.numpy()[0],acc.numpy()))
            #后向传播
            avg_loss.backward()
            #更新参数
            opt.step()
            #清除梯度
            opt.clear_grad()
    paddle.save(model.state_dict(),"./mnist")
model=MNIST()
#train(model)

#2.检查模型训练过程，识别潜在训练问题
#加入校验或测试，更好评价模型效果
#evaluation(model)
#3.加入正则化项，避免模型过拟合
#造成过拟合的原因是模型过于敏感，而训练数据量太少或其中的噪音太多。
#解决办法：加入正则化项
#加入办法：在优化器中加入weight_decay项
# opt = paddle.optimizer.Adam(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters()) 
def train_use_ceoff(model):
    
    model.train()
    #四种优化算法
    # opt=paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
    # opt=paddle.optimizer.Momentum(learning_rate=0.01,momentum=0.9,parameters=model.parameters())
    # opt=paddle.optimizer.Adagrad(learning_rate=0.01,parameters=model.parameters())
    opt=paddle.optimizer.Adam(learning_rate=0.01,weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),parameters=model.parameters())

    EPOCH_NUM=5
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)

            #前向计算
            predicts,acc=model.forword(images,labels,check_shape=False)
            #计算损失
            loss=F.cross_entropy(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%100==0:
                print("epoch:{},batch:{},loss:{}:acc:{}".format(epoch,batch_id,avg_loss.numpy()[0],acc.numpy()))
            #后向传播
            avg_loss.backward()
            #更新参数
            opt.step()
            #清除梯度
            opt.clear_grad()
    paddle.save(model.state_dict(),"./mnist")
#4.可视化分析
#使用matplotlib作图
#使用VisualDL可视化分析
from visualdl import LogWriter
log_writer=LogWriter("./log")

def train_use_visualDL(model):
    
    model.train()
    #四种优化算法
    # opt=paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
    # opt=paddle.optimizer.Momentum(learning_rate=0.01,momentum=0.9,parameters=model.parameters())
    # opt=paddle.optimizer.Adagrad(learning_rate=0.01,parameters=model.parameters())
    opt=paddle.optimizer.Adam(learning_rate=0.01,parameters=model.parameters())

    EPOCH_NUM=5
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images,labels=data
            images=paddle.to_tensor(images)
            labels=paddle.to_tensor(labels)

            #前向计算
            predicts,acc=model.forword(images,labels,check_shape=False)
            #计算损失
            loss=F.cross_entropy(predicts,labels)
            avg_loss=paddle.mean(loss)
            if batch_id%100==0:
                print("epoch:{},batch:{},loss:{}:acc:{}".format(epoch,batch_id,avg_loss.numpy()[0],acc.numpy()))
                log_writer.add_scalar(tag="acc",value=acc.numpy(),step=batch_id)
                log_writer.add_scalar(tag="loss",value=avg_loss.numpy(),step=batch_id)
            #后向传播
            avg_loss.backward()
            #更新参数
            opt.step()
            #清除梯度
            opt.clear_grad()
    paddle.save(model.state_dict(),"./mnist")

train_use_ceoff(model)