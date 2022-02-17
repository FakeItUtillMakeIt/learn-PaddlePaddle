
from posixpath import relpath
from numpy.core.defchararray import mod
import paddle
from paddle.fluid.incubate.fleet.base import mode
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt


#获取数据集
# train_dataset=paddle.vision.datasets.MNIST(mode="train")
# train_data=np.array(train_dataset[0][0])
# train_label=np.array(train_dataset[0][1])

# plt.figure()
# plt.imshow(train_data,cmap=plt.cm.binary)
# plt.axis('on')
# plt.show()

# print(train_data.shape,train_label.shape)


class Mnist_Net(paddle.nn.Layer):
    def __init__(self):
        super(Mnist_Net,self).__init__()
        self.conv1=paddle.nn.Conv2D(in_channels=1,out_channels=3,kernel_size=2)
        self.conv2=paddle.nn.Conv2D(in_channels=3,out_channels=8,kernel_size=2)
        self.pool=paddle.nn.MaxPool2D(kernel_size=2)
        self.relu=paddle.nn.ReLU(name=None)
        self.fc1=paddle.nn.Flatten()
        #修改输出为10（10分类任务，方便交叉熵函数计算概率）
        self.fc2=Linear(in_features=2*12*12,out_features=10)
        # self.fc3=Linear(in_features=10,out_features=1)
    def forword(self,input):
        ret1=self.conv1(input)
        ret1=self.relu(ret1)
        ret1=self.pool(ret1)
        ret2=self.conv2(ret1)
        ret2=self.relu(ret2)
        ret2=self.pool(ret2)
        ret3=self.fc1(ret2)
        ret3=self.relu(ret3)
        ret3=self.fc2(ret3)
        # ret3=self.relu(ret3)
        # ret4=self.fc3(ret3)
        # ret4=paddle.nn.Softmax(ret4)
        return ret3

class Mnist_Net1(paddle.nn.Layer):
    def __init__(self):
        super(Mnist_Net1,self).__init__()
        
        self.fc1=Linear(in_features=28*28,out_features=28)
        self.fc2=Linear(in_features=28,out_features=1)
    def forword(self,input):
        
        ret=self.fc1(input)
        ret=self.fc2(ret)
        return ret



def norm_img1(img):
    assert len(img.shape)==3
    batch_size,img_h,img_w=img.shape[0],img.shape[1],img.shape[2]
    #归一化图像
    img=img/255
    img=paddle.reshape(img,[batch_size,img_h*img_w])
    return img

def norm_img(img):
    assert len(img.shape)==3
    batch_size,img_h,img_w=img.shape[0],img.shape[1],img.shape[2]
    #归一化图像
    img=img/255
    img=paddle.reshape(img,shape=(img.shape[0],1,img.shape[1],img.shape[2]))
    return img

def train1(model):
    
    #模型训练模式
    model.train()
    train_loader=paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode="train"),
                batch_size=16,shuffle=True)
    #优化器
    opt=paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())

    EPOCH_NUM=10
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images=norm_img1(data[0]).astype('float32')
            labels=data[1].astype('float32')
            #前向计算过程
            predict=model.forword(images)
            #计算损失
            loss=F.square_error_cost(predict,labels)
            avg_loss=paddle.mean(loss)

            if batch_id%1000==0:
                print("epoch_id:{},batch_id:{},loss:{}".format(epoch,batch_id,avg_loss.numpy()))
            #后向传播
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

def train(model):
    #模型训练模式
    model.train()
    train_loader=paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode="train"),
                batch_size=32,shuffle=True)
    #优化器
    opt=paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())

    EPOCH_NUM=10
    losses=[]
    for epoch in range(EPOCH_NUM):
        for batch_id,data in enumerate(train_loader()):
            images=norm_img(data[0]).astype('float32')
            labels=data[1]
            #前向计算过程
            predict=model.forword(images)
            #计算损失  分类任务通常使用交叉熵，预测任务通常使用均方差  
            # 两者之间的预测准确率相差约0.2
            # loss=F.square_error_cost(predict,labels)
            loss=F.cross_entropy(predict,labels)
            avg_loss=paddle.mean(loss)
            losses.append(avg_loss.numpy()[0])
            if batch_id%1000==0:
                print("epoch_id:{},batch_id:{},loss:{}".format(epoch,batch_id,avg_loss.numpy()))
            #后向传播
            avg_loss.backward()
            #最小化loss过程
            opt.step()
            opt.clear_grad()
    plt.plot(losses)
    plt.savefig("./loss.png")
    plt.show()

paddle.vision.set_image_backend('cv2')
def do1():
    #仅包含全连接层
    model1=Mnist_Net1()
    train1(model1)
    paddle.save(model1.state_dict(),"./mnist1.pdparams")

    test_dataset=paddle.vision.datasets.MNIST(mode="test")
    param_dict=paddle.load("./mnist1.pdparams")
    model1.load_dict(param_dict)
    model1.eval()

    for i in range(10):
        test_data0=((test_dataset[i][0]).reshape(1,28*28)[0])/255.0

        ret=model1.forword(paddle.to_tensor(test_data0))
        print(ret.numpy().astype('int32'),test_dataset[i][1])

def do():
    #包含卷积层、池化层和非线性变化层
    model=Mnist_Net()
    
    train(model)
    paddle.save(model.state_dict(),"./mnist.pdparams")

    test_dataset=paddle.vision.datasets.MNIST(mode="test")
    param_dict=paddle.load("./mnist.pdparams")
    model.load_dict(param_dict)
    model.eval()

    for i in range(10):
        test_data0=((test_dataset[i][0]).reshape(1,1,28,28))/255.0
        ret=model.forword(paddle.to_tensor(test_data0))
        print(np.argsort(ret.numpy())[0][-1],test_dataset[i][1])

do()
