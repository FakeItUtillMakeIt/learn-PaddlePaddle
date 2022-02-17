from random import sample
import paddle
from paddle.fluid.layers.nn import relu 
from paddle.nn import Conv2D,MaxPool2D,UpsamplingBilinear2D,ReLU
import paddle.nn.functional as F
from paddle.tensor.manipulation import concat
import numpy as np


class UNet(paddle.nn.Layer):
    def __init__(self):
        super(UNet,self).__init__()
    
    #两次卷积后进行下采样
    def doubleConv_Down(self,in_channel,out_channel,inputs):
        conv1=Conv2D(in_channels=in_channel,out_channels=in_channel*2,kernel_size=3)
        relu=ReLU()
        conv2=Conv2D(in_channels=in_channel*2,out_channels=out_channel,kernel_size=3)
        maxpool=MaxPool2D(kernel_size=2,stride=2)
        out=conv1(inputs)
        out=relu(out)
        out=conv2(out)
        out=relu(out)
        out1=maxpool(out)
        return out1,out

    def doubleConvConcat(self,in_channel,out_channel,sameLayer_inputs,inputs):
        conv1=Conv2D(in_channels=in_channel,out_channels=in_channel*2,kernel_size=3)
        conv2=Conv2D(in_channels=in_channel*2,out_channels=out_channel,kernel_size=3)
        relu=ReLU()
        
        out1=conv1(inputs)
        out1=relu(out1)
        out2=conv2(out1)
        out2=relu(out2)
        upconv=UpsamplingBilinear2D(size=(out2.shape[-2]*2,out2.shape[-1]*2))
        out3=upconv(out2)
        conv3=Conv2D(in_channels=out3.shape[1],out_channels=out3.shape[1]//2,kernel_size=1)
        out3_1=conv3(out3)
        out3_2=paddle.crop(sameLayer_inputs,shape=out3_1.shape)
        out4=concat(x=[out3_1,out3_2],axis=1)
        print(out4.shape)
        return out4

    # 两次卷积后进行上采样 与同层进行拼接、
    def concat_doubleConv_Up(self,in_channel,out_channel,sameLayer_inputs,inputs):
        conv1=Conv2D(in_channels=in_channel,out_channels=in_channel//2,kernel_size=3)
        conv2=Conv2D(in_channels=in_channel//2,out_channels=out_channel,kernel_size=3)
        relu=ReLU()
        
        out1=conv1(inputs)
        out1=relu(out1)
        out2=conv2(out1)
        out2=relu(out2)
        upconv=UpsamplingBilinear2D(size=(out2.shape[-2]*2,out2.shape[-1]*2))
        out3=upconv(out2)
        conv3=Conv2D(in_channels=out3.shape[1],out_channels=out3.shape[1]//2,kernel_size=1)
        out3_1=conv3(out3)
        out3_2=paddle.crop(sameLayer_inputs,shape=out3_1.shape)
        out4=concat(x=[out3_1,out3_2],axis=1)
        return out4
    
    def outConv(self,in_channel,out_channel,inputs):
        conv1=Conv2D(in_channels=in_channel,out_channels=in_channel//2,kernel_size=3)
        conv2=Conv2D(in_channels=in_channel//2,out_channels=in_channel//2,kernel_size=3)
        conv1_1=Conv2D(in_channels=in_channel//2,out_channels=out_channel,kernel_size=1)

        out=conv1(inputs)
        out=conv2(out)
        out=conv1_1(out)
        return out

    def forward(self,input):
        out11,out12=self.doubleConv_Down(in_channel=1,out_channel=64,inputs=input)
        out21,out22=self.doubleConv_Down(in_channel=64,out_channel=128,inputs=out11)
        out31,out32=self.doubleConv_Down(in_channel=128,out_channel=256,inputs=out21)
        out41,out42=self.doubleConv_Down(in_channel=256,out_channel=512,inputs=out31)

        out5=self.doubleConvConcat(in_channel=512,out_channel=1024,sameLayer_inputs=out42,inputs=out41)

        out6=self.concat_doubleConv_Up(in_channel=1024,out_channel=512,sameLayer_inputs=out32,inputs=out5)
        out7=self.concat_doubleConv_Up(in_channel=512,out_channel=256,sameLayer_inputs=out22,inputs=out6)
        out8=self.concat_doubleConv_Up(in_channel=256,out_channel=128,sameLayer_inputs=out12,inputs=out7)

        out9=self.outConv(in_channel=128,out_channel=2,inputs=out8)
        return out9


input=np.ones(shape=(1,1,572,572)).astype("float32")
# input=np.arange(1,572,0.5)
# input.reshape((1,1,572,572)).astype("float32")
input=paddle.to_tensor(input)

print(input.shape)

model=UNet()
out=model.forward(input)
print(out.shape)
print(out)


