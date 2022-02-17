import paddle 
from paddle.nn import Conv2D,Layer,BatchNorm2D,LeakyReLU,Pad2D,MaxPool2D,AvgPool2D,UpsamplingBilinear2D
import paddle.nn.functional as F
from paddle.tensor.math import increment
import numpy as np


class YOLOv3(Layer):
    def __init__(self):
        super(YOLOv3,self).__init__()
    
    def DBLNet(self,in_channel,out_channel,inputs):
        conv_layer=Conv2D(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding=1)
        out1=conv_layer(inputs)
        bn_layer=BatchNorm2D(num_features=out1.shape[1])
        out2=bn_layer(out1)
        act_fun=LeakyReLU()
        out3=act_fun(out2)
        return out3

    def DBLNet5(self,in_channel,out_channel,inputs):
        out1=self.DBLNet(in_channel=in_channel,out_channel=out_channel,inputs=inputs)
        out2=self.DBLNet(in_channel=out1.shape[1],out_channel=out_channel,inputs=out1)
        out3=self.DBLNet(in_channel=out1.shape[1],out_channel=out_channel,inputs=out2)
        out4=self.DBLNet(in_channel=out1.shape[1],out_channel=out_channel,inputs=out3)
        out5=self.DBLNet(in_channel=out1.shape[1],out_channel=out_channel,inputs=out4)
        return out5
    
    def ResUnit(self,in_channel,out_channel,inputs):
        out1=self.DBLNet(in_channel=in_channel,out_channel=out_channel,inputs=inputs)
        out2=self.DBLNet(in_channel=in_channel,out_channel=out_channel,inputs=out1)
        out3=paddle.add(inputs,out2)
        return out3
    
    def ResBlockBody(self,in_channel,out_channel,inputs,res_block_num=1):
        def ResNUnit(res_block_num,in_channel,out_channel,inputs):
            outs=inputs
            for layer_n in range(res_block_num):
                # weight=layer_n+1
                outs=self.ResUnit(in_channel=in_channel,out_channel=out_channel,inputs=outs)
            return outs
        out1=self.DBLNet(in_channel=in_channel,out_channel=out_channel,inputs=inputs)
        out2=ResNUnit(in_channel=out1.shape[1],out_channel=out1.shape[1],inputs=out1,res_block_num=res_block_num)
        return out2

    def Conv(self,in_channel,out_channel,inputs):
        conv=Conv2D(in_channels=in_channel,out_channels=out_channel,kernel_size=1)
        out=conv(inputs)
        return out

    def Upsampling(self,size_w,size_h,inputs):
        upsamp=UpsamplingBilinear2D(size=(size_w,size_h))
        out=upsamp(inputs)
        return out
    def Maxpool(self,inputs):
        maxpool=MaxPool2D(kernel_size=2,stride=2)
        out=maxpool(inputs)
        return out

    def forward(self,inputs,out_target_num):
        y1_target_num=y2_target_num=y3_target_num=out_target_num//3
        #y1
        #3-->32
        out1_1=self.DBLNet(in_channel=inputs.shape[1],out_channel=32,inputs=inputs)
        
        #32-->64
        out1_2=self.ResBlockBody(in_channel=out1_1.shape[1],out_channel=out1_1.shape[1]*2,inputs=out1_1,res_block_num=1)
        out1_2=self.Maxpool(out1_2)
        #64-->128
        out1_3=self.ResBlockBody(in_channel=out1_2.shape[1],out_channel=out1_2.shape[1]*2,inputs=out1_2,res_block_num=2)
        out1_3=self.Maxpool(out1_3)
        #128-->256
        out1_4=self.ResBlockBody(in_channel=out1_3.shape[1],out_channel=out1_3.shape[1]*2,inputs=out1_3,res_block_num=8)
        out1_4=self.Maxpool(out1_4)
        #y1方向 256-->512
        out1_5=self.ResBlockBody(in_channel=out1_4.shape[1],out_channel=out1_4.shape[1]*2,inputs=out1_4,res_block_num=8)
        out1_5=self.Maxpool(out1_5)
        #512-->1024
        out1_6=self.ResBlockBody(in_channel=out1_5.shape[1],out_channel=out1_5.shape[1]*2,inputs=out1_5,res_block_num=4)
        out1_6=self.Maxpool(out1_6)
        out1_7=self.DBLNet5(in_channel=out1_6.shape[1],out_channel=out1_6.shape[1],inputs=out1_6)
        out1_8=self.DBLNet(in_channel=out1_7.shape[1],out_channel=out1_7.shape[1]//2,inputs=out1_7)
        y1=self.Conv(in_channel=out1_8.shape[1],out_channel=y1_target_num,inputs=out1_8)

        #y2
        out2_7=self.DBLNet(in_channel=out1_7.shape[1],out_channel=out1_7.shape[1]//4,inputs=out1_7)
        out2_8=self.Upsampling(size_w=out2_7.shape[-2]*2,size_h=out2_7.shape[-1]*2,inputs=out2_7)
        #256+512  --->768
        out2_9=paddle.concat(x=[out2_8,out1_5],axis=1)
        out2_10=self.DBLNet5(in_channel=out2_9.shape[1],out_channel=256,inputs=out2_9)
        
        out2_11=self.DBLNet(in_channel=out2_10.shape[1],out_channel=out2_10.shape[1],inputs=out2_10)
        y2=self.Conv(in_channel=out2_11.shape[1],out_channel=y2_target_num,inputs=out2_11)

        #y3
        out3_7=self.DBLNet(in_channel=out2_10.shape[1],out_channel=out2_10.shape[1]//2,inputs=out2_10)
        out3_8=self.Upsampling(size_w=out3_7.shape[-2]*2,size_h=out3_7.shape[-1]*2,inputs=out3_7)
        out3_9=paddle.concat(x=[out3_8,out1_4],axis=1)
        out3_10=self.DBLNet5(in_channel=out3_9.shape[1],out_channel=128,inputs=out3_9)
        out3_11=self.DBLNet(in_channel=out3_10.shape[1],out_channel=out3_10.shape[1],inputs=out3_10)
        y3=self.Conv(in_channel=out3_11.shape[1],out_channel=y3_target_num,inputs=out3_11)

        return y1,y2,y3


input=np.ones(shape=(1,3,416,416)).astype("float32")
# input=np.arange(1,572,0.5)
# input.reshape((1,1,572,572)).astype("float32")
input=paddle.to_tensor(input)

print(input.shape)

model=YOLOv3()
out=model.forward(input,215)
print(out[0].shape,out[1].shape,out[2].shape)

        

