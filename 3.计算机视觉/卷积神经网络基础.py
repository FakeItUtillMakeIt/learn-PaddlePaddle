
#计算机视觉模型
#https://www.paddlepaddle.org.cn/tutorials/projectdetail/2250773#anchor-2
#卷积神经网络
#涉及：卷积、池化、激活函数、批归一化、丢弃法等

#两个典型任务：图像分类  目标检测
#图像分类典型模型结构：LeNet、AlexNet、VGG
#目标检测典型模型结构：YOLOv3

#在计算机时间子任务图鉴中，包含了以下四类任务
#1.图像分类
#2.目标检测
#3.语义分割
#4.实例分割

#为何要使用卷积：
#1.一维向量会丢失数据的空间信息
#2.模型参数过多，容易产生过拟合
#为了解决上述问题，引入卷积神经网络进行特征提取，既能提取到相邻像素点之间的特征模式，又能保证参数的个数不随图片尺寸变化
#在卷积神经网络中，计算范围是在像素点的空间邻域内进行的，卷积核参数的数目也远小于全连接层。
# 卷积核本身与输入图片大小无关，它代表了对空间邻域内某种特征模式的提取。比如，有些卷积核提取
# 物体边缘特征，有些卷积核提取物体拐角处的特征，图像上不同区域共享同一个卷积核。当输入图片
# 大小不一样时，仍然可以使用同一个卷积核进行操作。

#感受野
#3×3卷积对应的感受野大小就是3×3，当通过两层3×3的卷积之后，感受野的大小将会增加到5×5
#因此，当增加卷积网络深度的同时，感受野将会增大，输出特征图中的一个像素点将会包含更多的图像语义信息。

#多输入通道、多输出通道和批量操作
#前面介绍的卷积计算过程比较简单，实际应用时，处理的问题要复杂的多。
# 例如：对于彩色图片有RGB三个通道，需要处理多输入通道的场景。
# 输出特征图往往也会具有多个通道，而且在神经网络的计算中常常是把一个批次的样本放在一起计算，
# 所以卷积算子需要具有批量处理多输入和多输出通道数据的功能，下面将分别介绍这几种场景的操作方式。

#多输入通道：
#多输出通道：通常将卷积核的输出通道数叫做卷积核的个数。
#批量操作：在卷积神经网络的计算中，通常将多个样本放在一起形成一个mini-batch进行批量操作，即输入数据的维度

#飞桨API  paddle.nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding)
#in_channles:输入数据通道数
#out_channels：卷积核个数

#卷积核  ：（卷积核个数，卷积核通道数，H,W）

#简单的图像黑白边界检测
import matplotlib.pyplot as plt
import numpy as np
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
#%matplotlib inline

# 创建初始化权重参数w
w = np.array([1, 0, -1], dtype='float32')
# 将权重参数调整成维度为[cout, cin, kh, kw]的四维张量
w = w.reshape([1, 1, 1, 3])
# 创建卷积算子，设置输出通道数，卷积核大小，和初始化权重参数
# kernel_size = [1, 3]表示kh = 1, kw=3
# 创建卷积算子的时候，通过参数属性weight_attr指定参数初始化方式
# 这里的初始化方式时，从numpy.ndarray初始化卷积参数
conv = Conv2D(in_channels=1, out_channels=1, kernel_size=[1, 3],
       weight_attr=paddle.ParamAttr(
          initializer=Assign(value=w)))

# 创建输入图片，图片左边的像素点取值为1，右边的像素点取值为0
img = np.ones([50,50], dtype='float32')
img[:, 30:] = 0.
# 将图片形状调整为[N, C, H, W]的形式
x = img.reshape([1,1,50,50])
# 将numpy.ndarray转化成paddle中的tensor
x = paddle.to_tensor(x)
# 使用卷积算子作用在输入图片上
y = conv(x)
# 将输出tensor转化为numpy.ndarray
out = y.numpy()
print(out.shape)
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img, cmap='gray')
f = plt.subplot(122)
f.set_title('output featuremap', fontsize=15)
# 卷积算子Conv2D输出数据形状为[N, C, H, W]形式
# 此处N, C=1，输出数据形状为[1, 1, H, W]，是4维数组
# 但是画图函数plt.imshow画灰度图时，只接受2维数组
# 通过numpy.squeeze函数将大小为1的维度消除
plt.imshow(out.squeeze(), cmap='gray')
plt.show()
print(conv.weight,conv.bias)

#检测图片中的边缘
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import paddle
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
img = Image.open('./1.png')

# 设置卷积核参数
w = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]], dtype='float32')/8
w = w.reshape([1, 1, 3, 3])
# 由于输入通道数是3，将卷积核的形状从[1,1,3,3]调整为[1,3,3,3]
w = np.repeat(w, 3, axis=1)
# 创建卷积算子，输出通道数为1，卷积核大小为3x3，
# 并使用上面的设置好的数值作为卷积核权重的初始化参数
conv = Conv2D(in_channels=3, out_channels=1, kernel_size=[3, 3], 
            weight_attr=paddle.ParamAttr(
              initializer=Assign(value=w)))
    
# 将读入的图片转化为float32类型的numpy.ndarray
x = np.array(img).astype('float32')
# 图片读入成ndarry时，形状是[H, W, 3]，
# 将通道这一维度调整到最前面
x = np.transpose(x, (2,0,1))
# 将数据形状调整为[N, C, H, W]格式
x = x.reshape(1, 3, img.height, img.width)
x = paddle.to_tensor(x)
y = conv(x)
out = y.numpy()
plt.figure(figsize=(20, 10))
f = plt.subplot(121)
f.set_title('input image', fontsize=15)
plt.imshow(img)
f = plt.subplot(122)
f.set_title('output feature map', fontsize=15)
plt.imshow(out.squeeze(), cmap='gray')
plt.show()

print(conv.weight,conv.bias)


#图像均值模糊
import paddle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from paddle.nn import Conv2D
from paddle.nn.initializer import Assign
# 读入图片并转成numpy.ndarray
# 换成灰度图
img = Image.open('./1.png').convert('L')
img = np.array(img)

# 创建初始化参数
w = np.ones([1, 1, 5, 5], dtype = 'float32')/25
conv = Conv2D(in_channels=1, out_channels=1, kernel_size=[5, 5], 
        weight_attr=paddle.ParamAttr(
         initializer=Assign(value=w)))
x = img.astype('float32')
x = x.reshape(1,1,img.shape[0], img.shape[1])
x = paddle.to_tensor(x)
y = conv(x)
out = y.numpy()

plt.figure(figsize=(20, 12))
f = plt.subplot(121)
f.set_title('input image')
plt.imshow(img, cmap='gray')

f = plt.subplot(122)
f.set_title('output feature map')
out = out.squeeze()
plt.imshow(out, cmap='gray')

plt.show()





#池化
#池化是使用某一位置的相邻输出的总体统计特征代替网络在该位置的输出，其好处是当输入数据做出少量平移时，
# 经过池化函数后的大多数输出还能保持不变。比如：当识别一张图像是否是人脸时，我们需要知道人脸左边有
# 一只眼睛，右边也有一只眼睛，而不需要知道眼睛的精确位置，这时候通过池化某一片区域的像素点来得到
# 总体统计特征会显得很有用。由于池化之后特征图会变得更小，如果后面连接的是全连接层，能有效的减小神经元
# 的个数，节省存储空间并提高计算效率。
#池化分为平均池化和最大池化



#ReLU激活函数
#前面介绍的网络结构中，普遍使用Sigmoid函数做激活函数。在神经网络发展的早期，Sigmoid函数用的比较多，
# 而目前用的较多的激活函数是ReLU。这是因为Sigmoid函数在反向传播过程中，容易造成梯度的衰减
#下面画出了sigmoid和relu函数曲线图
# ReLU和Sigmoid激活函数示意图
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.figure(figsize=(10, 5))

# 创建数据x
x = np.arange(-10, 10, 0.1)

# 计算Sigmoid函数
s = 1.0 / (1 + np.exp(0. - x))

# 计算ReLU函数
y = np.clip(x, a_min=0., a_max=None)

#####################################
# 以下部分为画图代码
f = plt.subplot(121)
plt.plot(x, s, color='r')
currentAxis=plt.gca()
plt.text(-9.0, 0.9, r'$y=Sigmoid(x)$', fontsize=13)
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)

f = plt.subplot(122)
plt.plot(x, y, color='g')
plt.text(-3.0, 9, r'$y=ReLU(x)$', fontsize=13)
currentAxis=plt.gca()
currentAxis.xaxis.set_label_text('x', fontsize=15)
currentAxis.yaxis.set_label_text('y', fontsize=15)

plt.show()
#梯度消失现象
# 在神经网络里，将经过反向传播之后，梯度值衰减到接近于零的现象称作梯度消失现象。
# 从上面的函数曲线可以看出，当xxx为较大的正数的时候，Sigmoid函数数值非常接近于1，
# 函数曲线变得很平滑，在这些区域Sigmoid函数的导数接近于零。当xxx为较小的负数时，
# Sigmoid函数值也非常接近于0，函数曲线也很平滑，在这些区域Sigmoid函数的导数也接近于0。
# 只有当xxx的取值在0附近时，Sigmoid函数的导数才比较大。


#批归一化
#批归一化方法（Batch Normalization，BatchNorm）是由Ioffe和Szegedy于2015年提出的，
# 已被广泛应用在深度学习中，其目的是对神经网络中间层的输出进行标准化处理，使得中间层的输出更加稳定。
# 通常我们会对神经网络的数据进行标准化处理，处理后的样本数据集满足均值为0，方差为1的统计分布，
# 这是因为当输入数据的分布比较固定时，有利于算法的稳定和收敛。对于深度神经网络来说，
# 由于参数是不断更新的，即使输入数据已经做过标准化处理，但是对于比较靠后的那些层，
# 其接收到的输入仍然是剧烈变化的，通常会导致数值不稳定，模型很难收敛。
# BatchNorm能够使神经网络中间层的输出变得更加稳定，并有如下三个优点：
# 使学习快速进行（能够使用较大的学习率）
# 降低模型对初始值的敏感性
# 从一定程度上抑制过拟合
# BatchNorm主要思路是在训练时以mini-batch为单位，对神经元的数值进行归一化，使数据的分布满足均值为0，方差为1。
#如果强行限制输出层的分布是标准化的，可能会导致某些特征模式的丢失，所以在标准化之后，BatchNorm会紧接着对数据做缩放和平移。
# 其中γ\gammaγ和β\betaβ是可学习的参数，可以赋初始值γ=1,β=0\gamma = 1, \beta = 0γ=1,β=0，在训练过程中不断学习调整。
#上面列出的是BatchNorm方法的计算逻辑，下面针对两种类型的输入数据格式分别进行举例。飞桨支持输入数据的维度大小为2、3、4、5
# 四种情况，这里给出的是维度大小为2和4的示例。

# 输入数据形状是 [N, K]时的示例
import numpy as np
import paddle
from paddle.nn import BatchNorm1D
# 创建数据
data = np.array([[1,2,3], [4,5,6], [7,8,9]]).astype('float32')
# 使用BatchNorm1D计算归一化的输出
# 输入数据维度[N, K]，num_features等于K
bn = BatchNorm1D(num_features=3)    
x = paddle.to_tensor(data)
y = bn(x)
print('output of BatchNorm1D Layer: \n {}'.format(y.numpy()))

# 使用Numpy计算均值、方差和归一化的输出
# 这里对第0个特征进行验证
a = np.array([1,4,7])
a_mean = a.mean()
a_std = a.std()
b = (a - a_mean) / a_std
print('std {}, mean {}, \n output {}'.format(a_mean, a_std, b))


# 输入数据形状是[N, C, H, W]时的batchnorm示例
import numpy as np
import paddle
from paddle.nn import BatchNorm2D

# 设置随机数种子，这样可以保证每次运行结果一致
np.random.seed(100)
# 创建数据
data = np.random.rand(2,3,3,3).astype('float32')
# 使用BatchNorm2D计算归一化的输出
# 输入数据维度[N, C, H, W]，num_features等于C
bn = BatchNorm2D(num_features=3)
x = paddle.to_tensor(data)
y = bn(x)
print('input of BatchNorm2D Layer: \n {}'.format(x.numpy()))
print('output of BatchNorm2D Layer: \n {}'.format(y.numpy()))

# 取出data中第0通道的数据，
# 使用numpy计算均值、方差及归一化的输出
a = data[:, 0, :, :]
a_mean = a.mean()
a_std = a.std()
b = (a - a_mean) / a_std
print('channel 0 of input data: \n {}'.format(a))
print('std {}, mean {}, \n output: \n {}'.format(a_mean, a_std, b))

# 提示：这里通过numpy计算出来的输出
# 与BatchNorm2D算子的结果略有差别，
# 因为在BatchNorm2D算子为了保证数值的稳定性，
# 在分母里面加上了一个比较小的浮点数epsilon=1e-05


# 预测时使用BatchNorm
# 上面介绍了在训练过程中使用BatchNorm对一批样本进行归一化的方法，但如果使用同样的方法对需要预测的
# 一批样本进行归一化，则预测结果会出现不确定性。
# 例如样本A、样本B作为一批样本计算均值和方差，与样本A、样本C和样本D作为一批样本计算均值和方差，
# 得到的结果一般来说是不同的。那么样本A的预测结果就会变得不确定，这对预测过程来说是不合理的。
# 解决方法是在训练过程中将大量样本的均值和方差保存下来，预测时直接使用保存好的值而不再重新计算。
# 实际上，在BatchNorm的具体实现中，训练时会计算均值和方差的移动平均值。


#丢弃法 dropout
#丢弃法（Dropout）是深度学习中一种常用的抑制过拟合的方法，其做法是在神经网络学习过程中，随机删除一部分神经元。
# 训练时，随机选出一部分神经元，将其输出设置为0，这些神经元将不对外传递信号。
#在预测场景时，会向前传递所有神经元的信号，可能会引出一个新的问题：训练时由于部分神经元被随机丢弃了，
# 输出数据的总大小会变小。比如：计算其L1范数会比不使用Dropout时变小，但是预测时却没有丢弃神经元，
# 这将导致训练和预测时数据的分布不一样。为了解决这个问题，飞桨支持如下两种方法
#downscale_in_infer
# 训练时以比例r随机丢弃一部分神经元，不向后传递它们的信号；预测时向后传递所有神经元的信号，但是将每个神经元上的数值乘以 (1−r)。
# upscale_in_train
# 训练时以比例r随机丢弃一部分神经元，不向后传递它们的信号，但是将那些被保留的神经元上的数值除以 (1−r)；预测时向后传递所有神经元的信号，不做任何处理。
# dropout操作 下面这段程序展示了经过Dropout之后输出数据的形式。
import paddle
import numpy as np

# 设置随机数种子，这样可以保证每次运行结果一致
np.random.seed(100)
# 创建数据[N, C, H, W]，一般对应卷积层的输出
data1 = np.random.rand(2,3,3,3).astype('float32')
# 创建数据[N, K]，一般对应全连接层的输出
data2 = np.arange(1,13).reshape([-1, 3]).astype('float32')
# 使用dropout作用在输入数据上
x1 = paddle.to_tensor(data1)
# downgrade_in_infer模式下
drop11 = paddle.nn.Dropout(p = 0.5, mode = 'downscale_in_infer')
droped_train11 = drop11(x1)
# 切换到eval模式。在动态图模式下，使用eval（）切换到求值模式，该模式禁用了dropout。
drop11.eval()
droped_eval11 = drop11(x1)
# upscale_in_train模式下
drop12 = paddle.nn.Dropout(p = 0.5, mode = 'upscale_in_train')
droped_train12 = drop12(x1)
# 切换到eval模式
drop12.eval()
droped_eval12 = drop12(x1)

x2 = paddle.to_tensor(data2)
drop21 = paddle.nn.Dropout(p = 0.5, mode = 'downscale_in_infer')
droped_train21 = drop21(x2)
# 切换到eval模式
drop21.eval()
droped_eval21 = drop21(x2)
drop22 = paddle.nn.Dropout(p = 0.5, mode = 'upscale_in_train')
droped_train22 = drop22(x2)
# 切换到eval模式
drop22.eval()
droped_eval22 = drop22(x2)
    
print('x1 {}, \n droped_train11 \n {}, \n droped_eval11 \n {}'.format(data1, droped_train11.numpy(),  droped_eval11.numpy()))
print('x1 {}, \n droped_train12 \n {}, \n droped_eval12 \n {}'.format(data1, droped_train12.numpy(),  droped_eval12.numpy()))
print('x2 {}, \n droped_train21 \n {}, \n droped_eval21 \n {}'.format(data2, droped_train21.numpy(),  droped_eval21.numpy()))
print('x2 {}, \n droped_train22 \n {}, \n droped_eval22 \n {}'.format(data2, droped_train22.numpy(),  droped_eval22.numpy()))

# 从上述代码的输出可以发现，经过dropout之后，tensor中的某些元素变为了0，这个就是dropout实现的功能，通过随机将输入数据的元素置0，
# 消除减弱了神经元节点间的联合适应性，增强模型的泛化能力。










