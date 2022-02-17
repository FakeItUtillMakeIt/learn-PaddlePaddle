"""
动态图有诸多优点，比如易用的接口、Python风格的编程体验、友好的调试交互机制等。
在动态图模式下，代码可以按照我们编写的顺序依次执行。这种机制更符合Python程序员的使用习惯，
可以很方便地将脑海中的想法快速地转化为实际代码，也更容易调试。

但在性能方面，由于Python执行开销较大，与C++有一定差距，因此在工业界的许多部署场景中
（如大型推荐系统、移动端）都倾向于直接使用C++进行提速。相比动态图，静态图在部署方面更具有性能的优势。
静态图程序在编译执行时，先搭建模型的神经网络结构，然后再对神经网络执行计算操作。
预先搭建好的神经网络可以脱离Python依赖，在C++端被重新解析执行，而且拥有整体网络结构也能进行一些
网络结构的优化。

那么，有没有可能，深度学习框架实现一个新的模式，同时具备动态图高易用性与静态图高性能的特点呢？
飞桨从2.0版本开始，新增新增支持动静转换功能，编程范式的选择更加灵活。用户依然使用动态图编写代码，
只需添加一行装饰器 @paddle.jit.to_static，即可实现动态图转静态图模式运行，进行模型训练或者推理部署。
在本章节中，将介绍飞桨动态图转静态图的基本用法和相关原理。

动态图转静态图训练
飞桨的动转静方式是基于源代码级别转换的ProgramTranslator实现，其原理是通过分析Python代码，
将动态图代码转写为静态图代码，并在底层自动使用静态图执行器运行。其基本使用方法十分简便，
只需要在要转化的函数（该函数也可以是用户自定义动态图Layer的forward函数）前添加一个装饰器 
@paddle.jit.to_static。这种转换方式使得用户可以灵活使用Python语法及其控制流来构建神经网络模型。
下面通过一个例子说明如何使用飞桨实现动态图转静态图训练。
"""

import paddle

# 定义手写数字识别模型
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=10)

    # 定义网络结构的前向计算过程
    @paddle.jit.to_static  # 添加装饰器，使动态图网络结构在静态图模式下运行
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


import paddle
import paddle.nn.functional as F
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[-1, 1]
def norm_img(img):
    batch_size = img.shape[0]
    # 归一化图像数据
    img = img/127.5 - 1
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, 784])
    
    return img

def train(model):
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), 
                                        batch_size=16, 
                                        shuffle=True)
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('int64')
            
            #前向计算的过程
            predicts = model(images)
            
            # 计算损失
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


model = MNIST() 

train(model)

paddle.save(model.state_dict(), './mnist.pdparams')
print("==>Trained model saved in ./mnist.pdparams")

"""
动转静的训练方式与动态图训练代码是完全相同的。因此，在动转静训练的时候，开发者只需要在动态图的组网
前向计算函数上添加一个装饰器即可实现动转静训练。 在模型构建和训练中，飞桨更希望借用动态图的易用性优势，
实际上，在加上@to_static装饰器运行的时候，飞桨内部是在静态图模式下执行OP的，但是展示给开发者的依然是
动态图的使用方式。

动态图转静态图模型保存
在推理&部署场景中，需要同时保存推理模型的结构和参数，但是动态图是即时执行即时得到结果，
并不会记录模型的结构信息。动态图在保存推理模型时，需要先将动态图模型转换为静态图写法，
编译得到对应的模型结构再保存，而飞桨框架2.0版本推出paddle.jit.save和paddle.jit.load接口，
无需重新实现静态图网络结构，直接实现动态图模型转成静态图模型格式。paddle.jit.save接口会
自动调用飞桨框架2.0推出的动态图转静态图功能，使得用户可以做到使用动态图编程调试，
自动转成静态图训练部署。

当用户使用paddle.jit.save保存Layer对象时，飞桨会自动将用户编写的动态图Layer模型转换为静态图写法，
并编译得到模型结构，同时将模型结构与参数保存。paddle.jit.save需要适配飞桨沿用已久的推理模型与参数格式，
做到前向完全兼容，因此其保存格式与paddle.save有所区别，具体包括三种文件：保存模型结构的*.pdmodel文件；
保存推理用参数的*.pdiparams文件和保存兼容变量信息的*.pdiparams.info文件，
这几个文件后缀均为paddle.jit.save保存时默认使用的文件后缀。

比如，如果保存上述手写字符识别的inference模型用于部署，可以直接用下面代码实现：
"""
# save inference model
from paddle.static import InputSpec
# 加载训练好的模型参数
state_dict = paddle.load("./mnist.pdparams")
# 将训练好的参数读取到网络中
model.set_state_dict(state_dict)
# 设置模型为评估模式
model.eval()

# 保存inference模型
paddle.jit.save(
    layer=model,
    path="inference/mnist",
    input_spec=[InputSpec(shape=[None, 784], dtype='float32')])

print("==>Inference model saved in inference/mnist.")

"""
其中，paddle.jit.save API 将输入的网络存储为 paddle.jit.TranslatedLayer 格式的模型，
载入后可用于预测推理或者微调(fine-tune)训练。 该接口会将输入网络转写后的模型结构 Program 
和所有必要的持久参数变量存储至输入路径 path 。

path 是存储目标的前缀，存储的模型结构 Program 文件的后缀为 .pdmodel ，
存储的持久参数变量文件的后缀为 .pdiparams ，同时这里也会将一些变量描述信息存储至文件，
文件后缀为 .pdiparams.info。

通过调用对应的paddle.jit.load接口，可以把存储的模型载入为 paddle.jit.TranslatedLayer格式，
用于预测推理或者fine-tune训练。
"""

import numpy as np
import paddle
import paddle.nn.functional as F
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 读取mnist测试数据，获取第一个数据
mnist_test = paddle.vision.datasets.MNIST(mode='test')
test_image, label = mnist_test[0]
# 获取读取到的图像的数字标签
print("The label of readed image is : ", label)

# 将测试图像数据转换为tensor，并reshape为[1, 784]
test_image = paddle.reshape(paddle.to_tensor(test_image), [1, 784])
# 然后执行图像归一化
test_image = norm_img(test_image)
# 加载保存的模型
loaded_model = paddle.jit.load("./inference/mnist")
# 利用加载的模型执行预测
preds = loaded_model(test_image)
pred_label = paddle.argmax(preds)
# 打印预测结果
print("The predicted label is : ", pred_label.numpy())

"""
paddle.jit.save API 可以把输入的网络结构和参数固化到一个文件中，
所以通过加载保存的模型，可以不用重新构建网络结构而直接用于预测，易于模型部署。

总结：
本章节中，介绍了飞桨动转静的功能和基本原理，并以一个例子介绍了如何将一个动态图模型转换到静态图模式下
训练，并将训练好的模型转换成更易于部署的inference模型，有关更多动转静的功能介绍，可以参考官方文档。
https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/index_cn.html
"""

