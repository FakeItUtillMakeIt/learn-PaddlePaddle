
from posixpath import relpath
from numpy.core.defchararray import mod
import paddle
from paddle.fluid.incubate.fleet.base import mode
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import gzip

class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        
        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        if mode=='train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode=='valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode=='eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        # img = np.array(self.imgs[idx]).astype('float32')
        # label = np.array(self.labels[idx]).astype('int64')
        img = np.reshape(self.imgs[idx], [1, self.IMG_ROWS, self.IMG_COLS]).astype('float32')
        label = np.reshape(self.labels[idx], [1]).astype('int64')

        return img, label

    def __len__(self):
        return len(self.imgs)
# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
train_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
val_dataset = MnistDataset(mode='valid')
val_loader = paddle.io.DataLoader(val_dataset, batch_size=128,drop_last=True)