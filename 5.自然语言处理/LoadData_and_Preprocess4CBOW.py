#####################################################
# https://www.paddlepaddle.org.cn/tutorials/projectdetail/2201246
'''
@说明        :word Embedding  词向量
@时间        :2021/12/27 15:31:37
@作者        :lijin
@版本        :1.0
'''
#先把每个词转换为一个高维空间的向量，这些向量一定意义上可代表词的词义，通过计算这些词向量之间的距离，
# 就可以得到词语之间的关联关系，从而达到让计算机像计算数值一样去计算自然语言的目的

#因此，词向量模型需要解决两个问题：
#1.如何把词转换为向量：词向量查询表  
#  词向量查询表----->张量  
#  以张量形式代替查询表中的非张量格式

#2.如何让词向量具有语义：
#  使用词的上下文了解单词词义   word2vec算法也是通过上下文学习语义信息（包含两个经典模型：CBOW Skip-gram）
#  CBOW通过上下文推理中心词
#  Skip-gram 通过中心词推理上下文


import io
import os
import random
from re import sub
import sys
from paddle.fluid.layers.nn import embedding
import requests
from collections import defaultdict,OrderedDict
import math
import numpy as np
import paddle
from paddle.nn import Embedding
import paddle.nn.functional as F


# 下载语料用来训练word2vec
def download():
    # 可以从百度云服务器下载一些开源数据集（dataset.bj.bcebos.com）
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    # 使用python的requests包下载数据集到本地
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    # 把下载后的文件存储在当前目录的text8.txt文件内
    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()



#将语料加载到内存
def load_text8(CURRENT_DIR):
    textname="text8.txt"
    filename=os.path.join(CURRENT_DIR,textname)
    with open(filename,"r") as f:
        corups=f.read().strip("\n")
    f.close()
    return corups

#对语料进行处理（分词）
def data_preprocess(corups):
    #把英文单词大写转为小写
    corups=corups.strip().lower()
    corups=corups.split(" ")
    return corups

#构造词典，统计词频，根据词频将每个词转换为一个整数ID
def build_dict(corups):
    #统计词频
    word_freq_dict=dict()
    for word in corups:
        if word not in word_freq_dict:
            word_freq_dict[word]=0
        word_freq_dict[word]+=1
    
    #将词典中的词根据频率进行由高到低排序
    word_freq_dict=sorted(word_freq_dict.items(),key=lambda x:x[1],reverse=True)

    #构造三个不同词典
    #每个词到id的映射关系 wrod2id_dict
    #每个id出现的频率 word2id_freq
    #每个id到词的映射关系 id2word_dict
    word2id_dict=dict()
    word2id_freq=dict()
    id2word_dict=dict()
    #
    for word,freq in word_freq_dict:
        curr_id=len(word2id_dict)
        word2id_dict[word]=curr_id
        word2id_freq[word2id_dict[word]]=freq
        id2word_dict[curr_id]=word
    return word2id_dict,word2id_freq,id2word_dict


#把语料替换为id序列
def convert_corpus_to_id(corups,word2id_dict):
    corups=[word2id_dict[word] for word in corups]
    return corups

#二次采样法处理语料，强化训练效果
def subsampling(corups,word2id_freq):
    #discard函数决定一个词会不会被替换，函数具有随机性，每次调用结果不同，如果一个词的频率很大，
    #它被遗弃的概率也很大
    def discard(word_id):
        return random.uniform(0,1)<1-math.sqrt(1e-4/word2id_freq[word_id]*len(corups))
    corups=[word for word in corups if not discard(word)]
    return corups

#在完成语料数据预处理之后，需要构造训练数据。根据上面的描述，我们需要使用一个滑动窗口对语料从左到右扫描，
# 在每个窗口内，中心词需要预测它的上下文，并形成训练数据。
# 在实际操作中，由于词表往往很大（50000，100000等），对大词表的一些矩阵运算（如softmax）需要消耗巨大的资源，
# 因此可以通过负采样的方式模拟softmax的结果。
# 给定一个中心词和一个需要预测的上下文词，把这个上下文词作为正样本。
# 通过词表随机采样的方式，选择若干个负样本。
# 把一个大规模分类问题转化为一个2分类问题，通过这种方式优化计算速度。

# 构造数据，准备模型训练
# max_window_size代表了最大的window_size的大小，程序会根据max_window_size从左到右扫描整个语料
# negative_sample_num代表了对于每个正样本，我们需要随机采样多少负样本用于训练，
# 一般来说，negative_sample_num的值越大，训练效果越稳定，但是训练速度越慢。 
def build_data(corpus, word2id_dict, word2id_freq, CBOW_window_size = 3, negative_sample_num = 4):
    
    # 使用一个list存储处理好的数据
    dataset = []
    vocab_size=len(corpus)
    # 从左到右，开始枚举每个中心点的位置
    for start_idx in range(len(corpus)):
        context_range=start_idx,start_idx+2*CBOW_window_size
        center_word_idx=start_idx+CBOW_window_size
        if context_range[1]<len(corpus):
            context_words=corpus[context_range[0]:context_range[1]+1]
            center_word=corpus[center_word_idx]
            del context_words[CBOW_window_size]
            # context_words=[word2id_dict[context_word] for context_word in context_words]
            # center_word=word2id_dict[center_word]
            dataset.append((context_words,center_word,1))

            i=0
            while i<negative_sample_num:
                negative_word_idx=random.randint(0,vocab_size-1)
                if negative_word_idx!=center_word_idx:
                    negative_word=corpus[negative_word_idx]
                    # negative_word=word2id_dict[negative_word]
                    dataset.append((context_words,negative_word,0))
                    i+=1
        
    return dataset


# 构造mini-batch，准备对模型进行训练
# 我们将不同类型的数据放到不同的tensor里，便于神经网络进行处理
# 并通过numpy的array函数，构造出不同的tensor来，并把这些tensor送入神经网络中进行训练
def build_batch(dataset, batch_size, epoch_num):
    
    # context_word_batch缓存batch_size个中心词
    context_word_batch = []
    # center_word_batch缓存batch_size个目标词（可以是正样本或者负样本）
    center_word_batch = []
    # label_batch缓存了batch_size个0或1的标签，用于模型训练
    label_batch = []

    for epoch in range(epoch_num):
        # 每次开启一个新epoch之前，都对数据进行一次随机打乱，提高训练效果
        random.shuffle(dataset)
        
        for context_word, center_word, label in dataset:
            # 遍历dataset中的每个样本，并将这些数据送到不同的tensor里
            context_word_batch.append([context_word])
            center_word_batch.append([center_word])
            label_batch.append(label)

            # 当样本积攒到一个batch_size后，我们把数据都返回回来
            # 在这里我们使用numpy的array函数把list封装成tensor
            # 并使用python的迭代器机制，将数据yield出来
            # 使用迭代器的好处是可以节省内存
            if len(context_word_batch) == batch_size:
                yield np.array(context_word_batch).astype("int64"), \
                    np.array(center_word_batch).astype("int64"), \
                    np.array(label_batch).astype("float32")
                context_word_batch = []
                center_word_batch = []
                label_batch = []

    if len(context_word_batch) > 0:
        yield np.array(context_word_batch).astype("int64"), \
            np.array(center_word_batch).astype("int64"), \
            np.array(label_batch).astype("float32")