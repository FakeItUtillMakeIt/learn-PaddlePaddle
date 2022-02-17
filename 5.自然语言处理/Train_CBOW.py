from LoadData_and_Preprocess4CBOW import *

corups=load_text8(CURRENT_DIR=os.path.abspath("."))
corups=data_preprocess(corups)
word2id_dict,word2id_freq,id2word_dict=build_dict(corups)

vocab_size=len(word2id_freq)



#定义 CBOW网络结构
#输入：上下文形状  上下文词数*vocab_size  参数形状vocab_size*embedding_size
#输出：1*vocab_size
class CBOW(paddle.nn.Layer):
    def __init__(self,vocab_size,embedding_size,init_scale=0.1) -> None:
        super(CBOW,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        #构造词向量参数 [vocab_size,embedding_size]
        self.embedding=Embedding(num_embeddings=self.vocab_size,
                        embedding_dim=self.embedding_size,
                        weight_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Uniform(low=-init_scale,
                            high=init_scale)))
        self.embedding_out=Embedding(num_embeddings=self.vocab_size,
                        embedding_dim=self.embedding_size,
                        weight_attr=paddle.ParamAttr(
                            initializer=paddle.nn.initializer.Uniform(low=-init_scale,
                            high=init_scale)))

    #前向计算逻辑
    #中心词应换成上下文
    def forward(self,surround_words,target_words,labels):
        #将输入转换为词向量
        surround_word_emb=self.embedding(surround_words)
        surround_word_emb=paddle.sum(surround_word_emb,axis=-2)
        target_words_emb=self.embedding_out(target_words)

        #通过点乘方式计算上下文词到目标词的输出概率，通过sigmoid估计这个词是正样本还是负样本
        word_sim=paddle.multiply(surround_word_emb,target_words_emb)
        word_sim=paddle.sum(word_sim,axis=-1)
        # word_sim=F.softmax(word_sim,axis=-1)
        word_sim=paddle.reshape(word_sim,shape=[-1])
        pred=F.sigmoid(word_sim)

        loss=F.binary_cross_entropy_with_logits(word_sim,labels)
        loss=paddle.mean(loss)
        return pred,loss
        



corpus=convert_corpus_to_id(corups,word2id_dict)
# corpus=subsampling(corups,word2id_freq)

corpus_light = corpus[:int(len(corpus)*0.2)]
dataset = build_data(corpus_light, word2id_dict, word2id_freq)


# 开始训练，定义一些训练过程中需要使用的超参数
batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 0.001

#定义一个使用word-embedding查询同义词的函数
#这个函数query_token是要查询的词，k表示要返回多少个最相似的词，embed是我们学习到的word-embedding参数
#我们通过计算不同词之间的cosine距离，来衡量词和词的相似度
#具体实现如下，x代表要查询词的Embedding，Embedding参数矩阵W代表所有词的Embedding
#两者计算Cos得出所有词对查询词的相似度得分向量，排序取top_k放入indices列表
def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:
        print('for word %s, the similar word is %s' % (query_token, str(id2word_dict[i])))

# 将模型放到GPU上训练
try:
    paddle.set_device('gpu:0')
except:
    paddle.set_device('cpu')

# 通过我们定义的CBOW类，来构造一个CBOW模型网络
# 253854*200
CBOW_model = CBOW(vocab_size, embedding_size)

# 构造训练这个网络的优化器
adam = paddle.optimizer.Adam(learning_rate=learning_rate, parameters = CBOW_model.parameters())

# 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
for context_words, target_words, label in build_batch(
    dataset, batch_size, epoch_num):
    # 使用paddle.to_tensor，将一个numpy的tensor，转换为飞桨可计算的tensor
    context_words_var = paddle.to_tensor(context_words)
    center_words_var = paddle.to_tensor(target_words)
    label_var = paddle.to_tensor(label)
    
    # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
    pred, loss = CBOW_model.forward(
        context_words_var, center_words_var, label_var)

    # 程序自动完成反向计算
    loss.backward()
    # 程序根据loss，完成一步对参数的优化更新
    adam.step()
    # 清空模型中的梯度，以便于下一个mini-batch进行更新
    adam.clear_grad()

    # 每经过1000个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
    step += 1
    if step % 1000 == 0:
        print("step %d, loss %.3f" % (step, loss.numpy()[0]))

    # 每隔10000步，打印一次模型对以下查询词的相似词，这里我们使用词和词之间的向量点积作为衡量相似度的方法，只打印了5个最相似的词
    if step % 10000 ==0:
        get_similar_tokens('movie', 5, CBOW_model.embedding.weight)
        get_similar_tokens('one', 5, CBOW_model.embedding.weight)
        get_similar_tokens('chip', 5, CBOW_model.embedding.weight)
        

