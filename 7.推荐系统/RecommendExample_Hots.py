from enum import EnumMeta
from logging import raiseExceptions
import pickle

from paddle.fluid.layers.nn import cos_sim
from paddle.tensor.search import topk
from MovieRecommendSModel import *

import argparse

# 定义特征保存函数
def get_usr_mov_features(model, params_file_path, poster_path):
    paddle.set_device('cpu') 
    usr_pkl = {}
    mov_pkl = {}
    
    # 定义将list中每个元素转成tensor的函数
    def list2tensor(inputs, shape):
        inputs = np.reshape(np.array(inputs).astype(np.int64), shape)
        return paddle.to_tensor(inputs)

    # 加载模型参数到模型中，设置为验证模式eval（）
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()
    # 获得整个数据集的数据
    dataset = model.Dataset.dataset

    for i in range(len(dataset)):
        # 获得用户数据，电影数据，评分数据  
        # 本案例只转换所有在样本中出现过的user和movie，实际中可以使用业务系统中的全量数据
        usr_info, mov_info, score = dataset[i]['usr_info'], dataset[i]['mov_info'],dataset[i]['scores']
        usrid = str(usr_info['usr_id'])
        movid = str(mov_info['mov_id'])

        # 获得用户数据，计算得到用户特征，保存在usr_pkl字典中
        if usrid not in usr_pkl.keys():
            usr_id_v = list2tensor(usr_info['usr_id'], [1])
            usr_age_v = list2tensor(usr_info['age'], [1])
            usr_gender_v = list2tensor(usr_info['gender'], [1])
            usr_job_v = list2tensor(usr_info['job'], [1])

            usr_in = [usr_id_v, usr_gender_v, usr_age_v, usr_job_v]
            usr_feat = model.get_usr_feat(usr_in)

            usr_pkl[usrid] = usr_feat.numpy()
        
        # 获得电影数据，计算得到电影特征，保存在mov_pkl字典中
        if movid not in mov_pkl.keys():
            mov_id_v = list2tensor(mov_info['mov_id'], [1])
            mov_tit_v = list2tensor(mov_info['title'], [1, 1, 15])
            mov_cat_v = list2tensor(mov_info['category'], [1, 6])

            mov_in = [mov_id_v, mov_cat_v, mov_tit_v, None]
            mov_feat = model.get_mov_feat(mov_in)

            mov_pkl[movid] = mov_feat.numpy()
    


    print(len(mov_pkl.keys()))
    # 保存用户特征和电影到本地  方便离线进行用户推荐和根据推荐特征进行推荐
    pickle.dump(usr_pkl, open('./usr_feat.pkl', 'wb'))
    pickle.dump(mov_pkl, open('./mov_feat.pkl', 'wb'))
    print("usr / mov features saved!!!")

# 定义根据用户兴趣推荐电影
def recommend_mov_for_usr(usr_id, top_k, pick_num, usr_feat_dir, mov_feat_dir, mov_info_path):
    assert pick_num <= top_k
    # 读取电影和用户的特征
    usr_feats = pickle.load(open(usr_feat_dir, 'rb'))
    mov_feats = pickle.load(open(mov_feat_dir, 'rb'))
    usr_feat = usr_feats[str(usr_id)]

    cos_sims = []

    # with dygraph.guard():
    paddle.disable_static()
    # 索引电影特征，计算和输入用户ID的特征的相似度
    for idx, key in enumerate(mov_feats.keys()):
        mov_feat = mov_feats[key]
        usr_feat = paddle.to_tensor(usr_feat)
        mov_feat = paddle.to_tensor(mov_feat)
        # 计算余弦相似度
        sim = paddle.nn.functional.common.cosine_similarity(usr_feat, mov_feat)
        
        cos_sims.append(sim.numpy()[0])
    # 对相似度排序
    index = np.argsort(cos_sims)[-top_k:]

    mov_info = {}
    
    # 读取电影文件里的数据，根据电影ID索引到电影信息
    with open(mov_info_path, 'r', encoding="ISO-8859-1") as f:
        data = f.readlines()
        for item in data:
            item = item.strip().split("::")
            mov_info[str(item[0])] = item

    hots_moives=dict()
    for i in range(1,10):
        hots_moives[str(i)]=mov_info[str(i)]

    print("当前的用户是：")
    print("usr_id:", usr_id)
    print("推荐可能喜欢的电影是：")

    res = []
    
    # 加入随机选择因素，确保每次推荐的都不一样
    while len(res) < pick_num:
        val = np.random.choice(len(index), 1)[0]
        idx = index[val]
        mov_id = list(mov_feats.keys())[idx]
        if mov_id not in res:
            res.append(mov_id)
        

    rest_pick=top_k-pick_num
    for i in range(rest_pick):
        if str(i+1) not in res:
            res.append(hots_moives[str(i+1)][0])

    for id in res:
        print("mov_id:", id, mov_info[str(id)])


# model=Model(use_poster=False,use_mov_title=True,use_mov_cat=True,use_age_job=True,fc_sizes=[128,64,32])
# param_path = "./checkpoint/epoch+9.pdparams"
# poster_path = POSTER_DIR
# get_usr_mov_features(model, param_path, poster_path)  


if __name__=="__main__":
    #根据用户喜好推荐电影
    mov_feat_path="./mov_feat.pkl"
    usr_feat_path="./usr_feat.pkl"

    parser=argparse.ArgumentParser(description="根据用户喜好推荐电影")
    parser.add_argument("--usr_id",nargs="?",help="用户ID")
    parser.add_argument("--topk",nargs="?",help="推荐个数")
    parser.add_argument("--pick_num",nargs="?",help="选中个数")

    res=parser.parse_args()
    

    try:
        usr_id=int(res.usr_id)
        usr_topk=int(res.topk)
        pick_num=int(res.pick_num)
    except:
        usr_id=10
        usr_topk=10
        pick_num=5

    if usr_id==None:
        usr_id=10
        topk=10
        pick_num=10
        
    else:
        if topk==None:
            topk=5
        if pick_num==None:
            pick_num=topk
    recommend_mov_for_usr(usr_id=usr_id,top_k=usr_topk,pick_num=pick_num,usr_feat_dir=usr_feat_path,
        mov_feat_dir=mov_feat_path,mov_info_path=MOVIE_FILE)
    

    

