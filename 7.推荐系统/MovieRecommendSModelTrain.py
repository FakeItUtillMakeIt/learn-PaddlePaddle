from enum import EnumMeta
from paddle.hapi import model

from paddle.nn.functional.pooling import avg_pool1d
from MovieRecommendSModel import *
from math import sqrt



def train(model):
    #超参数
    lr=0.001
    Epochs=10
    paddle.set_device("cpu")

    model.train()
    #数据迭代器
    data_loader=model.train_loader
    #优化器
    opt=paddle.optimizer.Adam(learning_rate=lr,parameters=model.parameters())

    for epoch in range(Epochs):
        for idx,data in enumerate(data_loader()):
            #获取数据
            usr,mov,score=data
            usr_v=[paddle.to_tensor(var) for var in usr]
            mov_v=[paddle.to_tensor(var) for var in mov]
            score_label=paddle.to_tensor(score)
            #
            _,_,score_predicts=model.forward(usr_v,mov_v)
            loss=F.square_error_cost(score_predicts,score_label)
            avg_loss=paddle.mean(loss)

            if idx%500==0:
                print("epoch:{},batch_id:{},loss:{}".format(epoch,idx,avg_loss.numpy()))

            #往梯度下降方向更新权重参数
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        paddle.save(model.state_dict(),"./checkpoint/epoch+"+str(epoch)+".pdparams")


def evaluation(model, params_file_path):
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    Epochs=10
    for epoch in range(Epochs): 
        acc_set = []
        avg_loss_set = []
        squaredError=[]
        for idx, data in enumerate(model.valid_loader()):
            usr, mov, score_label = data
            usr_v = [paddle.to_tensor(var) for var in usr]
            mov_v = [paddle.to_tensor(var) for var in mov]

            _, _, scores_predict = model(usr_v, mov_v)

            pred_scores = scores_predict.numpy()
            
            avg_loss_set.append(np.mean(np.abs(pred_scores - score_label)))
            squaredError.extend(np.abs(pred_scores - score_label)**2)

            diff = np.abs(pred_scores - score_label)
            diff[diff>0.5] = 1
            acc = 1 - np.mean(diff)
            acc_set.append(acc)
        RMSE=sqrt(np.sum(squaredError) / len(squaredError))
        # print("RMSE = ", sqrt(np.sum(squaredError) / len(squaredError)))#均方根误差RMSE
        acc,mae,RMSE= np.mean(acc_set), np.mean(avg_loss_set),RMSE
        print("acc:{},mae:{},RMSE:{}".format(acc,mae,RMSE))

model=Model(use_poster=False,use_mov_title=True,use_mov_cat=True,use_age_job=True,fc_sizes=[128,64,32])
# train(model)
model_params_path="./checkpoint/epoch+9.pdparams"
# evaluation(model,params_file_path=model_params_path)