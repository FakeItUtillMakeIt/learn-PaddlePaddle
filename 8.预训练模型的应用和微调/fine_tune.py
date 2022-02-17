from cv2 import transform
import paddlehub as hub

from paddlehub.finetune.trainer import Trainer
import paddle
from load_data import *
import paddlehub.vision.transforms as T


# model=hub.Module(name="resnet50_vd_imagenet_ssld",label_list=["R0","B1","M2","S3"])

# dataLoader=DatasetLoader1()
# peach_train = dataLoader.load_dataset(mode="train")
# peach_validate = dataLoader.load_dataset(mode="validate")

# print(peach_train[0].shape)

# optimizer=paddle.optimizer.Adam(learning_rate=0.001,parameters=model.parameters())
# trainer=Trainer(model,optimizer,checkpoint_dir="img_classification_ckpt")
# trainer.train(peach_train,epochs=10,batch_size=16,eval_dataset=peach_validate,save_interval=1)

img_path=os.path.join(PEACH_DIR,'test/R0/0.png')
im=cv2.imread(img_path).astype('float32')
print(im)
