import os

import paddle
from paddle.fluid.layers.control_flow import Print
import paddlehub as hub

import paddlehub.vision.transforms as T
from paddlehub.finetune.trainer import Trainer
import cv2


CURRENT_DIR=os.path.abspath(".")
PEACH_DIR=os.path.join(CURRENT_DIR,"peach-classification")
TRAIN_LIST_FILE=os.path.join(PEACH_DIR,"train_list.txt")
TEST_LIST_FILE=os.path.join(PEACH_DIR,"test_list.txt")
VALIDATE_LIST_FILE=os.path.join(PEACH_DIR,"validate_list.txt")

class DemoDataset(paddle.io.Dataset):
    def __init__(self, transforms, num_classes=4, mode='train'):	
        # 数据集存放位置
        self.dataset_dir = PEACH_DIR  #dataset_dir为数据集实际路径，需要填写全路径
        self.transforms = transforms
        self.num_classes = num_classes
        self.mode = mode

        if self.mode == 'train':
            self.file = TRAIN_LIST_FILE
        elif self.mode == 'test':
            self.file = TEST_LIST_FILE
        else:
            self.file = VALIDATE_LIST_FILE
        
        self.file = os.path.join(self.dataset_dir , self.file)
        with open(self.file, 'r') as file:
            self.data = file.read().split('\n')[:-1]
        
            
    def __getitem__(self, idx):
        img_path, grt = self.data[idx].split(' ')
        img_path = os.path.join(self.dataset_dir, img_path)
        
        im = self.transforms(img_path)
        return im, int(grt)


    def __len__(self):
        return len(self.data)



transforms = T.Compose(
        [T.Resize((256, 256)),
         T.CenterCrop(224),
         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
        to_rgb=True)

peach_train = DemoDataset(transforms)
peach_validate =  DemoDataset(transforms, mode='val')

model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["R0", "B1", "M2", "S3"])
optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=False) 
trainer.train(peach_train, epochs=10, batch_size=16, eval_dataset=peach_validate, save_interval=1)