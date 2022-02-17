import os
import numpy as np
from PIL import Image
import cv2
import paddle
import paddlehub as hub

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
            self.file = 'train_list.txt'
        elif self.mode == 'test':
            self.file = 'test_list.txt'
        else:
            self.file = 'validate_list.txt'
        
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

class DatasetLoader():
    def __init__(self) -> None:
        pass

    def normalize_arr(self,arr):
        mean=arr.mean()
        deviation=arr.std()
        std_arr=(arr-mean)/deviation
        return std_arr

    def load_dataset(self,mode="train",batch_size=128):
        if mode=="train":
            with open(TRAIN_LIST_FILE,"r") as f:
                all_line=f.readlines()
                train_data_list=[]
                train_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    train_data_list.append(img_arr)
                    train_label_list.append(label)
                    if len(train_data_list)==batch_size:
                        print("输出")
                        yield np.array(train_data_list),np.array(train_label_list)
                        train_data_list,train_label_list=[],[]
                if len(train_data_list)>0:
                    yield np.array(train_data_list),np.array(train_label_list)
        elif mode=="test":
            with open(TEST_LIST_FILE,"r") as f:
                all_line=f.readlines()
                train_data_list=[]
                train_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    label=int(label)
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    train_data_list.append(img_arr)
                    train_label_list.append(label)
                    if len(train_data_list)==batch_size:
                        print("输出")
                        yield np.array(train_data_list),np.array(train_label_list)
                        train_data_list,train_label_list=[],[]
                if len(train_data_list)>0:
                    yield np.array(train_data_list),np.array(train_label_list)
        elif mode=="validate":
            with open(VALIDATE_LIST_FILE,"r") as f:
                all_line=f.readlines()
                train_data_list=[]
                train_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    label=int(label)
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    train_data_list.append(img_arr)
                    train_label_list.append(label)
                    if len(train_data_list)==batch_size:
                        print("输出")
                        yield np.array(train_data_list),np.array(train_label_list)
                        train_data_list,train_label_list=[],[]
                if len(train_data_list)>0:
                    yield np.array(train_data_list),np.array(train_label_list)
        return self.load_dataset

class DatasetLoader1():
    def __init__(self) -> None:
        pass

    def normalize_arr(self,arr):
        mean=arr.mean()
        deviation=arr.std()
        std_arr=(arr-mean)/deviation
        return std_arr.resize(224,224)

    def load_dataset(self,mode="train",batch_size=128):
        if mode=="train":
            with open(TRAIN_LIST_FILE,"r") as f:
                all_line=f.readlines()
                train_data_list=[]
                train_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    train_data_list.append(img_arr)
                    train_label_list.append(label)
                    
                return np.array(train_data_list),np.array(train_label_list)
        elif mode=="test":
            with open(TEST_LIST_FILE,"r") as f:
                all_line=f.readlines()
                test_data_list=[]
                test_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    label=int(label)
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    test_data_list.append(img_arr)
                    test_label_list.append(label)
                    
                return np.array(test_data_list),np.array(test_label_list)
        elif mode=="validate":
            with open(VALIDATE_LIST_FILE,"r") as f:
                all_line=f.readlines()
                valid_data_list=[]
                valid_label_list=[]

                for line in all_line:
                    line,label=line.split(' ')
                    label=int(label)
                    img_path=os.path.join(PEACH_DIR,line)
                    img_arr=np.array(Image.open(img_path))
                    img_arr=self.normalize_arr(img_arr)
                    valid_data_list.append(img_arr)
                    valid_label_list.append(label)
                return np.array(valid_data_list),np.array(valid_label_list)
     



