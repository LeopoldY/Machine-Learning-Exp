import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image

class SegDataset(Dataset):
    def __init__(self, txt_path="train.txt", root=r'C:/Users/yangc/Developer/Machine-Learning-Exp/Resorce/TrainDataSet', transforms=None, mode='train'):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        imgs = []
        with open(os.path.join(root, txt_path), 'r') as fh:
            for line in fh:
                # 去掉换行符，得到图片路径
                line = line.rstrip()
                words = line.split()
                imgs.append((os.path.join(root, "TrainImg", words[0] + '.jpg'), 
                             os.path.join(root, "Label",  words[0] + '.png'))
                             )
        self.imgs = imgs


    def __getitem__(self, idx):
        fn, label = self.imgs[idx]
        img = cv2.imread(fn)
        if self.transforms is not None:
            img = self.transforms(img)
        label = Image.open(label)
        label = np.array(label).astype(np.int64)
        label = (label[:,:,0] == 128).astype(np.int64) # 将标签数据转换为二分类标签
        label = torch.from_numpy(label)
        return img, label
    
    def __len__(self):
        return len(self.imgs)

def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    random.shuffle(records)
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])

