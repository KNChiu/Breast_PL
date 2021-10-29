import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms

class BreastData(data.Dataset):
    def __init__(self, 
                 data_dir = r'data\histogram_cc', 
                 class_num=2,
                 mode = "train",
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):

        # Set all input args as attributes
        self.__dict__.update(locals())
        if (mode == "train" and not no_augment):
            self.aug = True
        else:
            self.aug = False

        self.check_files()

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, idx):
        img_tensor, labels = self.path_list[idx]
        file_name = "0"

        if self.aug:
            trans = torch.nn.Sequential(
                    transforms.RandomHorizontalFlip(self.aug_prob),
                    transforms.RandomVerticalFlip(self.aug_prob),
                    transforms.RandomRotation(10),
                    
                )  
            img_tensor = trans(img_tensor)
        
        return img_tensor, labels, file_name

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        TRAIN_PATH = op.join(self.data_dir,r'train')
        VAL_PATH = op.join(self.data_dir,r'val')
        TEST_PATH = op.join(self.data_dir,r'test')

        transform1 = transforms.Compose([transforms.Resize((640, 640)),
                                        transforms.ToTensor(),])

        transform2 = transforms.Compose([transforms.Resize((640, 640)),
                                        transforms.ToTensor(),])
        
        train = ImageFolder(TRAIN_PATH, transform1)
        val = ImageFolder(VAL_PATH, transform1)
        test = ImageFolder(TEST_PATH, transform2)

        if self.mode == 'train':
            self.path_list = train 
        elif self.mode == 'val':
            self.path_list = val
        elif self.mode == 'test':
            self.path_list = test