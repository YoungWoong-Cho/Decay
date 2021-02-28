# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture

import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Raises an error if a dataset cannot be located
class DirectoryNotFoundError(Exception):
    def __init__(self):
        pass

# Loads in the 2 datasets
class LoadDataset(Dataset):
    def __init__(self, data_root, img_size):
        self.img_size = img_size
        self.transform = transforms.Compose([
                           transforms.Resize(int(img_size), Image.BICUBIC),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        try:
            if not os.path.exists(os.path.join(data_root, 'A')) or not os.path.exists(os.path.join(data_root, 'B')):
                raise DirectoryNotFoundError()
            self.datasetA = sorted(glob.glob(os.path.join(data_root, 'A') + '/*.*'))
            self.datasetB = sorted(glob.glob(os.path.join(data_root, 'B') + '/*.*'))
        except DirectoryNotFoundError:
            print('ERROR: Dataset not found.')

    # Returns the 2 datasets. 
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.datasetA[index % len(self.datasetA)]).resize((self.img_size, self.img_size)))
        item_B = self.transform(Image.open(self.datasetB[index % len(self.datasetB)]).resize((self.img_size, self.img_size)))
        return {"A": item_A, "B": item_B}

    # Finds and returns the length of the dataset with the larger size
    def __len__(self):
        maximum_length = max(len(self.dataset_A), len(self.dataset_B)) 
        return maximum_length