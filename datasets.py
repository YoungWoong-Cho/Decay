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
    def __init__(self, data_root, image_size):
        self.image_size = image_size
        self.transform = transforms.Compose([
                           transforms.Resize(int(image_size), Image.BICUBIC),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        try:
            if not os.path.exists(os.path.join(data_root, 'A')) or not os.path.exists(os.path.join(data_root, 'B')):
                raise DirectoryNotFoundError()
            self.dataset_A = sorted(glob.glob(os.path.join(data_root, 'A') + '/*.*'))
            self.dataset_B = sorted(glob.glob(os.path.join(data_root, 'B') + '/*.*'))
        except DirectoryNotFoundError:
            print('ERROR: Dataset not found.')

    # Returns the 2 datasets. 
    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.dataset_A[index % len(self.dataset_A)]).resize((self.image_size, self.image_size)))
        item_B = self.transform(Image.open(self.dataset_B[index % len(self.dataset_B)]).resize((self.image_size, self.image_size)))
        return {"A": item_A, "B": item_B}

    # Finds and returns the length of the dataset with the larger size
    def __len__(self):
        maximum_length = max(len(self.dataset_A), len(self.dataset_B)) 
        return maximum_length