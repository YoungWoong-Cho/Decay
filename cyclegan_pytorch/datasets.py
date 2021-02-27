# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture

import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset

class DirectoryNotFoundError(Exception):
    def __init__(self):
        pass

class LoadDataset(Dataset):
    def __init__(self, input_root, transform=None, unaligned=False, mode='train'):
        self.transform = transform
        self.unaligned = unaligned

        try:
            if not os.path.exists(os.path.join(input_root, mode+'A')) or not os.path.exists(os.path.join(input_root, mode+'B')):
                raise DirectoryNotFoundError()
            self.datasetA = sorted(glob.glob(os.path.join(input_root, f'{mode}A') + '/*.*'))
            self.datasetB = sorted(glob.glob(os.path.join(input_root, f'{mode}B') + '/*.*'))
        except DirectoryNotFoundError:
            print('ERROR: Dataset not found.')


    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.datasetA[index % len(self.datasetA)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.datasetB[random.randint(0, len(self.datasetB) - 1)]))
        else:
            item_B = self.transform(Image.open(self.datasetB[index % len(self.datasetB)]))

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.datasetA), len(self.datasetB))