# coding:utf-8

import torch
from torch.utils.data import Dataset


class DahuoDataset(Dataset):
    def __init__(self, images, targets, shape=(-1, 1, 25, 20)):  # tensor 4D shape:[batch_size, channel, height, width]
        self.images, self.targets, self.shape = images, targets, shape
        self.data_x = torch.tensor(self.images) / 255.0
        self.data_x = self.data_x.reshape(*self.shape)
        self.data_y = torch.tensor(self.targets).long()

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def split(self, ratio=0.8):
        train_count = round(len(self) * ratio)
        train_set = self.__class__(self.images[:train_count], self.targets[:train_count], self.shape)
        test_set = self.__class__(self.images[train_count:], self.targets[train_count:], self.shape)
        return train_set, test_set
