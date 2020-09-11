# coding:utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import multiprocessing
from com.model import ConvNet
from com.dataset import DahuoDataset
from com.imagine import Imagine

# torch.set_printoptions(precision=None, threshold=None, edgeitems=None, linewidth=400, profile='full')

path = 'C:/Users/ab072804/Desktop/classes/'
epochs = 15
batch_size = 8
cpu_count = multiprocessing.cpu_count()
ratio = 0.8

if __name__ == '__main__':
    images, targets = Imagine.image_from_file(path)
    dataset = DahuoDataset(images, targets)
    train_set, test_set = dataset.split(ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=cpu_count, drop_last=False)

    model = ConvNet()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        sum_loss = 0.0
        for index, (data_x, data_y) in enumerate(train_loader):
            output = model(data_x)
            loss = loss_func(output, data_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            # print('Training: %d' % index)
        print('Epoch: %2d/%2d | Loss Rate: %.5f' % (epoch + 1, epochs, sum_loss / len(train_loader)))
        # output to predict test_load
    torch.save(model.state_dict(), 'dahuo.pkl')
