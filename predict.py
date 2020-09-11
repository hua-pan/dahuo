# coding:utf-8

import torch
from com.imagine import Imagine


class Predict:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels

    def image(self, image_base64):
        image = Imagine.image_from_base64(image_base64)
        return Imagine.image_size(image), image

    def predict(self, image):
        image = Imagine.denoise(image)
        image = [Imagine.padding_resize(e) for e in Imagine.divide(image)]
        data_x = torch.tensor(image)
        data_x = data_x.reshape(-1, 1, 25, 20) / 255.0
        with torch.no_grad():
            output = self.model(data_x)
            score, index = torch.max(output, dim=1)
            result = ''.join((self.labels[i.item()] for i in index))
            return result, ','.join(('%.2f' % e for e in score / len(self.labels)))
