"""
Net & Ensemble models
"""

import numpy as np
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as func


class SimpleNet(nn.Sequential):
    """
    Simple sequential net using ReLU
    """

    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    def __init__(self):
        super(SimpleNet, self).__init__(
            nn.Linear(self.input_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.output_size),
            nn.LogSoftmax(dim=1),
        )
        print(self)

    def adjustImages(self, images):
        # flatten MNIST images into 1D array (784 entries)
        return images.view(images.shape[0], -1)


class ConvNet(nn.Module):
    """
    10-class classificator:
        ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX
    """

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(25)
        self.conv2 = nn.Conv2d(25, 80, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(4 * 4 * 80, 500)
        self.fc_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = func.max_pool2d(self.conv1_bn(self.conv1(x)), 2, 2)
        x = func.relu(x)
        x = func.max_pool2d(self.conv2_bn(self.conv2(x)), 2, 2)
        x = func.relu(x)
        x = x.view(-1, 4 * 4 * 80)
        x = func.relu(self.fc_bn(self.fc1(x)))
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)

    def adjustImages(self, images):
        # images = images.unsqueeze(1).type(torch.cuda.FloatTensor)
        return images.unsqueeze(1).type(torch.FloatTensor)


class BinaryClassifier(nn.Module):
    """
    Binary classificator:
        ConvNet -> Max_Pool -> RELU -> ConvNet -> Max_Pool -> RELU -> FC -> RELU -> FC -> SIGMOID
    """

    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(25)
        self.conv2 = nn.Conv2d(25, 80, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(80)
        self.fc1 = nn.Linear(4 * 4 * 80, 400)
        self.fc1_bn = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 2)

    def forward(self, x):
        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = func.max_pool2d(x, 2, 2)
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 80)
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return func.log_softmax(x, dim=1)
        # return torch.sigmoid(x)

    def adjustImages(self, images):
        # images = images.unsqueeze(1).type(torch.cuda.FloatTensor)
        return images.unsqueeze(1).type(torch.FloatTensor)

    @staticmethod
    def ajustLabels(labels, label):
        """set labels[i] == label to 1, all others to 0"""
        labels = labels.tolist()
        labels = [0 if i != label else 1 for i in labels]
        return np.asarray(labels)


def createBinaryEnsemble():
    """
    Create list of 10 BinaryClassifier
    instances and return the list
    """
    size = 10
    nets = [None] * size
    for i in range(size):
        nets[i] = BinaryClassifier()
    return nets
