#Pixel Level Classification
import torch
from torch import nn
import torch.nn.functional as F

class DeepSpectra(nn.Module):
    def __init__(self, num_classes):
        super(DeepSpectra, self).__init__()
        self.num_classes = num_classes
        self.pre = pre_block(sampling_point = 200)
        self.conv1 = conv_block(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding = 1
        )
        self.conv2 = nn.Conv1d(8, 16, 3, 1, 1)

        # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
        self.inception1 = Naive_inception_block(16, 8, 8, 8, 8)
        #self.inception2 = Naive_inception_block(32, 16, 16, 16, 16)
        self.fc1 = nn.Linear(50*32, 128)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128, self.num_classes)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.pre(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.inception1(x)
        #x = self.inception2(x)
        x = x.view(-1, 50*32)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x)


class Naive_inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, out_3x3, out_5x5, out_1x1pool):
        super(Naive_inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = conv_block(in_channels, out_3x3, kernel_size=3, padding=1)

        self.branch3 = conv_block(in_channels, out_5x5, kernel_size=5, padding=2)
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )
        
    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )
        
    
class Inception_block(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    ):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1
        )

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.conv(x))
    
class pre_block(nn.Module):
    def __init__(self, sampling_point):
        super().__init__()
        self.pool1 = nn.AvgPool1d(kernel_size = 5, stride = 1, padding = 2)
        self.pool2 = nn.AvgPool1d(kernel_size = 13, stride = 1, padding = 6)
        self.pool3 = nn.AvgPool1d(kernel_size = 7, stride = 1, padding = 3)
        self.ln = nn.LayerNorm(sampling_point)
        
    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.ln(x)
        
        return x