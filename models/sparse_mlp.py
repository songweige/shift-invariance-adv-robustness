import torch
from torch import nn
import torch.nn.functional as F

class SmallFCN(nn.Module):
    def __init__(self, alpha, input_shape=(32, 32, 3), num_classes=10):
        super(SmallFCN, self).__init__()
        self.alpha = alpha
        self.image_s = input_shape[0]
        self.in_channels = input_shape[-1]
        self.input_size = input_shape[0] * input_shape[1] * input_shape[2]
        assert input_shape[0] == input_shape[1], "Currently equal sized images only supported"
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_size, self.image_s * self.image_s * self.alpha // 4)
        self.bn1 = nn.BatchNorm1d(self.image_s * self.image_s * self.alpha // 4)
        self.fc2 = nn.Linear(self.image_s * self.image_s * self.alpha // 4, 24 * self.alpha)
        self.bn2 = nn.BatchNorm1d(24 * self.alpha)
        self.linear = nn.Linear(self.alpha * 24, self.num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return self.linear(out)