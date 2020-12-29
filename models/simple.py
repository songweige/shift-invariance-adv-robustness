import torch
import torch.nn as nn

class simple_FC(nn.Module):
    def __init__(self, n_hidden):
        super(simple_FC, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class simple_Conv(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
        	nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
        	nn.ReLU(),
        	nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class simple_Conv_test(nn.Module):
    def __init__(self, n_hidden, padding_size=1, kernel_size=28):
        super(simple_Conv_test, self).__init__()
        # padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        print(out.shape)
        out = self.pool(out)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.classifier(out)
        return out


class simple_Conv_max(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv_max, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class simple_FC_linear(nn.Module):
    def __init__(self, n_hidden):
        super(simple_FC_linear, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, n_hidden),
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class simple_Conv_linear(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv_linear, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



class simple_Conv_linear_max(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv_linear_max, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 10)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def test():
    net = simple_Conv_test(15, padding_size=14, kernel_size=28)
    y = net(torch.randn(1, 1, 28, 28))
    print(y.size())