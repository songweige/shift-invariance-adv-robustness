# Fully connected neural network
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, num_classes, no_final=False):
        super(NeuralNet, self).__init__()
        self.in_normalize = nn.BatchNorm1d(input_size)
        self.num_hidden_layers = hidden_layers
        self.input_size = input_size
        self.hidden_linear = nn.ModuleList()
        self.hidden_act = nn.ModuleList()
        for idx in range(hidden_layers):
            if idx == 0:
                self.hidden_linear.append(nn.Linear(input_size, hidden_size))
            else:
                self.hidden_linear.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_act.append(nn.ReLU())

        self.final = nn.Linear(hidden_size, num_classes)
        self.no_final = no_final
        self.final_act = nn.Tanh()

    def forward(self, x):
        out = x.reshape(-1, self.input_size)
        out = self.in_normalize(out)
        for (linear, act) in zip(self.hidden_linear, self.hidden_act):
            out = linear(out)
            out = act(out)
        out = self.final(out)
        if not self.no_final:
            out = self.final_act(out)
        return out

class CNN_OneD(nn.Module):
    def __init__(self, hidden_channels=1024, kernel_size=5, num_classes=1, padding=-1, no_final=False):
        super(CNN_OneD, self).__init__()
        self.hidden_channels = hidden_channels
        if padding==-1:
            padding = kernel_size // 2
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size, padding=padding, padding_mode='circular')
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.no_final = no_final
        self.final = nn.Linear(hidden_channels, num_classes)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, self.hidden_channels * 1)
        x = self.final(x)
        if not self.no_final:
            x = self.final_act(x)
        return x


class CNN(nn.Module):
    def __init__(self, hidden_channels=1024, kernel_size=5, num_classes=1, no_final=False):
        super(CNN, self).__init__()
        self.hidden_channels = hidden_channels
        self.in_normalize = nn.BatchNorm2d(1)
        padding_size = kernel_size // 2
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=kernel_size,
                               padding=padding_size, padding_mode='circular')
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.no_final = no_final
        self.final = nn.Linear(hidden_channels, num_classes)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = self.in_normalize(x)
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, self.hidden_channels * 1)
        x = self.final(x)
        if not self.no_final:
            x = self.final_act(x)
        return x.squeeze()

class CNN_FC(nn.Module):
    def __init__(self, hidden_channels=1024, kernel_size=5, num_classes=1, no_final=False):
        super(CNN_FC, self).__init__()
        self.hidden_channels = hidden_channels
        self.in_normalize = nn.BatchNorm2d(1)
        padding_size = kernel_size // 2
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size,
                               padding=padding_size, padding_mode='circular')
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.no_final = no_final
        self.linear = nn.Linear(hidden_channels, 100)
        self.final = nn.Linear(100, num_classes)
        self.final_act = nn.Tanh()

    def forward(self, x):
        x = self.in_normalize(x)
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, self.hidden_channels * 1)
        x = self.linear(x)
        x = self.final(x)
        if not self.no_final:
            x = self.final_act(x)
        return x.squeeze()