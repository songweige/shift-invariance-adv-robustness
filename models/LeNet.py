import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = func.relu(self.conv1(x))
        print(x.shape)
        x = func.max_pool2d(x, 2)
        print(x.shape)
        x = func.relu(self.conv2(x))
        print(x.shape)
        x = func.max_pool2d(x, 2)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x




def test():
    net = LeNet()
    net.eval()
    # Randomly generate a cifar10-like sample
    X = torch.randn(1, 3, 32, 32)
    # First shift i pixels right then i pixels down
    y = net(X)
    X_shift_right = torch.zeros([32, 3, 32, 32])
    X_shift = torch.zeros([32, 3, 32, 32])
    for i in range(1, 32):
        X_shift_right[i, :, :, :i] = X[0, :, :, (32-i):]
        X_shift_right[i, :, :, i:] = X[0, :, :, :(32-i)]
        X_shift[i, :, :i, :] = X_shift_right[i, :, (32-i):, :]
        X_shift[i, :, i:, :] = X_shift_right[i, :, :(32-i), :]


    # compare the logits
    y_shift = net(X_shift)
    torch.isclose(y_shift, y)
    print(y.size())