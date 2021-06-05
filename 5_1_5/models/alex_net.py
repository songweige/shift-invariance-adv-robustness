import torch.nn as nn

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNet_MLP2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, n_hidden=4096):
        super(AlexNet_MLP2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNet_MLP(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, n_hidden=4096):
        super(AlexNet_MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x


class AlexNet_linear(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_linear, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, num_classes)
            # nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        # x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x



class AlexNet_NoPooling(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet_NoPooling, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='circular'),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x



def test():
    net = AlexNet()
    net.eval()
    # Randomly generate a cifar10-like sample
    X = torch.randn(1, 3, 32, 32)
    # First shift i pixels right then i pixels down
    X_shift_right = torch.zeros([32, 3, 32, 32])
    X_shift = torch.zeros([32, 3, 32, 32])
    for i in range(1, 32):
        X_shift_right[i, :, :, :i] = X[0, :, :, (32-i):]
        X_shift_right[i, :, :, i:] = X[0, :, :, :(32-i)]
        X_shift[i, :, :i, :] = X_shift_right[i, :, (32-i):, :]
        X_shift[i, :, i:, :] = X_shift_right[i, :, :(32-i), :]


    # compare the logits
    y = net(X)
    features = net.features(X)
    y_shift = net(X_shift)
    features_shift = net.features(X_shift)
    torch.isclose(y_shift, y)
    torch.isclose(features_shift, features).all(1)
    print(y.size())
    # First shift 8, 16, 24 pixels right then 8, 16, 24 pixels down
    X_shift_right = torch.zeros([32, 3, 32, 32])
    X_shift = torch.zeros([32, 3, 32, 32])
    shifts = [8, 16, 24]
    for i in range(3):
        for j in range(3):
            idx = i + j*3
            X_shift_right[idx, :, :, :shifts[i]] = X[0, :, :, (32-shifts[i]):]
            X_shift_right[idx, :, :, shifts[i]:] = X[0, :, :, :(32-shifts[i])]
            X_shift[idx, :, :shifts[j], :] = X_shift_right[idx, :, (32-shifts[j]):, :]
            X_shift[idx, :, shifts[j]:, :] = X_shift_right[idx, :, :(32-shifts[j]), :]
    # compare the logits
    y = net(X)
    y_shift = net(X_shift)
    torch.isclose(y_shift, y)