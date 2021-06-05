'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, padding_mode='circular'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, padding_mode='circular')
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, padding_mode='circular')
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, padding_mode='circular'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock_NoBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_NoBN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='circular')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, padding_mode='circular'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_NoBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_NoBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, padding_mode='circular')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, padding_mode='circular')
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, padding_mode='circular')

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, padding_mode='circular'),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_NoBN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NoBN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_NoPooling(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NoPooling, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_pool = nn.Sequential(nn.Linear(512*block.expansion*16, 512*block.expansion),
                            nn.ReLU(inplace=True))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear_pool(out)
        out = self.linear(out)
        return out


class ResNet_NoPooling2(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_NoPooling2, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*16, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



# class ResNet_FCPooling(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet_FCPooling, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False, padding_mode='circular')
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         # self.conv2 = nn.Conv2d(512, 512, kernel_size=4,
#         #                        stride=4, bias=False, padding_mode='circular')
#         self.linear_pooling = nn.Linear(4*4, 1)
#         torch.nn.init.xavier_uniform_(self.linear_pooling.weight)
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = out.view(out.size(0), out.size(1), -1)
#         out = self.linear_pooling(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


class ResNet_FCPooling(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_FCPooling, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.conv2 = nn.Conv2d(512, 512, kernel_size=4,
        #                        stride=4, bias=False, padding_mode='circular')
        # self.linear_pooling = nn.Linear(4*4, 1)
        torch.nn.init.xavier_uniform_(self.linear_pooling.weight)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0), out.size(1), -1)
        # out = self.linear_pooling(out)
        out = out[:, :, 0, 0]
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet_MaxPooling(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_MaxPooling, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.max_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(no_batchnorm=False, no_pooling=False, linear_pooling=False, max_pooling=False):
    if no_batchnorm:
        return ResNet_NoBN(BasicBlock_NoBN, [2, 2, 2, 2])
    if no_pooling:
        return ResNet_NoPooling(BasicBlock, [2, 2, 2, 2])
    if linear_pooling:
        return ResNet_FCPooling(BasicBlock, [2, 2, 2, 2])
    elif max_pooling:
        return ResNet_MaxPooling(BasicBlock, [2, 2, 2, 2])
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34(no_batchnorm=False, no_pooling=False, linear_pooling=False, max_pooling=False):
    if no_batchnorm:
        return ResNet_NoBN(BasicBlock_NoBN, [3, 4, 6, 3])
    if no_pooling:
        return ResNet_NoPooling(BasicBlock, [3, 4, 6, 3])
    if linear_pooling:
        return ResNet_FCPooling(BasicBlock, [3, 4, 6, 3])
    elif max_pooling:
        return ResNet_MaxPooling(BasicBlock, [3, 4, 6, 3])
    else:
        return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(no_batchnorm=False, no_pooling=False, linear_pooling=False, max_pooling=False):
    if no_batchnorm:
        return ResNet_NoBN(Bottleneck_NoBN, [3, 4, 6, 3])
    if no_pooling:
        return ResNet_NoPooling(Bottleneck, [3, 4, 6, 3])
    if linear_pooling:
        return ResNet_FCPooling(Bottleneck, [3, 4, 6, 3])
    elif max_pooling:
        return ResNet_MaxPooling(Bottleneck, [3, 4, 6, 3])
    else:
        return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101(no_batchnorm=False, no_pooling=False, linear_pooling=False, max_pooling=False):
    if no_batchnorm:
        return ResNet_NoBN(Bottleneck_NoBN, [3, 4, 23, 3])
    if no_pooling:
        return ResNet_NoPooling(Bottleneck, [3, 4, 23, 3])
    if linear_pooling:
        return ResNet_FCPooling(Bottleneck, [3, 4, 23, 3])
    elif max_pooling:
        return ResNet_MaxPooling(Bottleneck, [3, 4, 23, 3])
    else:
        return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(no_batchnorm=False, no_pooling=False, linear_pooling=False, max_pooling=False):
    if no_batchnorm:
        return ResNet_NoBN(Bottleneck_NoBN, [3, 8, 36, 3])
    if no_pooling:
        return ResNet_NoPooling(Bottleneck, [3, 8, 36, 3])
    if linear_pooling:
        return ResNet_FCPooling(Bottleneck, [3, 8, 36, 3])
    elif max_pooling:
        return ResNet_MaxPooling(Bottleneck, [3, 8, 36, 3])
    else:
        return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18(fc_pooling=True, max_pooling=True)
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
    y_shift = net(X_shift)
    torch.isclose(y_shift, y)
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