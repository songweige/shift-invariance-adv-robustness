import torch
import torchvision

### ResNet & Wide-ResNet & ResNetXT 7x7
net = torchvision.models.__dict__['resnet152']()
net = torchvision.models.__dict__['wide_resnet101_2']()
net = torchvision.models.__dict__['resnext50_32x4d']()

x = torch.randn(1,3,224,224)
x = net.conv1(x)
x = net.bn1(x)
x = net.relu(x)
x = net.maxpool(x)

x = net.layer1(x)
x = net.layer2(x)
x = net.layer3(x)
x = net.layer4(x)

print(x.shape)


### AlexNet 6x6
net = torchvision.models.__dict__['alexnet']()

x = torch.randn(1,3,224,224)
x = net.features(x)

print(x.shape)

### VGG 7x7
net = torchvision.models.__dict__['vgg11']()

x = torch.randn(1,3,224,224)
x = net.features(x)

print(x.shape)


### densenet121 7x7
net = torchvision.models.__dict__['densenet201']()

x = torch.randn(1,3,224,224)
x = net.features(x)

print(x.shape)

### Inception V3 8x8
### GoogleNet 7x7



### mobilenet_v2 7x7
net = torchvision.models.__dict__['mobilenet_v2']()

x = torch.randn(1,3,224,224)
x = net.features(x)

print(x.shape)

