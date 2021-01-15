from .vgg import *
from .resnet import *
from .resnet_mnist import *
from .wide_resnet import *
from .simple import *
from .alex_net import *
from .GoogleNet import *

resnet_dict = {'18':ResNet18, '34':ResNet34, '50':ResNet50, '101':ResNet101, '152':ResNet152}

def get_net(model):
    if model.startswith('VGG'):
        return VGG(model)
    elif model.startswith('alexnet'):
        if 'nopooling' in model:
            return AlexNet_NoPooling()
        else:
            return AlexNet()
    elif model.startswith('resnet'):
        n_layer = model.split('_')[-1]
        if 'nobatchnorm' in model:
            return resnet_dict[n_layer](no_batchnorm=True, no_pooling=False, linear_pooling=False, max_pooling=False)
        elif 'nopooling' in model:
            return resnet_dict[n_layer](no_pooling=True, linear_pooling=False, max_pooling=False)
        elif 'linear_pooling' in model:
            return resnet_dict[n_layer](no_pooling=False, linear_pooling=True, max_pooling=False)
        elif 'max_pooling' in model:
            return resnet_dict[n_layer](no_pooling=False, linear_pooling=False, max_pooling=True)
        else:
            return resnet_dict[n_layer](no_pooling=False, linear_pooling=False, max_pooling=False)
    elif model.startswith('wide_resnet'):
        n_layer = int(model.split('_')[-2])
        widen_factor = int(model.split('_')[-1])
        if 'max_pooling' in model:
            return Wide_ResNet_MaxPool(depth=n_layer, widen_factor=widen_factor)
        else:
            return Wide_ResNet(depth=n_layer, widen_factor=widen_factor)
