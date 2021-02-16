'''Train MNIST with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np

from models import *
from models.simple import *
from utils import progress_bar

from DDN import DDN

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_hidden', default=1000, type=int, help='number of hidden units')
parser.add_argument('--kernel_size', default=28, type=int, help='the size of convolutional kernel')
parser.add_argument('--epoch', default=200, type=int, help='which epoch to load')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(
    root='/vulcanscratch/songweig/datasets/mnist', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='/vulcanscratch/songweig/datasets/mnist', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

resnet_dict = {'18':ResNet_MNIST_18, '34':ResNet_MNIST_34, '50':ResNet_MNIST_50, '101':ResNet_MNIST_101, '152':ResNet_MNIST_152}

# Model
if args.model == 'FC':
    net = simple_FC(args.n_hidden)
elif args.model == 'FC_linear':
    net = simple_FC_linear(args.n_hidden)
elif args.model == 'Conv':
    net = simple_Conv(args.n_hidden, args.kernel_size)
elif args.model == 'Conv_max':
    net = simple_Conv_max(args.n_hidden, args.kernel_size)
elif args.model == 'Conv_linear':
    net = simple_Conv_linear(args.n_hidden, args.kernel_size)
elif args.model == 'Conv_linear_max':
    net = simple_Conv_linear_max(args.n_hidden, args.kernel_size)
elif args.model == 'Conv_linear_nopooling':
    net = simple_Conv_linear_nopooling(args.n_hidden, args.kernel_size)
elif args.model == 'Conv_linear_pooling':
    net = simple_Conv_linear_pooling(args.n_hidden, args.kernel_size)
elif args.model.startswith('resnet'):
    n_layer = args.model.split('_')[-1]
    if 'no_pooling' in args.model:
        net = resnet_dict[n_layer](pooling=False)
    elif 'max_pooling' in args.model:
        net = resnet_dict[n_layer](pooling=True, max_pooling=True)
    else:
        net = resnet_dict[n_layer](pooling=True, max_pooling=False)


print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)


if args.epoch == 200:
    if 'Conv' in args.model and args.kernel_size != 28:
        checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/mnist/simple_%s_%d_%d.pth'%(args.model, args.kernel_size, args.n_hidden))
    else:
        checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/mnist/simple_%s_%d.pth'%(args.model, args.n_hidden))
else:
    checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/mnist_gd/simple_%s_%d_%d.pth'%(args.model, args.n_hidden, args.epoch))

    
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']


classifier = PyTorchClassifier(
    model=net,
    loss=criterion,
    optimizer=optimizer,
    clip_values=(0., 1.),
    input_shape=(1, 28, 28),
    nb_classes=10,
)
print("Accuracy on clean test examples: {:.2f}%".format(best_acc))

attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
# attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=epsilon, norm=norm)
adv_correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    grad = classifier.loss_gradient(x=inputs, y=targets)
    break
    total += targets.size(0)