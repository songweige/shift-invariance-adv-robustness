'''Train CIFAR10 with PyTorch.'''
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

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--n_data', default=50000, type=int, help='level of verbos')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_path', type=str, help='path for the model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.n_data<50000:
    trainSet = torchvision.datasets.CIFAR10(
        root='./CIFAR/', train=True, download=True, transform=transform_test)
    trainSet = torch.utils.data.Subset(trainSet, indices=np.arange(args.n_data))
    trainloader = torch.utils.data.DataLoader(
        trainSet, batch_size=100, shuffle=False, num_workers=2)
    
testset = torchvision.datasets.CIFAR10(
    root='./CIFAR/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


# Model
net = get_net(args.model)
net = net.to(device)
net.eval()
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benchmark = True


print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

if args.n_data < 50000:
    checkpoint = torch.load('./ckpts/adv_pool/cifar10/%s_%d.pth'%(args.model, args.n_data))
    print("Accuracy on clean train examples: {:.2f}%".format(checkpoint['train_acc']))
else:
    checkpoint = torch.load(os.path.join(args.model_path, args.model+'.pth'))
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print("Accuracy on clean test examples: {:.2f}%".format(best_acc))


consistency = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(testloader):
    shift0 = np.random.randint(32,size=2)
    inputs_shift_v0 = torch.zeros([100, 3, 32, 32])
    inputs_shift_hv0 = torch.zeros([100, 3, 32, 32])
    inputs_shift_v0[:, :, :, :shift0[0]] = inputs[:, :, :, (32-shift0[0]):].clone()
    inputs_shift_v0[:, :, :, shift0[0]:] = inputs[:, :, :, :(32-shift0[0])].clone()
    inputs_shift_hv0[:, :, :shift0[1], :] = inputs_shift_v0[:, :, (32-shift0[1]):, :]
    inputs_shift_hv0[:, :, shift0[1]:, :] = inputs_shift_v0[:, :, :(32-shift0[1]), :]
    predicted0 = net(inputs_shift_hv0.cuda()).argmax(1)
    shift1 = np.random.randint(32,size=2)
    inputs_shift_v1 = torch.zeros([100, 3, 32, 32])
    inputs_shift_hv1 = torch.zeros([100, 3, 32, 32])
    inputs_shift_v1[:, :, :, :shift1[0]] = inputs[:, :, :, (32-shift1[0]):].clone()
    inputs_shift_v1[:, :, :, shift1[0]:] = inputs[:, :, :, :(32-shift1[0])].clone()
    inputs_shift_hv1[:, :, :shift1[1], :] = inputs_shift_v1[:, :, (32-shift1[1]):, :]
    inputs_shift_hv1[:, :, shift1[1]:, :] = inputs_shift_v1[:, :, :(32-shift1[1]), :]
    predicted1 = net(inputs_shift_hv1.cuda()).argmax(1)
    consistency += (predicted0==predicted1).sum().item()
    total += targets.size(0)
    # import ipdb;ipdb.set_trace()


print("Consistency on clean test examples: {:.2f}%".format(100.*consistency/total))
