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
parser.add_argument('--padding_size', default=0, type=int, help='the size of circular padding')
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
trainset = torch.utils.data.Subset(trainset, indices=np.arange(4000))
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
    net = simple_Conv(args.n_hidden, args.kernel_size, args.padding_size)
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

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.95)
# optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      # momentum=0.9, weight_decay=5e-4)


checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/double_descent_4k/simple_%s_%d.pth'%(args.model, args.n_hidden))
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

train_loss = 0.
test_loss = 0.
total = 0.
correct = 0.
for batch_idx, (inputs, targets) in enumerate(trainloader):
    targets = torch.nn.functional.one_hot(targets, num_classes=10).float()
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets.argmax(1)).sum().item()
    # correct += predicted.eq(targets).sum().item()
print('Training Loss: %.3f | Acc: %.3f%%' % (train_loss/(batch_idx+1), 100.*correct/total))



total = 0.
correct = 0.
for batch_idx, (inputs, targets) in enumerate(testloader):
    targets = torch.nn.functional.one_hot(targets, num_classes=10).float()
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets.argmax(1)).sum().item()
    # correct += predicted.eq(targets).sum().item()
print('Test Loss: %.3f | Acc: %.3f%%' % (train_loss/(batch_idx+1), 100.*correct/total))



criterion = nn.CrossEntropyLoss()

classifier = PyTorchClassifier(
    model=net,
    loss=criterion,
    optimizer=optimizer,
    clip_values=(0., 1.),
    input_shape=(1, 28, 28),
    nb_classes=10,
)
print("Accuracy on clean test examples: {:.2f}%".format(best_acc))

attack_params = [[2, [0.5, 1, 1.5, 2, 2.5]], [np.inf, [0.05, 0.1, 0.15, 0.2, 0.25]]]
for norm, epsilons in attack_params:
    for epsilon in epsilons:
        attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
        # attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=epsilon, norm=norm)
        adv_correct = 0
        adv_loss = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_adv = attack.generate(x=inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct_idx = predicted.eq(targets)
            # print(inputs.shape, inputs_adv.shape)
            outputs_adv = net(torch.FloatTensor(inputs_adv).cuda())
            loss = criterion(outputs_adv, targets.cuda())
            adv_loss += loss.item()
            adv_predicted = classifier.predict(inputs_adv).argmax(1)
            # import ipdb;ipdb.set_trace()
            adv_correct += (adv_predicted==targets.cpu().data.numpy())[correct_idx.cpu().data.numpy()].sum().item()
            total += correct_idx.sum().item()
        print("Accuracy on adversarial test examples (L_{:.0f}, eps={:.2f}): {:.2f}%. Loss: {:.2f}".format(norm, epsilon, 100.*adv_correct/total, adv_loss/(batch_idx+1)))