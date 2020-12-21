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

from models.simple import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_hidden', default=1000, type=int, help='number of hidden units')
parser.add_argument('--kernel_size', default=28, type=int, help='the size of convolutional kernel')
parser.add_argument('--verbos', default=0, type=int, help='level of verbos')
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


# Model
if args.model == 'FC':
    net = simple_FC(args.n_hidden)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
elif args.model == 'FC_linear':
    net = simple_FC_linear(args.n_hidden)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
elif args.model == 'Conv':
    net = simple_Conv(args.n_hidden, args.kernel_size)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
elif args.model == 'Conv_max':
    net = simple_Conv_max(args.n_hidden, args.kernel_size)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
elif args.model == 'Conv_linear':
    net = simple_Conv_linear(args.n_hidden, args.kernel_size)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
elif args.model == 'Conv_linear_max':
    net = simple_Conv_linear_max(args.n_hidden, args.kernel_size)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()



net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('/vulcanscratch/songweig/ckpts/adv_pool'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/%s.pth'%args.model)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if args.verbos > 1:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, net, model_name):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if args.verbos > 1:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    # if acc > best_acc:
    if epoch == start_epoch+199:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/vulcanscratch/songweig/ckpts/adv_pool'):
            os.mkdir('/vulcanscratch/songweig/ckpts/adv_pool')
        torch.save(state, '/vulcanscratch/songweig/ckpts/adv_pool/%s.pth'%model_name)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch, net)
    if 'Conv' in args.model and args.kernel_size != 28:
        test(epoch, net, 'simple_%s_%d_%d'%(args.model, args.kernel_size, args.n_hidden))
    else:
        test(epoch, net, 'simple_%s_%d'%(args.model, args.n_hidden))
    scheduler.step()