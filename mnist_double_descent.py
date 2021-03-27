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

from models import *
from models.simple import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_hidden', default=1000, type=int, help='number of hidden units')
parser.add_argument('--verbose', default=0, type=int, help='level of verbos')
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
    trainset, batch_size=256, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='/vulcanscratch/songweig/datasets/mnist', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

resnet_dict = {'18':ResNet_MNIST_18, '34':ResNet_MNIST_34, '50':ResNet_MNIST_50, '101':ResNet_MNIST_101, '152':ResNet_MNIST_152}

# Model
net = simple_FC(args.n_hidden)

print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
net = net.cuda()

net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    net = net.cuda()
    cudnn.benchmark = True


### initialization
if args.n_hidden <= 5:
    torch.nn.init.xavier_uniform_(net.features[1].weight, gain=1.0)
    torch.nn.init.xavier_uniform_(net.classifier.weight, gain=1.0)
elif args.n_hidden > 50:
    torch.nn.init.normal_(net.features[1].weight, mean=0.0, std=0.1)
    torch.nn.init.normal_(net.classifier.weight, mean=0.0, std=0.1)
else:
    torch.nn.init.normal_(net.features[1].weight, mean=0.0, std=0.1)
    torch.nn.init.normal_(net.classifier.weight, mean=0.0, std=0.1)
    checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/double_descent_4k/simple_%s_%d.pth'%(args.model, args.n_hidden-5))
    with torch.no_grad():
        net.features[1].weight[:args.n_hidden-5, :].copy_(checkpoint['net']['features.1.weight'])
        net.features[1].bias[:args.n_hidden-5].copy_(checkpoint['net']['features.1.bias'])
        net.classifier.weight[:, :args.n_hidden-5].copy_(checkpoint['net']['classifier.weight'])
        net.classifier.bias.copy_(checkpoint['net']['classifier.bias'])


# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.95)
                      # momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch, net):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = torch.nn.functional.one_hot(targets, num_classes=10).float()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        # import ipdb; ipdb.set_trace()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(1)).sum().item()
        # correct += predicted.eq(targets).sum().item()
    if args.verbose > 1:
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
            targets = torch.nn.functional.one_hot(targets, num_classes=10).float()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()
            # correct += predicted.eq(targets).sum().item()
            if args.verbose > 1:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if epoch == start_epoch+5999:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/vulcanscratch/songweig/ckpts/adv_pool/double_descent_4k'):
            os.mkdir('/vulcanscratch/songweig/ckpts/adv_pool/double_descent_4k')
        torch.save(state, '/vulcanscratch/songweig/ckpts/adv_pool/double_descent_4k/%s.pth'%model_name)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+6000):
    if (epoch+1) % 500 == 0 or epoch == start_epoch+5999:
        test(epoch, net, 'simple_%s_%d'%(args.model, args.n_hidden))
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.9
    train(epoch, net)
