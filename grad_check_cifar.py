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

from DDN import DDN

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--n_data', default=50000, type=int, help='level of verbos')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.n_data<50000:
    trainSet = torchvision.datasets.CIFAR10(
        root='/fs/vulcan-datasets/CIFAR/', train=True, download=True, transform=transform_test)
    trainSet = torch.utils.data.Subset(trainSet, indices=np.arange(args.n_data))
    trainloader = torch.utils.data.DataLoader(
        trainSet, batch_size=100, shuffle=False, num_workers=2)
    

testset = torchvision.datasets.CIFAR10(
    root='/fs/vulcan-datasets/CIFAR/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()

args.model = 'alexnet'
models = ['alexnet', 'resnet_18', 'resnet_34', 'resnet_50', 'resnet_101', 'resnet_152']
for args.model in models:
# Model
    print('==> Building model..')
    net = get_net(args.model)
    net = net.to(device)
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True
    net.eval()
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    if args.n_data < 50000:
        checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/cifar10/%s_%d.pth'%(args.model, args.n_data))
        print("Accuracy on clean train examples: {:.2f}%".format(checkpoint['train_acc']))
    else:
        checkpoint = torch.load('/vulcanscratch/songweig/ckpts/adv_pool/cifar10/%s.pth'%args.model)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("Accuracy on clean test examples: {:.2f}%".format(best_acc))
    adv_correct = 0
    total = 0
    w_one = np.ones(3*32*32)/(np.sqrt(3*32*32))
    # w_one = np.ones(32*32)/32
    np.linalg.norm(w_one)
    cos_thetas = [[] for _ in range(10)]
    grads = [[] for _ in range(10)]
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs.requires_grad = True
        net.zero_grad()
        output = net(inputs.cuda())
        loss = criterion(output, targets.cuda())
        loss.backward()
        # grad = inputs.grad.cpu().data.numpy()[:, 0:1].reshape([100, 32*32])
        grad = inputs.grad.cpu().data.numpy().reshape([100, 3*32*32])
        grad_norm = grad/np.linalg.norm(grad, axis=1, keepdims=True)
        # coss = np.matmul(grad_norm, w_one)
        # for cos, target in zip(coss, targets):
        #     cos_thetas[target].append(np.abs(cos))
        for g, target in zip(grad, targets):
            cos_thetas[target].append(g)
    # print(' '.join(['%.2f±%.2f'%(np.mean(cos_thetas[i]), np.std(cos_thetas[i])) for i in range(10)]))
    for i in range(10):
        mean_direction = np.mean(np.array(grads[i]), 1)
        cos_theta[i] = [np.matmul(grad_norm, mean_direction) for grad in grads]
    print(' '.join(['%.2f±%.2f'%(np.mean(cos_thetas[i]), np.std(cos_thetas[i])) for i in range(10)]))
    # for delta in range(5):
    #     delta = delta / (np.sqrt(3*32*32))
    #     print(delta)
    #     corrects_p = [0 for _ in range(10)]
    #     corrects_n = [0 for _ in range(10)]
    #     corrects_r = [0 for _ in range(10)]
    #     totals = [0 for _ in range(10)]
    #     for batch_idx, (inputs, targets) in enumerate(testloader):
    #         output = net(inputs.cuda() + delta)
    #         correct = (output.argmax(1).cpu()==targets)
    #         for c, target in zip(correct, targets):
    #             corrects_p[target] += c
    #             totals[target] += 1
    #         output = net(inputs.cuda() - delta)
    #         correct = (output.argmax(1).cpu()==targets)
    #         for c, target in zip(correct, targets):
    #             corrects_n[target] += c
    #         randvec = np.random.random(1024*3) - 0.5
    #         randvec_norm = torch.FloatTensor(randvec / np.linalg.norm(randvec)).cuda()*delta
    #         output = net(inputs.cuda() + randvec_norm.reshape(1, 3, 32, 32))
    #         correct = (output.argmax(1).cpu()==targets)
    #         for c, target in zip(correct, targets):
    #             corrects_r[target] += c
    #     print(' '.join(['%.2f'%(corrects_r[i]/totals[i]) for i in range(10)]))
    #     print(' '.join(['%.2f'%(corrects_p[i]/totals[i]) for i in range(10)]))
    #     print(' '.join(['%.2f'%(corrects_n[i]/totals[i]) for i in range(10)]))
