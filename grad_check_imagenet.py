'''Test ImageNet with PyTorch.'''
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
from time import time

from models import *
from models.simple import *
from utils import progress_bar

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--n_data', default=50000, type=int, help='level of verbos')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
# traindir = os.path.join('/fs/vulcan-datasets/imagenet/', 'train')
traindir =  '/vulcanscratch/songweig/datasets/imagenet/train_subset'
valdir = os.path.join('/fs/vulcan-datasets/imagenet/', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize,
#     ]))

# # trainset_sub = torch.utils.data.Subset(trainset, indices=np.arange(5000))
# train_loader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=200, shuffle=False,
#     num_workers=10, pin_memory=True)


val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=200, shuffle=False,
    num_workers=10, pin_memory=True)


log_dir = '/vulcanscratch/songweig/logs/adv_pool/imagenet_unnorm'
os.environ['TORCH_HOME'] = '/vulcanscratch/songweig/ckpts/pytorch_imagenet'
criterion = nn.CrossEntropyLoss()


model_list = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
            'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19']
# Model
for i, model_name in enumerate(model_list):
    print(model_name)
    # if i%2 != 1:
    #     continue
    fw = open(os.path.join(log_dir, '%s.txt'%model_name), 'a')
    net = torchvision.models.__dict__[model_name](pretrained=True)
    net = net.to(device)
    net.eval()
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    adv_correct = 0
    total = 0
    w_one = np.ones(224*224*3)/224.
    # np.linalg.norm(w_one)
    cos_thetas = [[] for _ in range(1099)]
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs.requires_grad = True
        net.zero_grad()
        output = net(inputs.cuda())
        loss = criterion(output, targets.cuda())
        loss.backward()
        grad = inputs.grad.cpu().data.numpy().reshape([200, 224*224*3])
        grad_norm = grad/np.linalg.norm(grad, axis=1, keepdims=True)
        coss = np.matmul(grad_norm, w_one)
        for cos, target in zip(coss, targets):
            cos_thetas[target].append(cos)
    mean_cos = [np.mean(cos_thetas[i]) for i in range(1000) if len(cos_thetas[i])>0]
    topk_idx = np.array(mean_cos).argsort()[-10:][::-1]
    print(' '.join(['%.2f±%.2f'%(np.mean(cos_thetas[topk_idx[i]]), np.std(cos_thetas[topk_idx[i]])) for i in range(10)]))
    for delta in range(5):
        delta = delta / 224.
        print(delta)
        corrects_p = [0 for _ in range(10)]
        corrects_n = [0 for _ in range(10)]
        corrects_r = [0 for _ in range(10)]
        totals = [0 for _ in range(10)]
        for batch_idx, (inputs, targets) in enumerate(testloader):
            output = net(inputs.cuda() + delta)
            correct = (output.argmax(1).cpu()==targets)
            for c, target in zip(correct, targets):
                corrects_p[target] += c
                totals[target] += 1
            output = net(inputs.cuda() - delta)
            correct = (output.argmax(1).cpu()==targets)
            for c, target in zip(correct, targets):
                corrects_n[target] += c
            randvec = np.random.random(224*224) - 0.5
            randvec_norm = torch.FloatTensor(randvec / np.linalg.norm(randvec)).cuda()*delta
            output = net(inputs.cuda() + randvec_norm.reshape(1, 1, 224, 224))
            correct = (output.argmax(1).cpu()==targets)
            for c, target in zip(correct, targets):
                corrects_r[target] += c
        print(' '.join(['%.2f'%(corrects_r[i]/totals[i]) for i in range(10)]))
        print(' '.join(['%.2f'%(corrects_p[i]/totals[i]) for i in range(10)]))
        print(' '.join(['%.2f'%(corrects_n[i]/totals[i]) for i in range(10)]))
