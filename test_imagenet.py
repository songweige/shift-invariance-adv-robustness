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

from models import *
from models.simple import *
from utils import progress_bar

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

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
traindir = os.path.join('/fs/vulcan-datasets/imagenet/', 'train')
valdir = os.path.join('/fs/vulcan-datasets/imagenet/', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=100, shuffle=False,
    num_workers=10, pin_memory=True)

model_list = ['alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 
            'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
            'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3', 'densenet121', 
            'densenet169', 'densenet201', 'densenet161', 'googlenet', 'mobilenet_v2', 'mnasnet1_0', 'shufflenetv2', 'shufflenet_v2_x1_0']


log_dir = '/vulcanscratch/songweig/logs/adv_pool/imagenet'
os.environ['TORCH_HOME'] = '/vulcanscratch/songweig/ckpts/pytorch_imagenet'
attack_params = [[2, [0.5, 1, 1.5, 2, 2.5]], [np.inf, [2/255., 4/255., 8/255., 16/255.,32/255.]]]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# Model
for model_name in model_list[:15]:
    fw = open(os.path.join(log_dir, '%s.txt'%model_name))
    net = torchvision.models.__dict__['alexnet'](pretrained=True)
    net = net.to(device)
    net.eval()
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True
    fw.write('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    total = 0.
    correct = 0.
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        predicted = net(inputs.cuda()).argmax(1)
        correct += (predicted.cpu().numpy()==targets.numpy()).sum().item()
        total += targets.size(0)
    fw.write("Accuracy on clean test examples: {:.2f}%".format(100.*correct/total))
    classifier = PyTorchClassifier(
        model=net,
        loss=criterion,
        optimizer=optimizer,
        clip_values=(0., 1.),
        input_shape=(1, 224, 224),
        nb_classes=1000,
    )
    for norm, epsilons in attack_params:
        for epsilon in epsilons:
            attack_FGM = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
            attack_PGD = ProjectedGradientDescentPyTorch(estimator=classifier, eps=epsilon, norm=norm)
            adv_correct_FGM = 0
            adv_correct_PGD = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs_adv_PGD = attack_PGD.generate(x=inputs)
                inputs_adv_FGM = attack_FGM.generate(x=inputs)
                adv_predicted_PGD = classifier.predict(inputs_adv_PGD).argmax(1)
                adv_predicted_FGM = classifier.predict(inputs_adv_FGM).argmax(1)
                adv_correct_PGD += (adv_predicted_PGD==targets.numpy()).sum().item()
                adv_correct_FGM += (adv_predicted_FGM==targets.numpy()).sum().item()
                total += targets.size(0)
            print("Accuracy on FGM test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct_FGM/total))
            print("Accuracy on PGD test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct_PGD/total))