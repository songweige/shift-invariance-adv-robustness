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
import logging

from models import *
from models.simple import *
from models.sparse_mlp import *
from utils import progress_bar

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--n_data', default=50000, type=int, help='level of verbos')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_path', type=str, help='path for the model')
parser.add_argument('--log_file', type=str, help='logfile name')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# setup logging
# os.makedirs(args.save_dir, exist_ok=True)
logfile = args.log_file
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)

# Data
logger.info('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
logger.info('==> Building model..')
net = get_net(args.model)
net = net.to(device)
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
    logger.info("Accuracy on clean train examples: {:.2f}%".format(checkpoint['train_acc']))
else:
    checkpoint = torch.load(os.path.join(args.model_path, args.model+'.pth'))
if 'net' in checkpoint.keys():
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
else:
    net.load_state_dict(checkpoint['state_dict'])
    best_acc = checkpoint['best_prec1']
# start_epoch = checkpoint['epoch']
if not args.model.startswith('sparse'):
    classifier = PyTorchClassifier(
        model=net,
        loss=criterion,
        input_shape=(1, 32, 32),
        nb_classes=10,
        preprocessing=(np.array((0.4914, 0.4822, 0.4465)),
                       np.array((0.2023, 0.1994, 0.2010)))
        )
else:
    classifier = PyTorchClassifier(
        model=net,
        loss=criterion,
        input_shape=(32*32*3,),
        nb_classes=10,
        preprocessing=(np.array([0.485, 0.456, 0.406]),
                       np.array([0.229, 0.224, 0.225]))
        )

logger.info("Accuracy on clean test examples: {:.2f}%".format(best_acc))

attack_params = [[2, [0., 0.1, 0.2, 0.3, 0.4]], [np.inf, [0., 0.5/255., 1./255., 2./255., 4./255.]]]

for norm, epsilons in attack_params:
    for epsilon in epsilons:
        if epsilon == 0:
            eps_step = 1e-8
        else:
            eps_step = epsilon/5
        if norm == 2:
            attack = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100, eps_step=eps_step, eps=epsilon, norm=norm)
        else:
            attack = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100, eps_step=eps_step, eps=epsilon, norm=norm)
        adv_correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs_adv = attack.generate(x=inputs)
            adv_predicted = classifier.predict(inputs_adv).argmax(1)
            adv_correct += (adv_predicted==targets.numpy()).sum().item()
            total += targets.size(0)
        logger.info("Accuracy on adversarial test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct/total))


if args.n_data < 50000:
    for norm, epsilons in attack_params:
        for epsilon in epsilons:
            # attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
            if norm == 2:
                attack = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100,
                                                         eps_step=epsilon/5, eps=epsilon, norm=norm, verbose=False)
            else:
                attack = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100,
                                                         eps_step=epsilon/5, eps=epsilon, norm=norm, verbose=False)
            adv_correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs_adv = attack.generate(x=inputs)
                adv_predicted = classifier.predict(inputs_adv).argmax(1)
                adv_correct += (adv_predicted==targets.numpy()).sum().item()
                total += targets.size(0)
            logger.info("Accuracy on adversarial train examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct/total))