'''Train Simple with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from datasets import *
from models.simple import *
from utils import progress_bar

from DDN import DDN

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


def create_dataset_dots(k=100):
    X_1 = np.zeros([1, 1, k, k])
    X_1[:, :, k//2, k//2] = 1
    X_2 = np.zeros([1, 1, k, k])
    X_2[:, :, k//2, k//2] = -1
    Xs = np.concatenate([X_1, X_2], 0)
    ys = np.hstack([np.zeros(1), np.ones(1)])
    return Xs, ys


class simple_FC_2D(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(simple_FC_2D, self).__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, n_hidden),
            nn.ReLU()
        )
        self.classifier = nn.Linear(n_hidden, 2)
        # self.features[1].bias.data.zero_()
        # self.features[1].weight.data.zero_()
        # self.classifier.bias.data.zero_()
        # self.classifier.weight.data.zero_()
    def forward(self, x):
        out = self.features(x)
        out = self.classifier(out)
        return out


class simple_Conv_2D(nn.Module):
    def __init__(self, n_hidden, kernel_size=28):
        super(simple_Conv_2D, self).__init__()
        padding_size = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, n_hidden, kernel_size=kernel_size, padding=padding_size, padding_mode='circular'),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(n_hidden, 2)
        # self.features[0].bias.data.zero_()
        # self.features[0].weight.data.zero_()
        # self.classifier.bias.data.zero_()
        # self.classifier.weight.data.zero_()
    def forward(self, x):
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# Training
def train(epoch, net, batch):
    print('\nEpoch: %d' % epoch)
    inputs, targets = batch
    batch_idx = 0
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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
    # progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)


def test(epoch, net, model_name, batch):
    global best_acc
    inputs, targets = batch
    batch_idx = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if epoch == start_epoch+199:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/vulcanscratch/songweig/ckpts/adv_pool/simple'):
            os.mkdir('/vulcanscratch/songweig/ckpts/adv_pool/simple')
        torch.save(state, '/vulcanscratch/songweig/ckpts/adv_pool/simple/%s.pth'%model_name)
        best_acc = acc



def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model = 'FC'

dists = []
losses = []
for i in range(30):
    k = i*2 + 1
    # k = 25
    # batch = create_dataset_sin_cos(k)
    batch = create_dataset_dots(k)
    n_hidden = 50
    kernel_size = k
    # Model
    if model == 'FC':
        net = simple_FC_2D(k*k, n_hidden)
    elif model == 'Conv':
        net = simple_Conv_2D(n_hidden, kernel_size)
    print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
    net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    torch_batch = [torch.FloatTensor(batch[0]), torch.LongTensor(batch[1])]
    for epoch in range(start_epoch, start_epoch+1000):
        loss_epoch = train(epoch, net, torch_batch)
        test(epoch, net, 'simple_%s_%d'%(model, n_hidden), torch_batch)
        # scheduler.step()
    inputs, targets = torch_batch
    n_steps = [1000]
    for n_step in n_steps:
        attacker = DDN(steps=n_step, device=torch.device('cuda'))
        adv_norm = 0.
        adv_correct = 0
        total = 0
        inputs, targets = inputs.to(device), targets.to(device)
        adv = attacker.attack(net, inputs.to(device), labels=targets.to(device), targeted=False)
        adv_outputs = net(adv)
        _, adv_predicted = adv_outputs.max(1)
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()
        adv_norm += l2_norm(adv - inputs.to(device)).sum().item()
        print('DDN (n-step = {:.0f}) done in Success: {:.2f}%, Mean L2: {:.4f}.'.format(
            n_step,
            100.*adv_correct/total,
            adv_norm/total
        ))
    dists.append(adv_norm/total)
    losses.append(loss_epoch)


torchvision.utils.save_image(inputs[0]/2.+0.5, 'clean1.png')
torchvision.utils.save_image(inputs[1]/2.+0.5, 'clean2.png')
torchvision.utils.save_image(adv[0]/2.+0.5, 'fc_adv1.png')
torchvision.utils.save_image(adv[1]/2.+0.5, 'fc_adv2.png')
torchvision.utils.save_image(adv[0]/2.+0.5, 'conv_adv1.png')
torchvision.utils.save_image(adv[1]/2.+0.5, 'conv_adv2.png')

net.features[0].bias.grad
net.features[0].weight.data[0, 0]
net.features[1].weight.data[0].abs().argmax()
net.features[1].weight.grad
# classifier = PyTorchClassifier(
#     model=net,
#     loss=criterion,
#     optimizer=optimizer,
#     clip_values=(0., 1.),
#     input_shape=(1, 28, 28),
#     nb_classes=2,
# )
# print("Accuracy on clean test examples: {:.2f}%".format(best_acc))

# inputs, targets = torch_batch
# attack_params = [[2, [0.5, 1, 1.5, 2, 2.5]], [np.inf, [0.5, 0.6, 0.7, 0.8, 0.9]]]
# for norm, epsilons in attack_params:
#     for epsilon in epsilons:
#         attack = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
#         # attack = ProjectedGradientDescentPyTorch(estimator=classifier, eps=epsilon, norm=norm)
#         adv_correct = 0
#         total = 0
#         inputs_adv = attack.generate(x=inputs)
#         adv_predicted = classifier.predict(inputs_adv).argmax(1)
#         adv_correct += (adv_predicted==targets.numpy()).sum().item()
#         total += targets.size(0)
#         print("Accuracy on adversarial test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct/total))

import matplotlib
import matplotlib.pyplot as plt

FCN_accs = [1.0039215087890625, 0.9850332140922546, 0.976715087890625, 0.9680507183074951, 0.971983790397644, 
            0.9777611494064331, 0.9765254855155945, 0.9740734696388245, 0.9753726720809937, 0.9727703928947449, 
            0.9746283292770386, 0.9760518670082092, 0.9730303287506104, 0.9819855093955994, 0.9761784076690674]

CNN_accs = [1.0039215087890625, 0.33606475591659546, 0.203160360455513, 0.1456798017024994, 0.11684942245483398, 
            0.10624721646308899, 0.08942484855651855, 0.07604095339775085, 0.06655122339725494, 0.07440652698278427, 
            0.08225951343774796, 0.09011077880859375, 0.09796074032783508, 0.10580970346927643, 0.11365785449743271]

CAND_COLORS = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f', '#ff7f00','#cab2d6','#6a3d9a', '#90ee90', '#9B870C', '#2f4554', '#61a0a8', '#d48265', '#c23531']
colors1 = CAND_COLORS[::2]
colors2 = CAND_COLORS[1::2]

font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)

plt.clf()
ks = [i*2 + 1 for i in range(15)]
ax = plt.subplot(111)
plt.plot(ks, FCN_accs[:15], marker='o', linewidth=3, markersize=8, label='FCN', color=colors2[0], alpha=0.8)
plt.plot(ks, CNN_accs[:15], marker='^', linewidth=3, markersize=8, label='CNN', color=colors2[2], alpha=0.8)
plt.plot(ks, [1.0/(i*2+1) for i in range(15)], marker='*', linewidth=3, markersize=8, label='CNTK', color=colors2[1], alpha=0.8)
# plt.tight_layout()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
             box.width, box.height * 0.9])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.178), ncol=3)
plt.ylabel('average distance of attack')
plt.xlabel('image width')
plt.savefig('/vulcanscratch/songweig/plots/adv_pool/toy2.png')




for i in range(4):
    plt.clf()
    plt.imshow(batch[0][i][0], cmap='gray')
    plt.savefig('/vulcanscratch/songweig/plots/adv_pool/synthetic/sincos%d.png'%i)


import cv2
import numpy as np
from PIL import Image


for i in range(2):
    # plt.clf()
    # plt.imshow((batch[0][i][0]+1)/2., cmap='gray')
    # plt.axis('off')
    # plt.savefig('/vulcanscratch/songweig/plots/adv_pool/synthetic/twodots%d.png'%i)
    img = np.ones([150, 150])*0.5
    img[70:80, 70:80] = i
    # im = Image.fromarray(np.uint8((batch[0][i][0]+1)/2* 255), 'L')
    im = Image.fromarray(np.uint8(img* 255), 'L')
    im.save('/vulcanscratch/songweig/plots/adv_pool/synthetic/twodots%d.png'%i)