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

from datasets import *
from models.simple import *
from utils import progress_bar

from DDN import DDN

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


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
    progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


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
        progress_bar(batch_idx, 1, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

k = 28
# batch = create_dataset_sin_cos(k)
batch = create_dataset_dots(k)
n_hidden = 100
kernel_size = k
model = 'Conv'

# Model
if model == 'FC':
    net = simple_FC_2D(k*k, n_hidden)
elif model == 'Conv':
    net = simple_Conv_2D(n_hidden, kernel_size)

print('Number of parameters: %d'%sum(p.numel() for p in net.parameters()))
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


torch_batch = [torch.FloatTensor(batch[0]), torch.LongTensor(batch[1])]
for epoch in range(start_epoch, start_epoch+200):
    train(epoch, net, torch_batch)
    test(epoch, net, 'simple_%s_%d'%(model, n_hidden), torch_batch)
    scheduler.step()


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



def l2_norm(x: torch.Tensor) -> torch.Tensor:
    return squared_l2_norm(x).sqrt()


def squared_l2_norm(x: torch.Tensor) -> torch.Tensor:
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).sum(1)


inputs, targets = torch_batch
n_steps = [500]
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



import matplotlib.pyplot as plt

for i in range(4):
    plt.clf()
    plt.imshow(batch[0][i][0], cmap='gray')
    plt.savefig('/vulcanscratch/songweig/plots/adv_pool/synthetic/sincos%d.png'%i)


import cv2
from PIL import Image


for i in range(2):
    # plt.clf()
    # plt.imshow((batch[0][i][0]+1)/2., cmap='gray')
    # plt.axis('off')
    # plt.savefig('/vulcanscratch/songweig/plots/adv_pool/synthetic/twodots%d.png'%i)
    img = np.zeros([150, 150])*0.5
    img[75:85, 75:85] = i
    # im = Image.fromarray(np.uint8((batch[0][i][0]+1)/2* 255), 'L')
    im = Image.fromarray(np.uint8(img* 255), 'L')
    im.save('/vulcanscratch/songweig/plots/adv_pool/synthetic/twodots%d.png'%i)