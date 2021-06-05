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

from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from models import vit
from models import resnet_bit

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model', default='VGG16', type=str, help='name of the model')
parser.add_argument('--n_data', default=50000, type=int, help='level of verbos')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--log_dir', help='name and directory for the log file',
                    default='./logs/adv_pool/imagenet_unnorm')
parser.add_argument('--model_path', help='location of the model checkpoint',
                    default='./ckpts/vit/')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
valdir = os.path.join('./imagenet/', 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

log_dir = args.log_dir
attack_params = [[2, [0.125, 0.25, 0.5, 1]], [np.inf, [0.5/255., 1/255., 2/255., 4/255.]]]

criterion = nn.CrossEntropyLoss()
val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.ToTensor(),
        # normalize,
    ])),
    batch_size=8, shuffle=False,
    num_workers=10, pin_memory=True)


########################################################################################################################################################
### ViT

def get_b16_config(config):
    """ ViT-B/16 configuration """
    config['patch_size'] = 16
    config['emb_dim'] = 768
    config['mlp_dim'] = 3072
    config['num_heads'] = 12
    config['num_layers'] = 12
    config['attn_dropout_rate'] = 0.0
    config['dropout_rate'] = 0.1
    return config


def get_b32_config(config):
    """ ViT-B/32 configuration """
    config = get_b16_config(config)
    config['patch_size'] = 32
    return config


def get_l16_config(config):
    """ ViT-L/16 configuration """
    config['patch_size'] = 16
    config['emb_dim'] = 1024
    config['mlp_dim'] = 4096
    config['num_heads'] = 16
    config['num_layers'] = 24
    config['attn_dropout_rate'] = 0.0
    config['dropout_rate'] = 0.1
    return config


def get_l32_config(config):
    """ Vit-L/32 configuration """
    config = get_l16_config(config)
    config['patch_size'] = 32
    return config


def get_vit_config(model_name):
    if model_name == 'ViT_B_16':
        return get_b16_config({})
    elif model_name == 'ViT_B_32':
        return get_b32_config({})
    elif model_name == 'ViT_L_16':
        return get_l16_config({})
    elif model_name == 'ViT_L_32':
        return get_l32_config({})

# Model
model_names = ['ViT_B_16', 'ViT_B_32', 'ViT_L_16', 'ViT_L_32']

for model_name in model_names:
    print(model_name)
    config = get_vit_config(model_name)
    config['image_size'] = 384
    config['num_classes'] = 1000
    net = vit.VisionTransformer(
                 image_size=(config['image_size'], config['image_size']),
                 patch_size=(config['patch_size'], config['patch_size']),
                 emb_dim=config['emb_dim'],
                 mlp_dim=config['mlp_dim'],
                 num_heads=config['num_heads'],
                 num_layers=config['num_layers'],
                 num_classes=config['num_classes'],
                 attn_dropout_rate=config['attn_dropout_rate'],
                 dropout_rate=config['dropout_rate'])
    ckpt_file = torch.load(os.path.join(args.model_path, '%s.pth'%model_name))
    net.load_state_dict(ckpt_file['state_dict'])
    net = net.to(device)
    net.eval()
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True
    with open(os.path.join(log_dir, '%s2.txt'%model_name), 'a') as fw:
        fw.write('Number of parameters: %d\n'%sum(p.numel() for p in net.parameters()))
        total = 0.
        correct = 0.
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            predicted = net(inputs.cuda()).argmax(1)
            correct += (predicted.cpu().numpy()==targets.numpy()).sum().item()
            total += targets.size(0)
        print("Accuracy on clean test examples: {:.2f}%".format(100.*correct/total))
        fw.write("Accuracy on clean test examples: {:.2f}%".format(100.*correct/total))
        classifier = PyTorchClassifier(
            model=net,
            loss=criterion,
            input_shape=(3, 384, 384),
            nb_classes=1000,
            preprocessing=(np.array([0.485, 0.456, 0.406]),
                           np.array([0.229, 0.224, 0.225]))
        )
        for norm, epsilons in attack_params[:1]:
            for epsilon in epsilons[:1]:
                # attack_FGM = FastGradientMethod(estimator=classifier, eps=epsilon, norm=norm)
                attack_PGD = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100, eps_step=epsilon/5, eps=epsilon, norm=norm)
                # adv_correct_FGM = 0
                adv_correct_PGD = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(val_loader):
                    # begin_time = time()
                    inputs_adv_PGD = attack_PGD.generate(x=inputs)
                    # inputs_adv_FGM = attack_FGM.generate(x=inputs)
                    adv_predicted_PGD = classifier.predict(inputs_adv_PGD).argmax(1)
                    # adv_predicted_FGM = classifier.predict(inputs_adv_FGM).argmax(1)
                    adv_correct_PGD += (adv_predicted_PGD==targets.numpy()).sum().item()
                    # adv_correct_FGM += (adv_predicted_FGM==targets.numpy()).sum().item()
                    total += targets.size(0)
                    # print('batch %d took %.4f seconds'%(batch_idx, time()-begin_time))
                    if total > 10000:
                        break
                print("Accuracy on FGM test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct_PGD/total))
                fw.write("Accuracy on PGD test examples (L_{:.0f}, eps={:.2f}): {:.2f}%\n".format(norm, epsilon, 100.*adv_correct_PGD/total))

########################################################################################################################################################
### BiT

def get_bit_model(model_name):
    if model_name == 'BiT_resnet_50x1':
        return resnet_bit.resnetv2_50x1_bitm(pretrained=True)
    elif model_name == 'BiT_resnet_50x3':
        return resnet_bit.resnetv2_50x3_bitm(pretrained=True)
    elif model_name == 'BiT_resnet_101x1':
        return resnet_bit.resnetv2_101x1_bitm(pretrained=True)
    elif model_name == 'BiT_resnet_101x3':
        return resnet_bit.resnetv2_101x3_bitm(pretrained=True)
    elif model_name == 'BiT_resnet_152x2':
        return resnet_bit.resnetv2_152x2_bitm(pretrained=True)
    elif model_name == 'BiT_resnet_152x4':
        return resnet_bit.resnetv2_152x4_bitm(pretrained=True)


model_names = ['BiT_resnet_50x1', 'BiT_resnet_50x3', 'BiT_resnet_101x1', 'BiT_resnet_101x3', 'BiT_resnet_152x2', 'BiT_resnet_152x4']

for model_name in model_names[5:6]:
    print(model_name)
    net = get_bit_model(model_name)
    fw = open(os.path.join(log_dir, '%s2.txt'%model_name), 'a')
    net = net.to(device)
    net.eval()
    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        net = net.cuda()
        cudnn.benchmark = True
    fw.write('Number of parameters: %d\n'%sum(p.numel() for p in net.parameters()))
    total = 0.
    correct = 0.
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        predicted = net(inputs.cuda()).argmax(1)
        correct += (predicted.cpu().numpy()==targets.numpy()).sum().item()
        total += targets.size(0)
    print("Accuracy on clean test examples: {:.2f}%".format(100.*correct/total))
    fw.write("Accuracy on clean test examples: {:.2f}%".format(100.*correct/total))
    classifier = PyTorchClassifier(
        model=net,
        loss=criterion,
        input_shape=(3, 384, 384),
        nb_classes=1000,
        preprocessing=(np.array([0.485, 0.456, 0.406]),
                       np.array([0.229, 0.224, 0.225]))
    )
    for norm, epsilons in attack_params[:1]:
        for epsilon in epsilons:
            attack_PGD = ProjectedGradientDescentPyTorch(estimator=classifier, max_iter=10, batch_size=100, eps_step=epsilon/5, eps=epsilon, norm=norm)
            # adv_correct_FGM = 0
            adv_correct_PGD = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # begin_time = time()
                inputs_adv_PGD = attack_PGD.generate(x=inputs)
                # inputs_adv_FGM = attack_FGM.generate(x=inputs)
                adv_predicted_PGD = classifier.predict(inputs_adv_PGD).argmax(1)
                # adv_predicted_FGM = classifier.predict(inputs_adv_FGM).argmax(1)
                adv_correct_PGD += (adv_predicted_PGD==targets.numpy()).sum().item()
                # adv_correct_FGM += (adv_predicted_FGM==targets.numpy()).sum().item()
                total += targets.size(0)
                if total > 10000:
                    break
                # print('batch %d took %.4f seconds'%(batch_idx, time()-begin_time))
            print("Accuracy on FGM test examples (L_{:.0f}, eps={:.2f}): {:.2f}%".format(norm, epsilon, 100.*adv_correct_PGD/total))
            fw.write("Accuracy on PGD test examples (L_{:.0f}, eps={:.2f}): {:.2f}%\n".format(norm, epsilon, 100.*adv_correct_PGD/total))


# consistency = 0
# total = 0
# for batch_idx, (inputs, targets) in enumerate(val_loader):
#     shift0 = np.random.randint(384,size=2)
#     inputs_shift_v0 = torch.zeros([16, 3, 384, 384])
#     inputs_shift_hv0 = torch.zeros([16, 3, 384, 384])
#     inputs_shift_v0[:, :, :, :shift0[0]] = inputs[:, :, :, (384-shift0[0]):].clone()
#     inputs_shift_v0[:, :, :, shift0[0]:] = inputs[:, :, :, :(384-shift0[0])].clone()
#     inputs_shift_hv0[:, :, :shift0[1], :] = inputs_shift_v0[:, :, (384-shift0[1]):, :]
#     inputs_shift_hv0[:, :, shift0[1]:, :] = inputs_shift_v0[:, :, :(384-shift0[1]), :]
#     predicted0 = net(inputs_shift_hv0.cuda()).argmax(1)
#     shift1 = np.random.randint(384,size=2)
#     inputs_shift_v1 = torch.zeros([16, 3, 384, 384])
#     inputs_shift_hv1 = torch.zeros([16, 3, 384, 384])
#     inputs_shift_v1[:, :, :, :shift1[0]] = inputs[:, :, :, (384-shift1[0]):].clone()
#     inputs_shift_v1[:, :, :, shift1[0]:] = inputs[:, :, :, :(384-shift1[0])].clone()
#     inputs_shift_hv1[:, :, :shift1[1], :] = inputs_shift_v1[:, :, (384-shift1[1]):, :]
#     inputs_shift_hv1[:, :, shift1[1]:, :] = inputs_shift_v1[:, :, :(384-shift1[1]), :]
#     predicted1 = net(inputs_shift_hv1.cuda()).argmax(1)
#     consistency += (predicted0==predicted1).sum().item()
#     total += targets.size(0)

# print("Consistency on clean test examples: {:.2f}%".format(100.*consistency/total))