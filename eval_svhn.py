import torch
import numpy as np
from torch import nn
import logging
import os
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD
import argparse
import torchvision
import ipdb
from models.basic_models import NeuralNet
from utils import get_loaders_svhn
import eagerpy as ep


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Type of architecture for the model',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'fc'])
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-size', type=int, default=1000)
    parser.add_argument('--load-dir', help='location of the state dictionary to load')
    parser.add_argument('--batch-size', default=128, type=int)
    return parser.parse_args()

def test_pgd(model, test_loader, device, epsilon, attack_type):
    # Accuracy counter
    correct = 0
    total = 0
    adv_examples = []
    model = model.eval()

    if attack_type == 'linf':
        adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), nb_iter=10,
                                  eps_iter=epsilon/5,
                                  rand_init=True, eps=epsilon, clip_min=-0.4914/0.2023, clip_max=(1-0.4465)/0.2010, targeted=False)
    else:
        adversary = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), nb_iter=10, eps_iter=epsilon/5,
                                rand_init=True, eps=epsilon, clip_min=-0.4914/0.2023, clip_max=(1-0.4465)/0.2010, targeted=False)

    # Loop over all examples in test set
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        perturbed_data = adversary.perturb(data, target)
        new_output = model(perturbed_data)

        total += target.size(0)
        correct += (new_output.max(1)[1] == target).sum().item()

    # Calculate final accuracy for this epsilon
    final_acc = 100. * correct / float(total)
    logger.info("Attack Type: {}, Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(attack_type, epsilon,
                                                                                        correct, total,
                                                                                        final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def test_clean_foolbox(model, test_loader, device):
    correct = 0.
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        clean_acc = accuracy(model, data, target)
        total+=target.size(0)
        correct+=clean_acc*target.size(0)
    logger.info("Clean Accuracy : {:.3f}".format(100.*correct/total))


def test_pgd_foolbox(model, attacker, test_loader, device, epsilons):
    robust_correct = np.array([0.]*len(epsilons))
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data_ep = ep.astensor(data)
        target_ep = ep.astensor(target)
        # Set requires_grad attribute of tensor. Important for Attack
        # data.requires_grad = True
        raw_advs, clipped_advs, success = attacker(model, data_ep, target_ep, epsilons=epsilons)
        robust_batch_accuracy = 1 - success.float32().mean(axis=-1)
        total+=target.size(0)
        # ipdb.set_trace()
        robust_batch_correct = robust_batch_accuracy.numpy()*target.size(0)
        robust_correct += robust_batch_correct
    return robust_correct/total


def main(args):
    logfile = os.path.join(args.load_dir, 'robustness.log')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)

    if args.model.startswith('resnet'):
        model = torch.nn.DataParallel(eval('torchvision.models.' + args.model + '(pretrained=False)'))
    else:
        model = NeuralNet(32*32*3, args.hidden_layers, args.hidden_size, 10, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    train_loader, test_loader = get_loaders_svhn(args.batch_size, normalize=False)
    model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.th'))['state_dict'])
    model.eval()

    # FoolBox Stuff
    preprocessing = dict(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    linf_epsilons = [2./255., 4./255., 6./255, 8./255., 16./255., 32./255.]
    l2_epsilons = [0.25, 0.5, 0.75, 1., 1.5, 2.]
    linf_attack = LinfPGD(rel_stepsize=1./5, steps=10)
    l2_attack = L2PGD(rel_stepsize=1./5, steps=10)

    # Check Clean Accuracy once to see if everything is right
    # ipdb.set_trace()
    test_clean_foolbox(fmodel, test_loader, device)
    robust_acc_linf = test_pgd_foolbox(fmodel, linf_attack, test_loader, device, linf_epsilons)
    for eps, acc in zip(linf_epsilons, robust_acc_linf):
        logger.info("L-Inf Attack - {:.3f} epsilon, {:.3f} accuracy".format(eps, 100.*acc))
    robust_acc_l2 = test_pgd_foolbox(fmodel, l2_attack, test_loader, device, l2_epsilons)
    for eps, acc in zip(l2_epsilons, robust_acc_l2):
        logger.info("L-2 Attack - {:.3f} epsilon, {:.3f} accuracy".format(eps, 100.*acc))

    # raw_advs, clipped_advs, success = linf_attack(fmodel, images, labels, epsilons=l2_epsilons)
    # RE-THINK EXPERIMENT FOR DIFFERENT L-INF EPSILONS
    # for eps in [2./255., 4./255., 6./255, 8./255., 16./255., 32./255.]:
    #     test_pgd(model, test_loader, device, eps, 'linf')
    # for eps in [0.25, 0.5, 0.75, 1., 1.5, 2.]:
    #     test_pgd(model, test_loader, device, eps, 'l2')
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)