# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import os
from advertorch.attacks import LinfPGDAttack, L2PGDAttack

from models.basic_models import CNN, NeuralNet, CNN_FC
from utils import get_loaders_mnist, get_loaders_mnist_full, get_loaders_fashion_mnist_full
import argparse
import logging
import ipdb

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', choices=['cnn', 'fc', 'cnn_fc'])
    parser.add_argument('--fashion-mnist', action='store_true')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--hidden-size', default=5000, type=int)
    parser.add_argument('--hidden-layers', default=1, type=int)
    parser.add_argument('--kernel-size', default=7, type=int)
    parser.add_argument('--hidden-channels', default=1024, type=int)
    parser.add_argument('--learning-rate', default=0.01, type=float)
    parser.add_argument('--images-use', default='all')
    parser.add_argument('--out-dir', type=str)
    parser.add_argument('--epsilons', type=str)
    parser.add_argument('--attack-type', choices=['l2', 'linf'])
    parser.add_argument('--eval-only', action='store_true')
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer, device, num_epochs):
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        if epoch==num_epochs//2 or epoch==int(num_epochs*3/4):
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            #         images = images.reshape(-1, 28*28).to(device)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # TODO - FIX THIS
            total += labels.size(0)
            # ipdb.set_trace()
            correct += (outputs.max(1)[1] == labels).sum().item()

            # Backprpagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), 100 * correct / total))

    logger.info("Total {} images used for training.".format(total))
    return model

def test(model, test_loader, device):
    # Test the model
    # In the test phase, don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            #         images = images.reshape(-1, 28*28).to(device)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            total += labels.size(0)
            correct += (outputs.max(1)[1] == labels).sum().item()

        logger.info('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

def test_pgd(model, test_loader, device, epsilon, attack_type):
    # Accuracy counter
    correct = 0
    total = 0
    adv_examples = []
    model = model.eval()

    if attack_type=='linf':
        adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), nb_iter=10, eps_iter=epsilon/5,
                                   rand_init=True, eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        adversary = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), nb_iter=10, eps_iter=epsilon/5,
                                rand_init=True, eps=epsilon, clip_min=0.0, clip_max=1.0, targeted=False)

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
    final_acc = 100.*correct / float(total)
    logger.info("Attack Type: {}, Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}".format(attack_type, epsilon,
                                                                                    correct, total,
                                                                                   final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

def main(args):
    # Check Device configuration
    os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, 'output.log')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    if not args.eval_only:
        logger.info(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Define Hyper-parameters
    input_size = 784
    hidden_size = args.hidden_size
    hidden_layers = args.hidden_layers
    hidden_channels = args.hidden_channels
    kernel_size = args.kernel_size
    num_classes = 10
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    images_use = int(args.images_use) if args.images_use!='all' else 60000
    if args.fashion_mnist:
        train_loader, test_loader = get_loaders_fashion_mnist_full(batch_size, images_use)
    else:
        train_loader, test_loader = get_loaders_mnist_full(batch_size, images_use)



    if args.model_type=='cnn':
        model = CNN(hidden_channels=hidden_channels, kernel_size=kernel_size, num_classes=num_classes, no_final=True)
    elif args.model_type=='cnn_fc':
        model = CNN_FC(hidden_channels=hidden_channels, kernel_size=kernel_size, num_classes=num_classes, no_final=True)
    else:
        model = NeuralNet(input_size, hidden_layers, hidden_size, num_classes, no_final=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if not args.eval_only:
        model = train(model, train_loader, criterion, optimizer, device, num_epochs)
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'final_checkpoint.pt'))
        test(model, test_loader, device)
    else:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, 'final_checkpoint.pt')))

    # model.eval()
    epsilons = [float(x) for x in args.epsilons.split(',')]
    for epsilon in epsilons:
        test_pgd(model, test_loader, device, epsilon, args.attack_type)


if __name__ == '__main__':
    args = parse_args()
    main(args)

