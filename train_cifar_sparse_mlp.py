import torch
from torchvision import datasets, transforms
import os
from models import SmallFCN
from utils import AverageMeter, accuracy, save_checkpoint

import time
import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--l1', default=1e-4, type=float)
    parser.add_argument('--beta', default=50., type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=50)
    # TODO - LOG AND SAVE EVERYTHING
    return parser.parse_args()


class BetaLassoOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, lambda_=1e-5, beta=50):
        defaults = dict(lr=lr)
        self.lambda_ = lambda_
        self.beta = beta
        self.thresh_val = self.beta * self.lambda_
        super(BetaLassoOpt, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                step_size = group["lr"]
                p.data.sub_((p.grad + self.lambda_ * p.data.sign()) * step_size)
                p.data[p.abs() < self.thresh_val] = 0.


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'
                .format(top1=top1))

    return top1.avg


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    logfile = os.path.join(args.save_dir, 'output.log')
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)

    epochs = args.epochs
    batch_size = args.batch_size

    model = SmallFCN(150)
    model = model.cuda()

    optimizer = BetaLassoOpt(model.parameters(), lr=args.lr, beta=args.beta, lambda_=args.l1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, verbose=False)

    if args.data_aug:
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomRotation(10),
                                              #                                       transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])
                                              ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    trainset = datasets.CIFAR10(
        root=os.getenv('CIFAR_DIR'), train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = datasets.CIFAR10(
        root=os.getenv('CIFAR_DIR'), train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    #     beta_lasso_opt = BetaLasso(model, lambda_=lambda_, beta=beta)
    criterion = torch.nn.CrossEntropyLoss().cuda()

    save_checkpoint({'epoch': 0,
                     'state_dict': model.state_dict()},
                    True, filename=os.path.join(args.save_dir, 'checkpoint_0.th'))

    best_prec1 = 0

    for epoch in range(0, epochs):
        start_epoch = time.time()
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion)

        lr_scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, True, filename=os.path.join(args.save_dir, 'checkpoint_{}.th'.format(epoch)))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'best.th'))

        logger.info("Epoch Time: {:.3f}, LR: {:.4f}".format(time.time() - start_epoch, lr_scheduler.get_last_lr()[0]))

    return


if __name__ == '__main__':
    args = parse_args()
    main(args)