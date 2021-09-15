import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.affine import CustomizedRandomAffine
from utils.metric import AverageMeter, accuracy
from common import load_model


parser = argparse.ArgumentParser(description='Baseline Model Training: CIFAR10/CIFAR100/TinyImagenet')

# training parameters
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
# experiment options
parser.add_argument('--model-name', default='densenet', type=str,
                    help='model architecture (densenet | resnet | wideresnet)')
parser.add_argument('--dataset-name', default='cifar10', type=str,
                    help='dataset')
# device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# fine tuning option
parser.add_argument('--sample-data', dest='sample_data', action='store_true',
                    help='use part data')
# special
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='whether to resume')
parser.add_argument('--test', dest='test', action='store_true',
                    help='whether to test')


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # 此处设置训练的参数
    degree = 15
    translate_range = 0.2
    choices = [0, 90, 180, 270]

    # Data loading code
    if args.dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    elif args.dataset_name == 'cifar100':
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    elif args.dataset_name == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    else:
        raise Exception(f'{args.dataset_name} does not exsit.')

    transform_train = transforms.Compose([
        CustomizedRandomAffine(degree, choices, translate_range),
        transforms.ToTensor(),
        normalize
        ])
    transform_test = transforms.Compose([
        CustomizedRandomAffine(degree, [90, 180, 270], translate_range),
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    if args.dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR10('./data/cifar10', train=False, transform=transform_test)
    elif args.dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data/cifar100', train=False, download=False, transform=transform_test)
    elif args.dataset_name == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(root="~/tiny-imagenet/tiny-imagenet-200/train", transform=transform_train)
        val_dataset = datasets.ImageFolder(root="~/tiny-imagenet/tiny-imagenet-200/val", transform=transform_test)
    else:
        return

    # for fine tuning，only take 1000 samples
    if args.sample_data:
        train_indices = torch.randperm(len(train_dataset))[:1000]
        train_dataset = Subset(train_dataset, train_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    print('data samples:', len(train_loader.dataset))

    # load the model
    model = load_model(method='baseline', model_name=args.model_name, load_data=args.resume)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    # for training on multiple GPUs. 
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    if args.test: # test only
        prec1 = validate(val_loader, model, criterion, 0)
        exit()


    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)

    best_prec1 = 0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        # default: checkpoint.pth.tar and model_best.pth.tar
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "snapshots/retrain/%s/"%(args.model_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'snapshots/retrain/%s/'%(args.model_name) + 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
