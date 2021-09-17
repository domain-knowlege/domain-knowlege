import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import ConcatDataset, Dataset, Subset
from tqdm import tqdm

from deconfnet import CosineDeconf, DeconfNet
from utils.affine import tensor_affine
from utils.metric import AverageMeter, accuracy
from common import load_model, load_dataset, load_parameter, load_revert
from revert_rotation import get_reward, get_affined, get_scores


def get_cmd_args():
    parser = argparse.ArgumentParser(description='Pytorch Evaluating the Effectiveness of Using Domain Knowledge')
    
    # General arguments
    parser.add_argument('--gpu-id', default='4', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--method', default='softmax', type=str,
                        help='ood detection method (softmax | godin)')
    # Model arguments
    parser.add_argument('--model-name', default='densenet', type=str,
                        help='model architecture (densenet | resnet | wideresnet)')
    # Data arguments
    parser.add_argument('--dataset-name', default='cifar10', type=str,
                        help='dataset')
    parser.add_argument('--data-root', default='./output', type=str,
                        help='data root')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    return parser.parse_args()


def testData_revert(model, data_loader, method, title = 'Testing'):
    model.eval()

    top1_original = AverageMeter()

    top1 = AverageMeter()

    no_affine = 0
    reject = 0
    reject2 = 0

    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)

    with torch.no_grad():
        for j, (images, targets) in enumerate(data_iter):
            data_iter.set_description(f'original: {top1_original.avg:.2f}, acc: {top1.avg:.2f}, ood_reject: {reject2}, skip_affine: {no_affine}, reject: {reject} | batch {j + 1}/{num_batches}')
            images, targets = images.cuda(), targets.cuda()
            current_size = images.size(0)

            # the baseline with ood detector
            max_scores = get_scores(model, images, method)
            reject_index2 = max_scores < 0.1 # ood detection tau
            images2 = images[reject_index2.logical_not()]
            targets2 = targets[reject_index2.logical_not()]
            reject2 += reject_index2.sum()
            if method == 'godin':
                logits, _, _ = model(images2)
            else:
                logits = model(images2)
            prec1, prec5 = accuracy(logits.data, targets2.data, topk=(1, 5))
            top1_original.update(prec1.item(), images2.size(0))

            ### the main method ###
            # line 1-2，S(x) > tau
            not_affine_index = max_scores > 0.4 # tau1
            no_affine += not_affine_index.sum()

            # line 4-5
            batch_params = revert_list[j*batch_size:j*batch_size+current_size]
            weights = torch.tensor(batch_params)
            _, affined_imgs = get_reward(weights=weights, model=model, images=images, method=method, population=0)
            # apply no_affine
            affined_imgs[not_affine_index] = images[not_affine_index]

            # line 7-8，evaluate again
            max_scores = get_scores(model, affined_imgs, method)
            reject_index = max_scores < 0.1 # tau2
            affined_imgs = affined_imgs[reject_index.logical_not()]
            targets = targets[reject_index.logical_not()]
            reject += reject_index.sum()

            # compute accuracy after revert
            if method == 'godin':
                logits, _, _ = model(affined_imgs)
            else:
                logits = model(affined_imgs)
            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            # top5.update(prec5.item(), images.size(0))

    data_iter.set_description(f'{title} , original acc: {top1_original.avg}, acc: {top1.avg}| Processing image batch {num_batches}/{num_batches}')
    data_iter.close()
    return np.array(results)


if __name__ == '__main__':
    args = get_cmd_args()

    # set the parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    method = args.method
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset_name
    data_root = args.data_root

    model = load_model(method, model_name)

    test_dataset = load_dataset(dataset_name, data_root)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False)

    # load the revert parameters
    revert_list = load_revert(dataset_name, method, data_root)
    testData_revert(model, test_dataloader, method)
