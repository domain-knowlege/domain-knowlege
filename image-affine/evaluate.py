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
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
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
    top5_original = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    no_affine = 0
    no_affine_replace = 0
    reject = 0
    reject2 = 0
    scores = []

    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)

    with torch.no_grad():
        for j, (images, targets) in enumerate(data_iter):
            data_iter.set_description(f'original: {top1_original.avg:.2f}, acc: {top1.avg:.2f}, reject2: {reject2}, no_affine: {no_affine_replace}/{no_affine}, reject: {reject} | batch {j + 1}/{num_batches}')
            images, targets = images.cuda(), targets.cuda()
            current_size = images.size(0)

            # hack: 到svhn了，给个提示
            # if j * batch_size > 10000 and (j-1) * batch_size <= 10000:
            #     print('to svhn')
            # 先计算恢复之前的数据
            # 需要过滤一波，把svhn检测出来
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
            # top5_original.update(prec5.item(), images2.size(0))

            #### 到我们的方法了
            scores.append(max_scores)
            # 第1.2步，S(x) > tau
            # print(max_scores)
            not_affine_index = max_scores > 0.3904 # tau1 0.3226, 0.3368, 0.3570, 0.3904
            no_affine += not_affine_index.sum()
            # 统计no_affine_replace的
            batch_replace = replace_index[j*batch_size:j*batch_size+current_size]
            assert not_affine_index.shape == batch_replace.shape
            no_affine_replace += not_affine_index.logical_and(batch_replace.cuda()).sum()

            # 第4.5步，恢复
            batch_params = revert_list[j*batch_size:j*batch_size+current_size]
            weights = torch.tensor(batch_params)
            _, affined_imgs = get_reward(weights=weights, model=model, images=images, method=method, population=0)
            # 打补丁：满足1条件的不需要变换
            affined_imgs[not_affine_index] = images[not_affine_index]

            # 第7.8步，再次评估
            max_scores = get_scores(model, affined_imgs, method)
            # scores.append(max_scores)
            # print(max_scores)
            reject_index = max_scores < 0.1 # tau2
            affined_imgs = affined_imgs[reject_index.logical_not()]
            targets = targets[reject_index.logical_not()]
            reject += reject_index.sum()

            # 计算恢复之后的数据
            if method == 'godin':
                logits, _, _ = model(affined_imgs)
            else:
                logits = model(affined_imgs)
            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            # top5.update(prec5.item(), images.size(0))

    torch.save(torch.cat(scores), 'scores.pt')
    data_iter.set_description(f'{title} , original acc: {top1_original.avg}, acc: {top1.avg}| Processing image batch {num_batches}/{num_batches}')
    data_iter.close()
    return np.array(results)


class MyDataset(Dataset):

    def __init__(self, dataset1, dataset2, replace_index):
        assert len(dataset1) == len(dataset2)
        assert len(dataset1[0]) == len(dataset2[0])
        assert dataset1[0][0].shape == dataset2[0][0].shape
        # assert dataset1[0][1].shape == dataset2[0][1].shape
        # print(dataset1)
        # print(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.replace_index = replace_index

    def __getitem__(self, index):
        # print(self.dataset1[index][1].shape)
        # print(self.dataset2[index][1].shape)
        if self.replace_index[index]:
            return self.dataset2[index]
        else:
            return self.dataset1[index][0], self.dataset1[index][1].item()

    def __len__(self):
        return len(self.dataset1)


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

    # 搞一个混合的数据集，混一点svhn进去
    raw_dataset = load_dataset(dataset_name, data_root)
    # print(dataset_name, ':', len(raw_dataset))
    # 替换旋转的样本为普通样本
    param = load_parameter(dataset_name, data_root)
    # 4-way的replace
    # replace_index = torch.tensor([p[0] for p in param])
    # replace_index = torch.logical_and(replace_index > -30, replace_index < 30)
    # 3-way的replace
    replace_index = torch.rand([10000]) < 0.25
    # print(replace_index.sum())
    dataset2 = datasets.ImageFolder(root="~/tiny-imagenet/tiny-imagenet-200/val", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255)),
    ]))
    replaced_dataset = MyDataset(raw_dataset, dataset2, replace_index)
    # print(replaced_dataset.__getitem__(0))


    # 加载，切块，合并
    # svhn_used = 500
    # svhn_dataset = load_dataset('svhn', data_root)
    # svhn_dataset = Subset(svhn_dataset, torch.arange(svhn_used))

    # dataset_all = ConcatDataset([raw_dataset, svhn_dataset])
    # print(dataset_name + '+svhn', ':', len(dataset_all))

    test_dataloader = DataLoader(
        replaced_dataset,
        batch_size=batch_size, shuffle=False)

    # revert_list同样要搞svhn进来
    revert_list = load_revert(dataset_name, method, data_root)
    # svhn_revert_list = load_revert('svhn', method, data_root)
    # revert_list.extend(svhn_revert_list[:svhn_used])
    # print('revert_list:', len(revert_list))

    testData_revert(model, test_dataloader, method)