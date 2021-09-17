import argparse
import csv
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deconfnet import CosineDeconf, DeconfNet
from strategies.evolution import EvolutionModule
from utils.affine import tensor_affine
from utils.metric import AverageMeter, accuracy
from common import load_model, load_dataset, load_parameter


def get_cmd_args():
    parser = argparse.ArgumentParser(description='Pytorch Revert the Affined Dataset')
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
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    return parser.parse_args()


def get_scores(model, batch_affined_imgs, method):
    if method == 'godin':
        logits, h, _ = model(batch_affined_imgs)
        scores = h
        max_scores, _ = torch.max(scores, dim = 1)
    elif method == 'softmax':
        logits = model(batch_affined_imgs)
        softmax_ = F.softmax(logits, dim=1)
        max_scores = softmax_.max(dim=1)[0]
    else:
        exit()
    return max_scores


def get_affined(weights, images, population=4):
    batch_affined_imgs = None
    for index, image in enumerate(images):
        image_weights = weights[(population+1) * index: (population+1) * (index + 1)]
        # do not need to copy~!
        # image_temp = image.clone() 
        affined_imgs = []
        for _, weight in enumerate(image_weights):
            angle = weight[0].item()
            translate_x = weight[1].item()
            translate_y = weight[2].item()
            affined_img = tensor_affine(image, angle=angle, translate=[translate_x, translate_y])
            affined_imgs.append(affined_img)

        affined_imgs = torch.stack(affined_imgs)
        # print('affined:', affined_imgs.shape)

        if batch_affined_imgs is None:
            batch_affined_imgs = affined_imgs
        else:
            batch_affined_imgs = torch.cat((batch_affined_imgs, affined_imgs), 0)
            # print(batch_affined_imgs.shape)
    return batch_affined_imgs


# affine and get scores
def get_reward(weights, model, images, method='godin', population=4):
    batch_affined_imgs = get_affined(weights, images, population)
    max_scores = get_scores(model, batch_affined_imgs, method)
    return max_scores, batch_affined_imgs


def testData_rotate(model, data_loader, csv_writer, title = 'Testing'):
    model.eval()

    top1_original = AverageMeter()
    # top5_original = AverageMeter()

    top1 = AverageMeter()
    # top5 = AverageMeter()

    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)

    with torch.no_grad():
        for j, (images, targets) in enumerate(data_iter):
            data_iter.set_description(f'{title} , original acc: {top1_original.avg}, acc: {top1.avg}| Processing image batch {j + 1}/{num_batches}')
            images, targets = images.cuda(), targets.cuda()
            # take care of the last batch...
            input_length = images.size(0)
            if params_list is not None:
                batch_params = params_list[j*batch_size:j*batch_size+input_length]
                print(batch_params[0])

            if method == 'godin':
                logits, _, _ = model(images)
            else:
                logits = model(images)
            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))
            top1_original.update(prec1.item(), images.size(0))
            # top5_original.update(prec5.item(), images.size(0))

            population_size = 10
            partial_func = partial(get_reward, model=model, images=images, population=population_size)
            weights = torch.from_numpy(np.array([[0.0, 0.0, 0.0]] * input_length))
            weights = weights.cuda()
            es = EvolutionModule(
                list(weights), partial_func, population_size=population_size, sigma=1, 
                learning_rate=0.1, threadcount=15, cuda=True, reward_goal=0.5,
                consecutive_goal_stopping=10, batch_size=input_length
            )
            
            final_weights = es.run(100, print_step=10)
            for fw in final_weights:
                # print(final_weights)
                csv_writer.writerow([fw[0].item(), fw[1].item(), fw[2].item()])
                f.flush()

            max_scores, affined_imgs = get_reward(weights=final_weights, model=model, images=images, population=0)

            if method == 'godin':
                logits, _, _ = model(affined_imgs)
            else:
                logits = model(affined_imgs)
            prec1, prec5 = accuracy(logits.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            # top5.update(prec5.item(), images.size(0))

            results.extend(max_scores.data.cpu().numpy())

    data_iter.set_description(f'{title} , original acc: {top1_original.avg}, acc: {top1.avg}| Processing image batch {num_batches}/{num_batches}')
    data_iter.close()
    return np.array(results)

if __name__ == 'main':
    args = get_cmd_args()

    # set the parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    method = args.method
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_name = args.dataset_name

    # load model and dataset
    model = load_model(method, model_name)
    test_dataloader = DataLoader(load_dataset(dataset_name, './output'),
        batch_size=batch_size, shuffle=False)
    params_list = load_parameter(dataset_name, './output')

    with open(f'./output/{dataset_name}_{method}_revert.csv', 'a') as f:
        csv_writer = csv.writer(f)
        rotation_test_results = testData_rotate(model, test_dataloader, csv_writer, title = 'Reverting')
