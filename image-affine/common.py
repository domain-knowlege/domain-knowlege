import os
import torch
import csv
from deconfnet import CosineDeconf, DeconfNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


def load_model(method, model_name, load_data=True):
    if method == 'godin':
        if model_name == 'densenet':
            from models.densenet import DenseNet3x
            underlying_net = DenseNet3x(depth = 100, num_classes = 10).cuda()
            h = CosineDeconf(underlying_net.output_size, 10).cuda()
            model = DeconfNet(underlying_net, underlying_net.output_size, 10, h, False)
        elif model_name == 'resnet':
            from models.resnet import ResNet34x
            underlying_net = ResNet34x(num_classes=100).cuda()
            h = CosineDeconf(underlying_net.output_size, 100).cuda()
            model = DeconfNet(underlying_net, underlying_net.output_size, 100, h, False)
        elif model_name == 'wideresnet':
            from models.wideresnet import WideResNetx
            underlying_net = WideResNetx(depth = 28, num_classes = 200, widen_factor = 2, dropRate=0.3)
            h = CosineDeconf(underlying_net.output_size, 200)
            model = DeconfNet(underlying_net, underlying_net.output_size, 200, h, False)
        else:
            raise Exception('No such model: ' + model_name)
        if load_data:
            model.load_state_dict(torch.load(f'./snapshots/generalized_odin/{model_name}/checkpoint.pth')['deconf_net'])
            print('load model file:', f'./snapshots/generalized_odin/{model_name}/checkpoint.pth')
        model.cuda()
    else: # method is not godin
        if model_name == 'densenet':
            from models.densenet import DenseNet3
            model = DenseNet3(depth=100, num_classes=10)
            pre_trained_net  = os.path.join('./snapshots/mahalanobis', 'densenet_cifar10.pth')
        elif model_name == 'resnet':
            from models.resnet import ResNet34
            model = ResNet34(num_classes=100)
            pre_trained_net  = os.path.join('./snapshots/mahalanobis', 'resnet_cifar100.pth')
        elif model_name == 'wideresnet':
            from models.wideresnet import WideResNet
            model = WideResNet(depth=40, num_classes=200, widen_factor=2, dropRate=0.3)
            pre_trained_net = os.path.join('../TinyImagenet/models', 'wrn_baseline_epoch_99.pt')
        else:
            raise Exception('No such model: ' + model_name)
        if load_data:
            model.load_state_dict(torch.load(pre_trained_net))
            print('load model file: ' + pre_trained_net)
        model.cuda()

    return model


def load_dataset(dataset_name, data_root):
    if dataset_name == 'svhn':
        dataset = datasets.SVHN(root='./data/svhn', split='test', download=True, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]))
    elif dataset_name == 'fgsm':
        pass
    else:
        with open(os.path.join(data_root, f'{dataset_name}_rotation.pth'), 'rb') as f:
            data = torch.load(f)
        dataset = TensorDataset(data['imgs'], data['targets'])
    return dataset


def load_parameter(name, data_root):
    params_list = []
    with open(os.path.join(data_root, f'{name}_params.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            angle = float(line[0])
            tx, ty = int(line[1]), int(line[2])
            scale = float(line[3])
            shear_x, shear_y = float(line[4]), float(line[5])
            params_list.append([angle, (tx, ty), scale,(shear_x, shear_y)])
    
    return params_list


def load_revert(dataset_name, method, data_root):
    revert_list = []
    with open(os.path.join(data_root, f'{dataset_name}_{method}_revert.csv')) as f:
        reader = csv.reader(f)
        for line in reader:
            a,b,c = float(line[0]), float(line[1]), float(line[2])
            revert_list.append([a,b,c])
    return revert_list