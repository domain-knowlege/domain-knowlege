import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import affine

degree = 15
translate_range = 0
choices = [90, 180, 270]

output_path = './output'


def save_to_file(_dataset, save_path):
    print('saving to:', save_path)

    # collect the inputs, save the params
    inputs = []
    for item in tqdm(_dataset):
        inputs.append(item)

    # save imgs and targets
    imgs, targets = zip(*inputs)
    imgs = torch.stack(imgs)
    targets = torch.tensor(targets)

    with open(save_path, 'wb') as f:
        obj = {'imgs': imgs, 'targets': targets}
        torch.save(obj, f)

    print('imgs:', tuple(imgs.size()))
    print('targets:', tuple(targets.size()))


if __name__ == '__main__':
    name = 'cifar10'
    rotation_transform = transforms.Compose([
        affine.CustomizedRandomAffine(degree, choices, translate_range, f'{output_path}/{name}_params.csv'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    save_to_file(datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=rotation_transform),
                 save_path=f'{output_path}/{name}_rotation.pth')

    name = 'cifar100'
    rotation_transform = transforms.Compose([
        affine.CustomizedRandomAffine(degree, choices, translate_range, f'{output_path}/{name}_params.csv'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    save_to_file(datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=rotation_transform),
                 save_path=f'{output_path}/{name}_rotation.pth')

    name = 'tinyimagenet'
    rotation_transform = transforms.Compose([
        affine.CustomizedRandomAffine(degree, choices, translate_range, f'{output_path}/{name}_params.csv'),
        transforms.ToTensor(),
        transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255, 66.7/255)),
    ])
    save_to_file(datasets.ImageFolder(root="./data/tiny-imagenet/tiny-imagenet-200/train", transform=rotation_transform),
                 save_path=f'{output_path}/{name}_rotation.pth')
