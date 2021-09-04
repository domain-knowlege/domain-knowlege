import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm
import os
from models.densenet import DenseNet3
from torch.utils.data import DataLoader


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def save_adversarial(test_loader, model, epsilon, save_path):
    correct = 0
    outputs = []
    for data, target in tqdm(test_loader):
        data, target = data.cuda(), target.cuda()
        data.requires_grad = True

        output = model(data)
        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)

        outputs.append(perturbed_data.detach().cpu())
    
        # Check for success
        final_pred = output.max(1, keepdim=False)[1] # get the index of the max log-probability
        correct += torch.sum(final_pred == target)


    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader.dataset), final_acc))
    res = torch.cat(outputs)
    print('res:', res.shape)
    torch.save(res, save_path)

    # Return the accuracy and an adversarial example
    return final_acc


if __name__ == '__main__':
    model_name = 'densenet'
    name = 'cifar10'
    epsilon = 8/256
    rotation_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model = DenseNet3(depth=100, num_classes=10)
    pre_trained_net  = os.path.join('./snapshots/mahalanobis', f'{model_name}_{name}.pth')
    model.load_state_dict(torch.load(pre_trained_net))
    model.cuda()
    print('load model file: ' + pre_trained_net)

    test_loader = DataLoader(datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=rotation_transform), batch_size=128)
    save_adversarial(test_loader, model=model, epsilon=epsilon, save_path=f'./output/cifar10_densenet_fgsm.pth')