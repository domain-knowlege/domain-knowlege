import csv
import random
import os

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class CustomizedRandomAffine(torch.nn.Module):
    def __init__(self, degree, choices, translate_range, savefile=None):
        super().__init__()
        self.degree = degree
        self.choices = choices
        self.translate_range = translate_range
        self.savefile = savefile

    def forward(self, img):
        base_degree = random.choice(self.choices)
        randAffine = transforms.RandomAffine(degrees=[-self.degree + base_degree, self.degree + base_degree],
                                             translate=[self.translate_range, self.translate_range], scale=None, shear=None)
        img_size = TF._get_image_size(img)
        angle, translations, scale, shear = randAffine.get_params(randAffine.degrees, randAffine.translate, randAffine.scale, randAffine.shear, img_size)
        img = TF.affine(img, angle, translations, scale, shear)

        # optional save the parameters
        if self.savefile is not None:
            with open(self.savefile, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([angle, translations[0], translations[1], scale, shear[0], shear[1]])
        
        return img


def tensor_affine(img, angle, translate=[0,0], scale=1):
    angle = angle * 180
    translate_range = [i * 0.1 for i in translate]
    affined_img = TF.affine(img, angle, translate=translate_range, scale=1, shear=0)
    return affined_img
