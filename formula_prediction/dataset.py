import enum
from utils import *
import random
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

class MathExprDataset(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None):
        super(MathExprDataset, self).__init__()
        
        self.split = split
        self.dataset = json.load(open('./data/expr_%s.json'%split))
        if numSamples:
            if randomSeed:
                random.seed(randomSeed)
                random.shuffle(self.dataset)
            self.dataset = self.dataset[:numSamples]
            
        for x in self.dataset:
            x['len'] = len(x['expr'])

        self.img_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (1,))])
    
    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        img_seq = []
        for img_path in sample['img_paths']:
            img = Image.open(img_dir+img_path).convert('L')
            #print(img.size, img.mode)
            img = self.img_transform(img)
            img_seq.append(img)
        del sample['img_paths']
        
        label_seq = [sym2id(sym) for sym in sample['expr']]
        sample['img_seq'] = img_seq
        sample['label_seq'] = label_seq
        sample['len'] = len(sample['expr'])

        res = eval(sample['expr'])
        res = round(res, res_precision)
        sample['res'] = res
        return sample
            
    
    def __len__(self):
        return len(self.dataset)

    def filter_by_len(self, max_len):
        self.dataset = [x for x in self.dataset if x['len'] <= max_len]


def MathExpr_collate(batch):
    max_len = np.max([x['len'] for x in batch])
    zero_img = torch.zeros_like(batch[0]['img_seq'][0])
    for sample in batch:
        sample['img_seq'] += [zero_img] * (max_len - sample['len'])
        sample['img_seq'] = torch.stack(sample['img_seq'])
        
        sample['label_seq'] += [sym2id('UNK')] * (max_len - sample['len'])
        sample['label_seq'] = torch.tensor(sample['label_seq'])
        
    batch = default_collate(batch)
    return batch


degree = 0
# translate_range = 0
choices = [90, 180, 270]

def MathExpr_collate_rotate(batch):
    max_len = np.max([x['len'] for x in batch])
    zero_img = torch.zeros_like(batch[0]['img_seq'][0])
    for sample in batch:
        sample['img_seq'] += [zero_img] * (max_len - sample['len'])
        # 对每一个图片分别进行变换
        # for i, img in enumerate(sample['img_seq']):
            # print('img:', img.shape) # [len, 45, 45]
        sample['img_seq'] = torch.stack(sample['img_seq'])

        if random.random() > 0.25:
            # 直接对seq_len张图片进行变换
            imgs = sample['img_seq']
            # print('imgs:', imgs)
            # save_image(imgs, './imgs.png')
            
            # 先计算参数
            base_degree = random.choice(choices)
            randAffine = transforms.RandomAffine(degrees=[-degree + base_degree, degree + base_degree])
            img_size = TF._get_image_size(imgs)
            angle, translations, scale, shear = randAffine.get_params(randAffine.degrees, randAffine.translate, randAffine.scale, randAffine.shear, img_size)
            sample['img_seq'] = TF.affine(imgs, angle, translations, scale, shear)
            # print('imgs2:', sample['img_seq'])
            # save_image(sample['img_seq'], './imgs2.png')
            # exit()
            # sample['affined'] = True
        
        sample['label_seq'] += [sym2id('UNK')] * (max_len - sample['len'])
        sample['label_seq'] = torch.tensor(sample['label_seq'])
    
    batch = default_collate(batch)
    return batch