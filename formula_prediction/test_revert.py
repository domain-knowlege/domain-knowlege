from utils import *
from dataset import *
from NN_AOG import NNAOG
from diagnosis import ExprTree
from strategies.evolution import EvolutionModule
from functools import partial
from torchvision.utils import save_image

import torch
import numpy as np
import os
import argparse


def train_model(opt):
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.manual_seed)
    train_set = MathExprDataset('train', numSamples=int(10000*opt.data_used), randomSeed=777)
    test_set = MathExprDataset('test')
    print('train:', len(train_set), '  test:', len(test_set))
    model = NNAOG().to(device)
    if opt.pretrain:
        model.sym_net.load_state_dict(torch.load(opt.pretrain))
    train(model, train_set, test_set, opt)


def evaluate(model, dataloader, fix=False):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        img_seq = img_seq.to(device)
        label_seq = label_seq.to(device)
        masked_probs = model(img_seq)
        selected_probs, preds = torch.max(masked_probs, -1)
        # selected_probs = torch.log(selected_probs+1e-12)
        if fix:
            expr_preds, res_preds =eval_expr_fix(masked_probs, seq_len)
        else:
            expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    print('Grammar Error: %.2f'%(np.isinf(res_pred_all).mean()*100))
    acc = equal_res(res_pred_all, res_all).mean()

    
    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc

rules = [
    "GAMMA -> expression",
    "expression -> term | expression '+' term | expression '-' term",
    "term -> factor | term '*' factor | term '/' factor"
]
num_rule = "factor -> " + ' | '.join(["'%s'"%str(x) for x in range(1, 10)])
rules.append(num_rule)

symbol_index = {x:sym2id(x) for x in sym_list}
grammar_rules = grammarutils.get_pcfg(rules, index=True, mapping=symbol_index)
print('\n'.join(rules))
# print(grammar_rules)
grammar = nltk.CFG.fromstring(grammar_rules)
gep_parser = GeneralizedEarley(grammar)


def evaluate_revert(model, dataloader, fix=False, method='softmax'):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    with torch.no_grad():
        for sample in tqdm(dataloader):
            img_seq = sample['img_seq']
            label_seq = sample['label_seq']
            res = sample['res']
            seq_len = sample['len']
            expr = sample['expr']
            img_seq = img_seq.cuda()
            label_seq = label_seq.cuda()
            input_length = img_seq.size(0)

            scores, affined_img_seq = get_reward(weights=torch.from_numpy(np.array([[0.0, 0.0, 0.0]] * input_length)), model=model, images=img_seq, seq_len=seq_len, population=0, method=method)

            population_size = 10
            partial_func = partial(get_reward, model=model, images=img_seq, seq_len=seq_len, population=population_size, method=method)
            weights = torch.from_numpy(np.array([[0.0, 0.0, 0.0]] * input_length))
            weights = weights.cuda()
            es = EvolutionModule(
                list(weights), partial_func, population_size=population_size, sigma=1, 
                learning_rate=0.1, threadcount=15, cuda=True, reward_goal=0.5,
                consecutive_goal_stopping=10, batch_size=input_length
            )
            final_weights = es.run(100, print_step=10)
            _, affined_img_seq = get_reward(weights=final_weights, model=model, images=img_seq, seq_len=seq_len, population=0, method=method)
            masked_probs = model(affined_img_seq)
            selected_probs, preds = torch.max(masked_probs, -1)
            # print('shape:', img_seq.shape)
            # [batch, seq_len, channel, size, size] [64, 7, 1, 45, 45]
            if fix:
                expr_preds, res_preds = eval_expr_fix(masked_probs, seq_len)
            else:
                expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)
            
            res_pred_all.append(res_preds)
            res_all.append(res)
            expr_pred_all.extend(expr_preds)
            expr_all.extend(expr)
            print('acc:', equal_res(np.concatenate(res_pred_all, axis=0), np.concatenate(res_all, axis=0)).mean())
            print('sym_acc:', np.mean([x == y for x,y in zip(''.join(expr_pred_all), ''.join(expr_all))]))

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    print('Grammar Error: %.2f'%(np.isinf(res_pred_all).mean()*100))
    acc = equal_res(res_pred_all, res_all).mean()

    
    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc


def get_reward(weights, model, images, seq_len, population=0, method='softmax'):
    with torch.no_grad():
        batch_affined_imgs = None
        for index, img_seq in enumerate(images):
            image_weights = weights[(population+1) * index: (population+1) * (index + 1)]
            affined_imgs = []
            for _, weight in enumerate(image_weights):
                angle = weight[0].item() * 180
                translate_x = weight[1].item() * 0.1
                translate_y = weight[2].item() * 0.1
                affined_img = TF.affine(img_seq, angle=angle, translate=[translate_x, translate_y], scale=1, shear=0)
                affined_imgs.append(affined_img)

            affined_imgs = torch.stack(affined_imgs)

            if batch_affined_imgs is None:
                batch_affined_imgs = affined_imgs
            else:
                batch_affined_imgs = torch.cat((batch_affined_imgs, affined_imgs), 0)

        if method == 'softmax':
            logits = model(batch_affined_imgs)
            softmax_ = F.log_softmax(logits, dim=2)
            max_scores = softmax_.max(dim=2)[0]
            # max_scores: [64, seq_len]
            max_scores = max_scores.sum(dim=1)
            # max_scores: [64]
        else:
            max_scores = []
            probs = model(batch_affined_imgs)
            _, preds = torch.max(probs, -1)
            for i, (i_pred, i_prob) in enumerate(zip(preds, probs)):
                i_len = seq_len[i//(population + 1)]
                i_pred = i_pred[:i_len]
                i_prob = i_prob[:i_len]
                i_expr = ''.join([id2sym(idx) for idx in i_pred])
                i_prob = i_prob.detach().cpu().numpy()
                best_string, prob = gep_parser.parse(i_prob)
                best_string = ''.join([id2sym(int(x)) for x in best_string.split(' ')])
                max_scores.append(prob)
            max_scores = torch.tensor(max_scores)
    return max_scores, batch_affined_imgs


def train(model, train_set, test_set, opt):
    num_workers = opt.num_workers
    batch_size = opt.batch_size

    criterion = nn.NLLLoss(ignore_index=-1)

    
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                      shuffle=True, num_workers=num_workers, collate_fn=MathExpr_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, collate_fn=MathExpr_collate_rotate)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)


    acc, sym_acc = evaluate(model, eval_dataloader, fix=False)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
    print()

    acc, sym_acc = evaluate_revert(model, eval_dataloader, fix=True, method='softmax')
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))

    acc, sym_acc = evaluate(model, eval_dataloader, fix=True)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))

    return


parser = argparse.ArgumentParser()
# Model
parser.add_argument('--mode', default='BS', type=str, help='choose mode. BS or RL or MAPO' )
parser.add_argument('--nstep', default=5, type=int, help='number of steps of backsearching')
parser.add_argument('--pretrain', default=None, type=str, help='pretrained symbol net')
# Dataloader
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--batch_size', default=64, type=int)
# seed
parser.add_argument('--random_seed', default=123, type=int, help="numpy random seed")
parser.add_argument('--manual_seed', default=17, type=int, help="torch manual seed")
# Run
parser.add_argument('--lr', default=1e-5, type=float, help="learning rate")
parser.add_argument('--decay', default=0.99, type=float, help="reward decay")
parser.add_argument('--num_epochs', default=5, type=int, help="number of epochs")
parser.add_argument('--n_epochs_per_eval', default=1, type=int, help="test every n epochs")
parser.add_argument('--output_dir', default='output', type=str, help="output directory")

opt = parser.parse_args()
print(opt)
train_model(opt)