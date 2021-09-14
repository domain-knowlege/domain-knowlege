from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical
import os
from PIL import Image
import json
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import time
from utils import *
import nltk
from GEP import *

np.set_printoptions(precision=2, suppress=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

op_list = ['+', '-', '*', '/']
digit_list = [str(i) for i in range(1, 10)]
sym_list =  ['UNK'] + digit_list + op_list
def sym2id(sym):
    return sym_list.index(sym)
def id2sym(idx):
    return sym_list[idx]

unk_idx = sym2id('UNK')
digit_idx_list = [sym2id(x) for x in digit_list]
op_idx_list = [sym2id(x) for x in op_list]


root_dir = './'
img_dir = root_dir + 'data/Handwritten_Math_Symbols/'
img_size = 45

def equal_res(preds, gts):
    return (np.abs(preds - gts)) < 1e-2

res_precision = 5


def eval_expr(preds, seq_len):
    res_preds = []
    expr_preds = []
    for i_pred, i_len in zip(preds, seq_len):
        i_pred = i_pred[:i_len]
        i_expr = ''.join([id2sym(idx) for idx in i_pred])
        try:
            i_res_pred = np.float(eval(i_expr))
        except:
            i_res_pred = np.inf
        res_preds.append(i_res_pred)
        expr_preds.append(i_expr)
    return expr_preds, res_preds

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
parser = GeneralizedEarley(grammar)

def eval_expr_fix(probs, seq_len):
    res_preds = []
    expr_preds = []
    # probs = torch.log(probs)
    _, preds = torch.max(probs, -1)
    for i_pred, i_len, i_prob in zip(preds, seq_len, probs):
        i_pred = i_pred[:i_len]
        i_prob = i_prob[:i_len]
        i_expr = ''.join([id2sym(idx) for idx in i_pred])
        try:
            i_res_pred = np.float(eval(i_expr))
        except:
            i_res_pred = np.inf
            # i_prob = np.concatenate([np.zeros([i_prob.size(0), 1]) + 1e-12, i_prob.detach().cpu()], axis=1)
            i_prob = i_prob.detach().cpu().numpy()
            print('[fix]', i_expr,)
            best_string, prob = parser.parse(i_prob)
            # print(best_string)
            best_string = ''.join([id2sym(int(x)) for x in best_string.split(' ')])
            print(' -> ', best_string)
            i_res_pred = np.float(eval(best_string))
            i_expr = best_string
            x = prob

        res_preds.append(i_res_pred)
        expr_preds.append(i_expr)
    return expr_preds, res_preds


def compute_rewards(preds, res, seq_len):
    expr_preds, res_preds = eval_expr(preds, seq_len)
    rewards = equal_res(res_preds, res)
    rewards = [1.0 if x else 0. for x in rewards]
    return np.array(rewards)