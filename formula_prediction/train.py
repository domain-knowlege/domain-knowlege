import argparse

from utils import *
from dataset import *
from NN_AOG import NNAOG

import torch
import numpy as np


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


def train(model, train_set, test_set, opt):
    num_workers = opt.num_workers
    batch_size = opt.batch_size
    lr = opt.lr
    reward_decay = opt.decay
    num_epochs = opt.num_epochs
    n_epochs_per_eval = opt.n_epochs_per_eval
    buffer_weight = 0.5

    criterion = nn.NLLLoss(ignore_index=-1)

    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=lr)
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    reward_moving_average = None
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, collate_fn=MathExpr_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers, collate_fn=MathExpr_collate)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    stats_path = os.path.join(opt.output_dir, "stats_%s_%.2f_%d.json"%('mode', opt.data_used, opt.pretrain != None))
    stats = {
            'train_accs': [],
            'val_accs': []
    }
        
    acc, sym_acc = evaluate(model, eval_dataloader, fix=True)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
    print()

    iter_counter = -1
    for epoch in range(num_epochs):
        since = time.time()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        model.train()

        for sample in train_dataloader:
            iter_counter += 1

            img_seq = sample['img_seq']
            label_seq = sample['label_seq']
            res = sample['res']
            seq_len = sample['len']
            expr = sample['expr']

            img_seq = img_seq.to(device)
            label_seq = label_seq.to(device)
            max_len = img_seq.shape[1]
            masked_probs = model(img_seq)

            selected_probs, preds = torch.max(masked_probs, -1)
            selected_probs = torch.log(selected_probs+1e-20)
            masked_probs = torch.log(masked_probs + 1e-20)
            probs = masked_probs

            loss = criterion(probs.reshape((-1, probs.shape[-1])), label_seq.reshape((-1,)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            selected_probs2, preds2 = torch.max(masked_probs, -1)
            selected_probs2 = torch.log(selected_probs2+1e-12)
            expr_preds, res_pred_all = eval_expr(preds2.data.cpu().numpy(), seq_len)
            acc = equal_res(np.asarray(res_pred_all), np.asarray(res)).mean()
          
            expr_pred_all = ''.join(expr_preds)
            expr_all = ''.join(expr)
            sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            
            acc = round(acc, 4)
            sym_acc = round(sym_acc, 4)
            stats['train_accs'].append((iter_counter, acc, sym_acc))

        print("Average reward:", reward_moving_average)
            
        if (epoch+1) % n_epochs_per_eval == 0:
            acc, sym_acc = evaluate(model, eval_dataloader)
            print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
            if acc > best_acc:
                best_acc = acc
                best_model_wts = deepcopy(model.state_dict())

            acc = round(acc, 4)
            sym_acc = round(sym_acc, 4)
            stats['val_accs'].append((iter_counter, acc, sym_acc))
            json.dump(stats, open(stats_path, 'w'))

                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print(flush=True)

    acc, sym_acc = evaluate(model, eval_dataloader)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))

    if acc > best_acc:
        best_acc = acc
        best_model_wts = deepcopy(model.state_dict())
    torch.save(best_model_wts, './pretrain-sym_net/model.pt')
    print('Save to: ./pretrain-sym_net/model.pt')
    print('Best val acc: {:2f}'.format(100*best_acc))
    return


parser = argparse.ArgumentParser()
# Model
parser.add_argument('--pretrain', default=None, type=str, help='pretrained symbol net')
# Dataloader
parser.add_argument('--data_used', default=1.00, type=float, help='percentage of data used')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--batch_size', default=64, type=int)
# seed
parser.add_argument('--random_seed', default=123, type=int, help="numpy random seed")
parser.add_argument('--manual_seed', default=17, type=int, help="torch manual seed")
# Run
parser.add_argument('--lr', default=1e-3, type=float, help="learning rate")
parser.add_argument('--decay', default=0.99, type=float, help="reward decay")
parser.add_argument('--num_epochs', default=10, type=int, help="number of epochs")
parser.add_argument('--n_epochs_per_eval', default=1, type=int, help="test every n epochs")
parser.add_argument('--output_dir', default='output', type=str, help="output directory")

opt = parser.parse_args()
print(opt)
train_model(opt)