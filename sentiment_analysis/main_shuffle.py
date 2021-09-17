import argparse
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import EmpiricalCovariance
from torchtext.legacy import data, datasets
from torchtext.legacy.data import Dataset, Example, Iterator

import strategies.evolution as evo
from model import ClfGRU


# handle arguments
parser = argparse.ArgumentParser(description='Pytorch Evaluating the Effectiveness of Using Domain Knowledge')
parser.add_argument('--method', default='softmax', type=str,
                    help='ood detection method (softmax | mahalanobis | gmm)')
parser.add_argument('--batch-size', default=64, type=int,
                    help='batch size')
parser.add_argument('--population', default=50, type=int, help='population size of evolution algorithm')
parser.add_argument('--evolution_epoch', default=10, type=int, help='epoch number of evolution algorithm')
args = parser.parse_args()

warnings.simplefilter("ignore", UserWarning)


def evaluate(test_iter):
    model.eval()
    running_loss = 0
    num_examples = 0
    correct = 0

    for _, batch in enumerate(iter(test_iter)):
        inputs = batch.text.t()
        labels = batch.label - 1
        inputs, labels = inputs.cuda(), labels.cuda()

        _, logits = model(inputs)

        loss = F.cross_entropy(logits, labels, size_average=False)
        running_loss += loss.data.cpu().numpy()

        pred = logits.max(1)[1]
        correct += pred.eq(labels).sum().data.cpu().numpy()

        num_examples += inputs.shape[0]

    acc = correct / num_examples
    loss = running_loss / num_examples

    return acc, loss


def get_hidden_layer(test_iter):
    model.eval()

    hidden_list = []
    label_list = []
    with torch.no_grad():
        for _, batch in enumerate(iter(test_iter)):
            inputs = batch.text.t()
            labels = batch.label - 1
            inputs, labels = inputs.cuda(), labels.cuda()

            hidden, _ = model(inputs)
            hidden_list.append(hidden)
            label_list.append(labels)

    return torch.cat(hidden_list), torch.cat(label_list)


def test_example(index):
    text = test.examples[index].text
    print(text, get_score([text]).item())

    shuffled = text.copy()
    random.shuffle(shuffled)
    print(shuffled, get_score([shuffled]).item())

    es = evo.EvolutionModule(
        shuffled.copy(), get_score, population_size=args.population, sigma=1,
        learning_rate=0.1, threadcount=15, cuda=True, reward_goal=0.5,
        consecutive_goal_stopping=10
    )
    repaired = es.run(args.evolution_epoch)
    print(repaired, get_score([repaired]).item())


if __name__ == '__main__':
    # make dataset 
    TEXT = data.Field(pad_first=True, lower=True)
    LABEL = data.Field(sequential=False)
    train, test = datasets.TREC.splits(TEXT, LABEL, fine_grained=True)

    # build vocab
    TEXT.build_vocab(train, max_size=10000)
    LABEL.build_vocab(train, max_size=10000)
    print('vocab length (including special tokens):', len(TEXT.vocab))
    print('num labels:', len(LABEL.vocab))

    # make iterators
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=args.batch_size, repeat=False)

    # make model
    model = ClfGRU(50, TEXT).cuda()
    model.load_state_dict(torch.load('./snapshots/trec/baseline/model.dict'))

    acc, loss = evaluate(test_iter)
    print('test acc: {} \t| test loss: {}\n'.format(acc, loss))

    # define the score function used by searching
    if args.method == 'mahalanobis':
        hiddens, labels = get_hidden_layer(train_iter)
        hiddens_by_class = [hiddens[labels == i] for i in range(50)]
        sample_class_mean = [hidden.mean(axis=0) for hidden in hiddens_by_class]

        group_lasso = EmpiricalCovariance(assume_centered=False)
        X = torch.cat([hiddens_by_class[i] - sample_class_mean[i] for i in range(50)])
        group_lasso.fit(X.cpu())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()

        def get_score(sentences):
            assert type(sentences[0]) == list
            model.eval()

            inputs = TEXT.process(sentences)

            with torch.no_grad():
                inputs = inputs.t()
                inputs = inputs.cuda()

                hidden, logits = model(inputs)

            tmp_list = []
            for i in range(50):
                zero_f = hidden - sample_class_mean[i]
                # print(zero_f.shape) # (1, 512)
                term_gau = -0.5 * \
                    torch.mm(torch.mm(zero_f, temp_precision), zero_f.t()).diag()
                noise_gaussian_score = term_gau.view(-1, 1)
                tmp_list.append(noise_gaussian_score)

            scores_all = torch.cat(tmp_list, 1)
            score, _ = torch.max(scores_all, dim=1)

            return score.cpu()

    elif args.method == 'gmm':
        hiddens, labels = get_hidden_layer(train_iter)
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=200, covariance_type='diag', max_iter=1000, init_params='kmeans', reg_covar=1e-1, 
                            verbose=1, random_state=1234)
        gmm.fit(hiddens.cpu())


        def get_score(sentences):
            assert type(sentences[0]) == list
            model.eval()

            ###
            inputs = TEXT.process(sentences)
            ###

            with torch.no_grad():
                inputs = inputs.t()
                inputs = inputs.cuda()

                hidden, logits = model(inputs)

            ####
            # print(hidden.shape) #  (1, 512)
            score = gmm.score_samples(hidden.cpu())
            ####

            return torch.tensor(score)

    elif args.method == 'softmax':
        def get_score(sentences):
            assert type(sentences[0]) == list
            model.eval()

            inputs = TEXT.process(sentences)

            with torch.no_grad():
                inputs = inputs.t()
                inputs = inputs.cuda()

                _, logits = model(inputs)

                out = nn.functional.softmax(logits)
                # print(out.shape)
                score = out.max(axis=1)[0]

            return score

    # make ten examples
    print("==============")
    for _ in range(10):
        example_idx = random.randint(0, len(test) - 1)
        test_example(example_idx)
        print("==============")

    # generate the shuffled dataset
    shuffled_list = []
    rearranged_list = []
    skip = 0
    skip_and_normal = 0

    fields = {
        'text': ('text', TEXT),
        'label': ('label', LABEL),
    }

    for example in test:
        text = vars(example)['text']
        label = vars(example)['label']
        shuffled = text.copy()

        # keep 1/4 normal samples
        affined = False
        if random.random() > 0.25:
            random.shuffle(shuffled)
            affined = True
        shuffled_list.append(Example.fromdict(
            {'text': shuffled, 'label': label}, fields))

        if get_score([shuffled]).item() > 1:
            rearranged_list.append(Example.fromdict(
                {'text': shuffled.copy(), 'label': label}, fields))
            skip += 1
            skip_and_normal += (affined == False)
        else:
            es = evo.EvolutionModule(
                shuffled.copy(), get_score, population_size=args.population, sigma=1,
                learning_rate=0.1, threadcount=15, cuda=True, reward_goal=0.5,
                consecutive_goal_stopping=10
            )
            rearranged = es.run(args.evolution_epoch)
            rearranged_list.append(Example.fromdict(
                {'text': rearranged, 'label': label}, fields))

    print(f'skip: {skip_and_normal}/{skip}')

    # test the datasets
    test_shuffled = Dataset(shuffled_list, {'text': TEXT, 'label': LABEL})
    test_rearranged = Dataset(rearranged_list, {'text': TEXT, 'label': LABEL})

    shuffled_iterator = Iterator(
        test_shuffled, batch_size=args.batch_size, repeat=False, shuffle=False)
    rearranged_iterator = Iterator(
        test_rearranged, batch_size=args.batch_size, repeat=False, shuffle=False)


    acc, loss = evaluate(test_iter)
    print('test acc: {:.4f} | test loss: {:.4f}\n'.format(acc, loss))

    acc, loss = evaluate(shuffled_iterator)
    print('test acc: {:.4f} | test loss: {:.4f}\n'.format(acc, loss))

    acc, loss = evaluate(rearranged_iterator)
    print('test acc: {:.4f} | test loss: {:.4f}\n'.format(acc, loss))
