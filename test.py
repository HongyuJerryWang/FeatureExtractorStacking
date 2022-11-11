import os
import sys
import torch
import pickle

import numpy as np

from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader

# select fine-tuning algorithm
DATASETS_DIR = 'datasets/tsa/'

DATASETS = list(filter(lambda dataset: dataset in os.listdir(DATASETS_DIR), ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower', 'traffic_sign', 'mscoco', 'mnist', 'cifar10', 'cifar100', 'CropDisease', 'EuroSAT', 'ISIC', 'ChestX', 'Food101']))

all_accs = dict()
criterion = torch.nn.CrossEntropyLoss()
METHODS = ['FES', 'ConFES', 'ReFES']

# candidate hyperparameter values for ReFES
LAMBDA1POOL = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
LAMBDA2POOL = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# classifier for ConFES
# can be used for FES and ReFES
# because ConFES is a generalisation of FES
class ConFESClassifier(torch.nn.Module):

    def __init__(self, sizes, strides, dilations, num_backbones):
        super(ConFESClassifier, self).__init__()
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(num_backbones, num_backbones, size, stride=stride, dilation=dilation, groups=num_backbones, bias=False) for size, stride, dilation in zip(sizes, strides, dilations)])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.shape[0]
        model_size = x.shape[1]
        checkpoint_size = x.shape[2]
        class_size = x.shape[3]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size * class_size, model_size, checkpoint_size)
        x = (x - x.min()) / (x.max() - x.min())
        for conv in self.convs:
            x = conv(x)
        x = x.view(batch_size, class_size, model_size, -1)
        x = self.avgpool(x)
        return x.squeeze(-1).squeeze(-1)

# cached data loading pipeline
class CachedLogitsDataset(Dataset):

    def __init__(self, dataset, root_dir=DATASETS_DIR):
        self.dataset_dir = root_dir + dataset

    def __len__(self):
        return len(os.listdir(self.dataset_dir))

    def __getitem__(self, run):
        if torch.is_tensor(run):
            run = run.tolist()
        return torch.load(f'{self.dataset_dir}/{run}.pt')

for dataset in DATASETS:

    print(dataset, flush=True)
    all_accs[dataset] = {method: [] for method in METHODS}
    cached_loader = DataLoader(CachedLogitsDataset(dataset), batch_size=None, shuffle=False)

    # for every episode
    for cached_data in tqdm(cached_loader):

        context_logits = cached_data['context_logits'].float()
        context_labels = cached_data['context_labels'].long()
        target_logits = cached_data['target_logits'].float()
        target_labels = cached_data['target_labels'].long()

        # strictly one-shot problem
        # cross-validation not possible
        # cannot perform FES
        # fallback to weighing un-fine-tuned backbones
        if len(context_logits.shape) == 3:
            meta_classifier = torch.nn.Conv1d(8, 1, 1, bias=False)
            meta_classifier.weight.data.fill_(1e-3)
            optimizer = torch.optim.LBFGS(meta_classifier.parameters(), line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = criterion(meta_classifier(context_logits).squeeze(-2).squeeze(-2), context_labels) + 1e-3 * (meta_classifier.weight ** 2).sum()
                loss.backward()
                return loss
            optimizer.step(closure)
            acc = torch.eq(meta_classifier(target_logits).squeeze(-2).argmax(1), target_labels).float().mean().item()
            all_accs[dataset]['FES'].append(acc)
            all_accs[dataset]['ConFES'].append(acc)
            all_accs[dataset]['ReFES'].append(acc)
        else:
            # perform FES
            FES = ConFESClassifier([context_logits.shape[2]], [1], [1], 8)
            for conv in FES.convs:
                conv.weight.data.fill_(1e-3)
            optimizer = torch.optim.LBFGS(FES.parameters(), line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = criterion(FES(context_logits), context_labels) + 1e-3 * (FES.convs[0].weight ** 2).sum()
                loss.backward()
                return loss
            optimizer.step(closure)
            acc = torch.eq(FES(target_logits).argmax(1), target_labels).float().mean().item()
            all_accs[dataset][f'FES'].append(acc)

            # perform ConFES
            if context_logits.shape[2] == 41:
                ConFES = ConFESClassifier([9, 9], [4, 1], [1, 1], 8)
            elif context_logits.shape[2] == 7:
                ConFES = ConFESClassifier([3, 3], [2, 1], [1, 1], 8)
            for conv in ConFES.convs:
                conv.weight.data.fill_((1e-3) ** (1 / 2))
            optimizer = torch.optim.LBFGS(ConFES.parameters(), line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = criterion(ConFES(context_logits), context_labels) + 1e-3 * (ConFES.convs[0].weight ** 2 + ConFES.convs[1].weight ** 2).sum()
                loss.backward()
                return loss
            optimizer.step(closure)
            acc = torch.eq(ConFES(target_logits).argmax(1), target_labels).float().mean().item()
            all_accs[dataset][f'ConFES'].append(acc)

            # perform ReFES hyperparameter selection
            ReFES = ConFESClassifier([context_logits.shape[2]], [1], [1], 8)

            # determine the split point for the two cross-validation folds
            monotonicity = True
            for split_point in range(1, len(context_labels)):
                if context_labels[split_point-1] > context_labels[split_point]:
                    monotonicity = False
                    break
            if monotonicity:
                split_point = (len(context_labels) + ((context_labels[0] + 1) % 2)) // 2

            # perform grid search on regularisation hyperparameters
            best_correct = -1
            for lambda_1 in LAMBDA1POOL:
                for lambda_2 in LAMBDA2POOL:
                    correct_count = 0
                    for fold in [0, 1]:

                        # split training data for cross-validation
                        if fold == 0:
                            cv_train_logits = context_logits[:split_point]
                            cv_train_labels = context_labels[:split_point]
                            cv_test_logits = context_logits[split_point:]
                            cv_test_labels = context_labels[split_point:]
                        else:
                            cv_train_logits = context_logits[split_point:]
                            cv_train_labels = context_labels[split_point:]
                            cv_test_logits = context_logits[:split_point]
                            cv_test_labels = context_labels[:split_point]
                        ReFES.convs[0].weight.data.fill_(1e-3)
                        optimizer = torch.optim.LBFGS(ReFES.parameters(), line_search_fn='strong_wolfe')
                        def closure():
                            optimizer.zero_grad()
                            loss = criterion(ReFES(cv_train_logits), cv_train_labels)
                            loss += lambda_1 * ReFES.convs[0].weight.abs().sum()
                            loss += lambda_2 * (ReFES.convs[0].weight[:, :, 1:] - ReFES.convs[0].weight[:, :, :-1]).abs().sum()
                            loss.backward()
                            return loss
                        optimizer.step(closure)

                        # accumulate the number of correct predictions
                        correct_count += torch.eq(ReFES(cv_test_logits).argmax(1), cv_test_labels).int().sum().item()

                    # update best hyperparameter choice
                    # stronger regularisation is favoured in tie-breaks
                    if correct_count >= best_correct:
                        best_lambda_1 = lambda_1
                        best_lambda_2 = lambda_2
                        best_correct = correct_count

            # perform ReFES with selected hyperparameters
            ReFES.convs[0].weight.data.fill_(1e-3)
            optimizer = torch.optim.LBFGS(ReFES.parameters(), line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = criterion(ReFES(context_logits), context_labels)
                loss += best_lambda_1 * ReFES.convs[0].weight.abs().sum()
                loss += best_lambda_2 * (ReFES.convs[0].weight[:, :, 1:] - ReFES.convs[0].weight[:, :, :-1]).abs().sum()
                loss.backward()
                return loss
            optimizer.step(closure)
            acc = torch.eq(ReFES(target_logits).argmax(1), target_labels).float().mean().item()
            all_accs[dataset][f'ReFES'].append(acc)

# produce result table
rows = []
for dataset in DATASETS:
    row = [dataset]
    for method in METHODS:
        acc = np.array(all_accs[dataset][method]) * 100
        mean_acc = acc.mean()
        conf = (1.96 * acc.std()) / np.sqrt(len(acc))
        row.append(f"{mean_acc:0.1f} +- {conf:0.1f}")
    rows.append(row)
table = tabulate(rows, headers=['model \\ data'] + METHODS, floatfmt=".1f")
print(table)
print("\n")
