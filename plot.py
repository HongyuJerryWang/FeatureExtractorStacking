import os
import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from tabulate import tabulate
from matplotlib.colors import TwoSlopeNorm
from torch.utils.data import Dataset, DataLoader

# select the episode used to produce heat maps
CACHEFILENAME = 'datasets/flute/traffic_sign/0.pt'

criterion = torch.nn.CrossEntropyLoss()
row_labels = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quick_draw', 'fungi', 'vgg_flower']

# ReFES hyperparameter candidates
LAMBDA1POOL = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
LAMBDA2POOL = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

# ConFES and ReFES classifier
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

cached_data = torch.load(CACHEFILENAME) 
context_logits = cached_data['context_logits'].float()
context_labels = cached_data['context_labels'].long()
target_logits = cached_data['target_logits'].float()
target_labels = cached_data['target_labels'].long()
os.makedirs('heatmaps', exist_ok=True)

if len(context_logits.shape) == 3:
    print('Selected episode is a strict one-shot problem, unable to perform cross-validation required by ConFES and ReFES, please select a different episode.', flush=True)

else:
    # perform FES
    FES = ConFESClassifier([context_logits.shape[2]], [1], [1], 8)
    FES.convs[0].weight.data.fill_(1e-3)
    optimizer = torch.optim.LBFGS(FES.parameters(), line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = criterion(FES(context_logits), context_labels) + 1e-3 * (FES.convs[0].weight ** 2).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    acc = torch.eq(FES(target_logits).argmax(1), target_labels).float().mean().item()
    print('FES accuracy:', acc, flush=True)

    FES_kernel = FES.convs[0].weight.data.detach().cpu().numpy().squeeze()
    plt.figure(figsize = (max(context_logits.shape[2] / 4, 3.25), 2))
    plt.imshow(FES_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
    plt.colorbar(fraction=0.046*8/context_logits.shape[2], pad=0.04)
    plt.yticks(list(range(8)), row_labels)
    plt.savefig(f'heatmaps/FES.png')

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
    print('ConFES accuracy:', acc, flush=True)

    if context_logits.shape[2] == 41:

        ConFES_depthwise_kernel = ConFES.convs[0].weight.data.detach().cpu().numpy().squeeze()
        ConFES_global_kernel = ConFES.convs[1].weight.data.detach().cpu().numpy().squeeze()
        product = ConFES_depthwise_kernel.reshape((8, 1, 9)) * ConFES_global_kernel.reshape((8, 9, 1))
        ConFES_expanded_kernel = np.zeros((8, 41))
        for i in range(9):
            ConFES_expanded_kernel[:, (i * 4):(i * 4 + 9)] += product[:, i]

        # plot ConFES depthwise kernel
        plt.figure(figsize = (3.75, 2))
        plt.imshow(ConFES_depthwise_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(list(range(9)))
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_depthwise.png')

        # plot ConFES global kernel
        plt.figure(figsize = (3.75, 2))
        plt.imshow(ConFES_global_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(list(range(9)))
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_global.png')

        # plot ConFES expanded kernel
        plt.figure(figsize = (10.25, 2))
        plt.imshow(ConFES_expanded_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046*8/41, pad=0.04)
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_expanded.png')

    elif context_logits.shape[2] == 7:

        ConFES_depthwise_kernel = ConFES.convs[0].weight.data.detach().cpu().numpy().squeeze()
        ConFES_global_kernel = ConFES.convs[1].weight.data.detach().cpu().numpy().squeeze()
        product = ConFES_depthwise_kernel.reshape((8, 1, 3)) * ConFES_global_kernel.reshape((8, 3, 1))
        ConFES_expanded_kernel = np.zeros((8, 7))
        for i in range(3):
            ConFES_expanded_kernel[:, (i * 2):(i * 2 + 3)] += product[:, i]

        # plot ConFES depthwise kernel
        plt.figure(figsize = (2.25, 2))
        plt.imshow(ConFES_depthwise_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(list(range(3)))
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_depthwise.png')

        # plot ConFES global kernel
        plt.figure(figsize = (2.25, 2))
        plt.imshow(ConFES_global_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.xticks(list(range(3)))
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_global.png')

        # plot ConFES expanded kernel
        plt.figure(figsize = (3.25, 2))
        plt.imshow(ConFES_expanded_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
        plt.colorbar(fraction=0.046*8/7, pad=0.04)
        plt.yticks(list(range(8)), row_labels)
        plt.savefig(f'heatmaps/ConFES_expanded.png')

    # perform ReFES
    ReFES = ConFESClassifier([context_logits.shape[2]], [1], [1], 8)
    monotonicity = True
    for split_point in range(1, len(context_labels)):
        if context_labels[split_point-1] > context_labels[split_point]:
            monotonicity = False
            break
    if monotonicity:
        split_point = (len(context_labels) + ((context_labels[0] + 1) % 2)) // 2
    best_correct = -1
    for lambda_1 in LAMBDA1POOL:
        for lambda_2 in LAMBDA2POOL:
            correct_count = 0
            for fold in [0, 1]:
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
                correct_count += torch.eq(ReFES(cv_test_logits).argmax(1), cv_test_labels).int().sum().item()
            if correct_count >= best_correct:
                best_lambda_1 = lambda_1
                best_lambda_2 = lambda_2
                best_correct = correct_count
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
    print('ReFES accuracy:', acc, flush=True)

    # plot ReFES kernel
    ReFES_kernel = ReFES.convs[0].weight.data.detach().cpu().numpy().squeeze()
    plt.figure(figsize = (max(context_logits.shape[2] / 4, 3.25), 2))
    plt.imshow(ReFES_kernel, norm=TwoSlopeNorm(0), cmap=plt.cm.seismic)
    plt.colorbar(fraction=0.046*8/context_logits.shape[2], pad=0.04)
    plt.yticks(list(range(8)), row_labels)
    plt.savefig(f'heatmaps/ReFES.png')
