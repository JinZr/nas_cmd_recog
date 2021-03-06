from typing import List
import random
import os

import torch.nn as nn

from tqdm import tqdm

import conf
from dataloader import training_dataloader, testing_dataloader
from model import CmdRecogNetwork
import train, test
import logger

def random_init_model(
    feature_dim: int,
    network_depth: int,
    num_class: int,
    network_width: List[int],
    network_context_size: List[int]
) -> nn.Module:
    return CmdRecogNetwork(
        feature_dim=feature_dim,
        network_depth=network_depth,
        network_width=network_width,
        network_context_size=network_context_size,
        num_class=num_class
    )

def tdnn_main():
    if not os.path.exists('./dataset'): os.makedirs('./dataset')
    if not os.path.exists('./plots'): os.makedirs('./plots')

    model = random_init_model(
        feature_dim=conf.FEATURE_DIM,
        network_depth=conf.DEPTH,
        num_class=conf.NUM_CLASS,
        network_width=random.choices(conf.WIDTH_SPACE, k=conf.DEPTH),
        network_context_size=random.choices(conf.CONTEXT_SPACE, k=conf.DEPTH),
    )
    losses = []
    acc = []
    acc_per_epoch = []
    pbar_update = 1 / (len(training_dataloader) + len(testing_dataloader))
    with tqdm(total=conf.EPOCH_NUM) as pbar:
        for epoch in range(1, conf.EPOCH_NUM + 1):
            training_losses, training_acc = \
                train.train(pbar, pbar_update, model, epoch, conf.LOG_INTERVAL)
            test_acc = \
                test.test(pbar, pbar_update, model, epoch)
            losses += training_losses
            acc += training_acc
            acc_per_epoch += [test_acc]
            # scheduler.step()
    logger.draw_loss_acc_fig(
        acc=acc,
        loss=losses
    )

if __name__ == '__main__':
    tdnn_main()
