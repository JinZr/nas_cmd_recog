from typing import List
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

import conf
import dataloader
from model import CmdRecogNetwork
import train, test

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

if __name__ == '__main__':
    model = random_init_model(
        feature_dim=conf.FEATURE_DIM,
        network_depth=conf.DEPTH,
        num_class=conf.NUM_CLASS,
        network_width=random.choices(conf.WIDTH_SPACE, k=conf.DEPTH),
        network_context_size=random.choices(conf.CONTEXT_SPACE, k=conf.DEPTH),
    )
    losses = []
    with tqdm(total=conf.EPOCH_NUM) as pbar:
        for epoch in range(1, conf.EPOCH_NUM + 1):
            training_losses = \
                train.train(pbar, model, epoch, conf.LOG_INTERVAL)
            test.test(pbar, model, epoch)
            losses += training_losses
            # scheduler.step()
