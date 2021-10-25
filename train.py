from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer

from model import CmdRecogNetwork
from dataloader import training_dataloader
import conf

def train(
    feature_dim: int,
    network_depth: int,
    num_class: int,
    network_width: List[int],
    network_context_size: List[int]
):
    device = conf.device
    criterion = nn.CrossEntropyLoss()
    dataloader = training_dataloader
    model = CmdRecogNetwork(
        feature_dim=feature_dim,
        network_depth=network_depth,
        network_width=network_width,
        network_context_size=network_context_size,
        num_class=num_class
    ).to(device=device)
    optim = conf.optimizer(
        params=model.parameters(), 
        lr=conf.LEARNING_RATE,
        weight_decay=0.0001,
    )
    # loop over the dataset multiple times
    for epoch in range(5):
        running_loss = 0.0
        for data in dataloader:
            inputs, labels = data['mfcc'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels.shape)
            # print(inputs.shape)    
            # zero the parameter gradients
            optim.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            # print(outputs.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()
    
            running_loss += loss.item()
    
        print('Loss: {}'.format(running_loss))
    print('Finished Training')

if __name__ == '__main__':
    import random
    train(
        feature_dim=conf.FEATURE_DIM,
        network_depth=conf.DEPTH,
        num_class=conf.NUM_CLASS,
        network_width=random.choices(conf.WIDTH_SPACE, k=conf.DEPTH),
        network_context_size=random.choices(conf.CONTEXT_SPACE, k=conf.DEPTH),
    )