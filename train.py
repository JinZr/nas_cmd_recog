from typing import List

import torch.nn as nn


from dataloader import training_dataloader
import conf
from test import test_during_training


def train(pbar, pbar_update, model: nn.Module, epoch: int, log_interval: int) -> List:
    
    model.train()

    device = conf.device
    criterion = nn.CrossEntropyLoss()
    dataloader = training_dataloader
    optim = conf.optimizer(
        params=model.parameters(),
        lr=conf.LEARNING_RATE,
        weight_decay=0.0001,
    )
    model = model.to(device=device)

    # loop over the dataset multiple times
    running_loss = []
    running_acc = []
    for batch_idx, data in enumerate(dataloader):
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

        running_loss.append(loss.item())
        running_acc.append(test_during_training(model=model, epoch=epoch))

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
        
        # update progress bar
        pbar.update(pbar_update)

    return running_loss, running_acc


if __name__ == '__main__':
    pass
