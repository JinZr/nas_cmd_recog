import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import testing_dataloader
import utils
import conf

def test(pbar, pbar_update, model: nn.Module, epoch: int):
    model.eval()
    correct = 0
    device = conf.device
    for batch_idx, data in enumerate(testing_dataloader):
        inputs, labels = data['mfcc'], data['label']
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)

        pred = utils.get_likely_index(output)
        labels = utils.get_likely_index(labels)
        correct += utils.number_of_correct(pred, labels)

        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(testing_dataloader.dataset)} ({100. * correct / len(testing_dataloader.dataset):.0f}%)\n")
    return correct / len(testing_dataloader.dataset)

def test_during_training(model: nn.Module, epoch: int):
    model.eval()
    correct = 0
    device = conf.device
    for batch_idx, data in enumerate(testing_dataloader):
        inputs, labels = data['mfcc'], data['label']
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)

        pred = utils.get_likely_index(output)
        labels = utils.get_likely_index(labels)
        correct += utils.number_of_correct(pred, labels)

    return correct / len(testing_dataloader.dataset)

