from typing import List
import torch
from matplotlib import pyplot as plt

import conf

def label_to_index(word: str) -> torch.Tensor:
    # Return the position of the word in labels
    return torch.tensor(conf.LABELS.index(word))


def index_to_label(index) -> str:
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return conf.LABELS[index]

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def draw_curve_fig(
    data: List[float],
    path: str
):
    x_index = list(range(len(data)))
    plt.plot(x_index, data)
    plt.savefig(path)

if __name__ == '__main__':
    word_start = "yes"
    index = label_to_index(word_start)
    word_recovered = index_to_label(index)

    print(word_start, "-->", index, "-->", word_recovered)
