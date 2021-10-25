import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import conf

def label_to_index(word: str) -> torch.Tensor:
    # Return the position of the word in labels
    return torch.tensor(conf.LABELS.index(word))


def index_to_label(index) -> str:
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return conf.LABELS[index]


if __name__ == '__main__':
    word_start = "yes"
    index = label_to_index(word_start)
    word_recovered = index_to_label(index)

    print(word_start, "-->", index, "-->", word_recovered)
