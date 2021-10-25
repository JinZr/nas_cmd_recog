import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import conf
import utils

def predict(model, tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(conf.device)
    tensor = model(tensor.unsqueeze(0))
    tensor = utils.get_likely_index(tensor)
    tensor = utils.index_to_label(tensor)
    return tensor
