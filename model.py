from typing import List
from numpy import timedelta64

import torch
from torch._C import dtype
import torch.nn as nn
from torch.nn.modules import pooling

import tdnn

class CmdRecogNetwork(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        network_depth: int,
        network_width: List[int],
        network_context_size: List[int],
        num_class: int
    ):
        super(CmdRecogNetwork, self).__init__()

        assert len(network_width) == len(network_context_size)  \
            and len(network_context_size) == network_depth  \
            and len(network_width) == network_depth  \
            and feature_dim > 0  \
            and num_class > 1  \
            and network_depth > 0
        
        self.module_list = []
        
        # NOTE: construct TDNN layers
        for width, context_size, index in zip(network_width, network_context_size, range(network_depth)):
            print('index: {}, width: {}, context_size: {}'.format(index, width, context_size))
            # NOTE: first layer
            if len(self.module_list) == 0:
                self.module_list.append(
                    tdnn.TDNN(input_dim=feature_dim, output_dim=width, context_size=context_size)
                )
            # NOTE: last layer
            elif len(self.module_list) == network_depth - 1:
                self.module_list.append(
                    tdnn.TDNN(input_dim=last_width, output_dim=num_class, context_size=context_size)
                )
            # NOTE: intermediate layer
            else:
                self.module_list.append(
                    tdnn.TDNN(input_dim=last_width, output_dim=width, context_size=context_size)
                )
            self.module_list.append(nn.ReLU(inplace=True))
            last_width, last_context_size = width, context_size

        self.module_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.module_list.append(nn.Flatten(start_dim=1))
        self.model = nn.Sequential(
            *self.module_list
        )
        self.hidden_layer_input_size = self.__calc_linear_layer_dim__()
        print('hidden_layer_input_size: ', self.hidden_layer_input_size)
        self.module_list.append(nn.Linear(in_features=self.hidden_layer_input_size, out_features=1024))
        self.module_list.append(nn.Linear(in_features=1024, out_features=num_class))
        self.model = nn.Sequential(
            *self.module_list
        )


    def forward(self, x):
        result = self.model(x)
        # print(result.size())
        return result
    
    def __calc_linear_layer_dim__(self) -> int:
        import time
        import numpy as np
        import conf
        begin_time = time.time()
        random_num = np.random.rand(conf.BATCH_SIZE, conf.MAX_LEN, conf.FEATURE_DIM)
        random_num_tensor = torch.from_numpy(random_num).float()
        result = self.model(random_num_tensor)
        _, dim = result.shape
        print('calc time (second): ', time.time() - begin_time)
        return dim

if __name__ == '__main__':
    import random

    import numpy as np

    import conf

    model = CmdRecogNetwork(
        feature_dim=conf.FEATURE_DIM,
        num_class=conf.NUM_CLASS,
        network_depth=conf.DEPTH,
        network_context_size=random.choices(conf.CONTEXT_SPACE, k=conf.DEPTH),
        network_width=random.choices(conf.WIDTH_SPACE, k=conf.DEPTH)
    )
    random_num = np.random.rand(conf.BATCH_SIZE, 512, 40)
    random_num_tensor = torch.from_numpy(random_num).float()
    print(random_num_tensor.dtype)
    result = model.forward(
        random_num_tensor
    )
    print(result.shape)