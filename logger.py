import os
import sys
import time
from typing import List

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from numpy import lib

def print(obj):
    sys.stdout.write(obj)

def draw_fig(label: str, data: List[float]):
    x_index = list(range(0, len(data)))
    plt.plot(x_index, data)
    plt.xlabel("iter")
    plt.ylabel(label)

    x_major_locator = MultipleLocator(25)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.savefig('./tdnn_log/{}.png'.format(time.time()))


def draw_loss_acc_fig(acc: List[float], loss: List[float]):
    plt.plot()