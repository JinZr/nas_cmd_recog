import os
import sys
import time
from typing import List

from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from numpy import lib

import conf

def print(obj):
    sys.stdout.write(obj)

def draw_fig(label: str, data: List[float]):
    x_index = list(range(0, len(data)))
    plt.plot(x_index, data)
    plt.xlabel("iter")
    plt.ylabel(label)

    x_major_locator = MultipleLocator(250)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    if not os.path.exists(conf.TDNN_LOG_PATH): os.makedirs(conf.TDNN_LOG_PATH)
    plt.savefig('{}/{}_{}.png'.format(conf.TDNN_LOG_PATH, label, time.time()))
    plt.cla()
    plt.clf()
    plt.close()


def draw_loss_acc_fig(acc: List[float], loss: List[float]):
    draw_fig(label="acc", data=acc)
    draw_fig(label="loss", data=loss)

# def draw_composed_loss_acc_fig(acc: List[float], loss: List[float]):
