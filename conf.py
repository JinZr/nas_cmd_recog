import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam

# Feature
FEATURE_DIM = 40
MAX_LEN = 81
LABELS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
          'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

# Training
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCH_NUM = 20
LOG_INTERVAL = 20
NUM_CLASS = len(LABELS)

# Arch
DEPTH = 4
WIDTH_SPACE = [32, 64, 128, 256, 512]
CONTEXT_SPACE = [3, 4, 5, 6, 7, 8, 9]
