import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam

TDNN_LOG_PATH = "./tdnn_log"

# Feature
FEATURE_DIM = 40
MAX_LEN = 81
LABELS = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', \
          'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

# Training TDNN
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCH_NUM = 3
LOG_INTERVAL = 20
NUM_CLASS = len(LABELS)

# TDNN Arch
DEPTH = 4
WIDTH_SPACE = [32, 64, 128, 256, 512]
CONTEXT_SPACE = [5, 6, 7, 8, 9]

# Training RL Agent
RL_EPOCH_NUM = 50
RL_ALPHA = 5e-3        # learning rate
RL_BATCH_SIZE = 3      # how many episodes we want to pack into an epoch
RL_HIDDEN_SIZE = 256   # number of hidden nodes we have in our dnn
RL_BETA = 0.1          # the entropy bonus multiplier
RL_INPUT_SIZE = 5
RL_ACTION_SPACE = 5
RL_NUM_STEPS = 8       # num_steps = max_layer * 2 
                       # four tdnn layer * 2 h-parameters (context and dim)
RL_GAMMA = 0.99