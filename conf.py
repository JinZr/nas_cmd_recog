import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training
BATCH_SIZE = 256

# Arch
DEPTH = 4
WIDTH_SPACE = [32, 64, 128, 256, 512]
CONTEXT_SPACE = [3, 4, 5, 6, 7, 8, 9]
