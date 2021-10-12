import torch

import dataset
import conf

def __get_dataloader__(
    subset: str, 
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    
    assert subset in ['training', 'validation', 'testing']

    return torch.utils.data.DataLoader(
        dataset.SpeechCmdDataset(subset=subset),
        batch_size=batch_size,
        shuffle=shuffle
    )

training_dataloader = __get_dataloader__(
    subset='training', 
    batch_size=conf.BATCH_SIZE,
    shuffle=True
)
testing_dataloader = __get_dataloader__(
    subset='testing', 
    batch_size=conf.BATCH_SIZE,
    shuffle=False
)
validation_dataloader = __get_dataloader__(
    subset='validation', 
    batch_size=conf.BATCH_SIZE,
    shuffle=False
)

if __name__ == '__main__':
