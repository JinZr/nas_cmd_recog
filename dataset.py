import torch
from torch.nn.functional import pad
import torch.nn.functional as F
import torchaudio

import conf
import utils

class SpeechCmdDataset(torch.utils.data.Dataset):

    def __init__(self, subset: str):
        super(SpeechCmdDataset, self).__init__()

        assert subset in ['training', 'validation', 'testing']
            
        self.base_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root='./dataset',
            download=True,
            subset=subset
        )
        self.mfcc = None

    def __getitem__(self, index):
        waveform, sample_rate, label, spkr_id, utter_num = self.base_dataset.__getitem__(
            index)
        if self.mfcc != None:
            mfcc = self.mfcc.forward(waveform=waveform).squeeze()
        else:
            self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate).to(conf.device)
            mfcc = self.mfcc.forward(waveform=waveform).squeeze()
        mfcc_len = mfcc.shape[-1]
        pad_mfcc = torch.nn.ConstantPad1d(
            padding=(0, conf.MAX_LEN - mfcc_len),
            value=0.
        )(mfcc).T
        return {
            'mfcc': pad_mfcc,
            'label': F.one_hot(
                utils.label_to_index(label), 
                num_classes=conf.NUM_CLASS
                ).double(),
            'spkr_id': spkr_id,
            'utter_num': utter_num
        }

    def __len__(self):
        return len(self.base_dataset)


def get_dataset(subset: str):
    if subset not in ['training', 'validation', 'testing']:
        raise NotImplementedError()
    return torchaudio.datasets.SPEECHCOMMANDS(
        root='./dataset',
        download=True,
        subset=subset
    )

if __name__ == '__main__':
    import numpy as np
    dataset = SpeechCmdDataset('training')
    # labels = max(set(np.shape(datapoint['mfcc'])[-1] for datapoint in dataset))
    # print(labels)
    # dataset = SpeechCmdDataset('testing')
    # labels = max(set(np.shape(datapoint['mfcc'])[-1] for datapoint in dataset))
    # print(labels)
    # dataset = SpeechCmdDataset('validation')
    # labels = max(set(np.shape(datapoint['mfcc'])[-1] for datapoint in dataset))
    # print(labels)
    # print(np.shape(dataset.__getitem__(0)['mfcc']))