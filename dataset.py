import torch
import torchaudio


class SpeechCmdDataset(torch.utils.data.Dataset):

    def __init__(self, subset: str):
        super(SpeechCmdDataset, self).__init__()

        assert subset in ['training', 'validation', 'testing']
            
        self.base_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root='./dataset',
            download=True,
            subset=subset
        )
        self.mfcc = torchaudio.transforms.MFCC()

    def __getitem__(self, index):
        waveform, sample_rate, label, spkr_id, utter_num = self.base_dataset.__getitem__(
            index)
        return {
            'mfcc': self.mfcc.forward(waveform=waveform),
            'label': label,
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
    dataset = SpeechCmdDataset('training')
    print(dataset.__getitem__(0))