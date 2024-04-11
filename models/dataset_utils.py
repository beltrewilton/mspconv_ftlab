import sys
sys.path.append("../")
from models.data_processor import MSPDataProcessor, MSP_PATH, ROOT, AUDIO_SEGMENTS, AUDIO_PARTS
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from models.architecture import Timem
from vad.vad_lab import VAD
from pathlib import Path

vad = VAD(minmax=[-100, 100], mapping="OCC")

timem = Timem(verbose=True)

terms = ['RESENTMENT', 'SATISFACTION', 'REPROACH', 'LOVE', 'NEUTRAL', 'PRIDE', 'ANGER', 'HOPE', 'GLOATING', 'PITY', 'HATE', 'HAPPY FOR?', 'DISLIKING', 'JOY']

class MSPDataset(Dataset):
    def __init__(self, input_features: dict, verbose: bool = False):
        super(MSPDataset, self).__init__()
        self.input_features = input_features
        self.verbose = verbose

    def seqCollate(self, batch):
        """
         [Wilton] overriding collate_fn -> calling default_collate at the end.
        :param batch:
        :return:
        """
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        # getlen_label = lambda x: x[1].shape[0]
        # max_seqlen_label = max(map(getlen_label, batch))

        def pad_(x):
            input, label = x[0], x[1]
            dif = max_seqlen - input.size(0)
            if dif > 0:
                input = F.pad(input, (0, dif), "constant", 0)
            # dif = max_seqlen_label - label.size(0)
            # if dif > 0:
            #     label = F.pad(label, (0, dif), "constant", 41) #TODO: space token
            return input, label

        batch = list(map(pad_, batch))
        return default_collate(batch)

    # def __insert_zeros_between_elements(self, label):
    #     # Insert zeros between elements
    #     # zeros = torch.zeros(label.size(0) - 1, dtype=label.dtype, device=label.device)
    #     # result = torch.zeros(2 * label.size(0) - 1, dtype=label.dtype, device=label.device)
    #
    #
    #     zeros = torch.tensor([41], dtype=label.dtype, device=label.device).repeat(label.size(0) - 1)
    #     result = torch.tensor([41], dtype=label.dtype, device=label.device).repeat(2 * label.size(0) - 1)
    #     result[::2] = label
    #     result[1::2] = zeros
    #     return result

    def __len__(self):
        return len(self.input_features['inputs'])

    def __getitem__(self, idx):
        """
        lee del disco la parte segmentada de parte de audio
        :param idx:
        :return: tensor que representa el wave, tensor label
        """
        if self.verbose:
            print(f"idx:{idx} {self.input_features['inputs'][idx]}")
        input  = Path(self.input_features['inputs'][idx]).name
        part = f"{AUDIO_SEGMENTS}/{input}" # TODO: to device????
        waveform, _ = torchaudio.load(str(part), normalize=True)
        waveform = waveform.squeeze()
        label = self.input_features['labels'][idx]
        label_id = terms.index(label)
        label_id = torch.tensor(label_id, dtype=torch.long)
        return waveform, label_id # waveform is also float32 by default, cause of normalize=True


if __name__ == "__main__":
    datapros = MSPDataProcessor(msp_path=MSP_PATH, chunk_size=15, split="Test", verbose=True)
    test_dataset = MSPDataset(input_features=datapros.load_input_features())
    for inputs, labels in test_dataset:
        print()