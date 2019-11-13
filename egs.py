from kaldi_python_io import  Nnet3EgsReader
import torch

class Nnet3EgsDataset(torch.utils.data.Dataset):
    def __init__(self, egs_files):
        self.features = []
        self.speaker_ids = []
        for egs_file in egs_files:
            for key, data in Nnet3EgsReader(egs_file):
                chunk_features = data[0]['matrix']
                speaker_id = data[1]['matrix'][0][0][0] 
                self.features.append(torch.from_numpy(chunk_features).T)
                self.speaker_ids.append(speaker_id)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.speaker_ids[idx]
