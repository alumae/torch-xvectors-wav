import random
import torch
import kaldiio
import logging
import itertools

from tqdm import tqdm

class ChunkDataset(torch.utils.data.Dataset):
    
    def __init__(self, outer, is_train, num_batches=None, proportion=None):
        super(ChunkDataset, self).__init__()
        if num_batches is not None and proportion is not None:
            raise Exception("Both num_batches or proportion should not be specified")
        self.outer = outer
        self.is_train = True
        if num_batches is not None:
            self.num_batches = num_batches
        elif proportion is not None:
            avg_chunk_length = (outer.min_length + outer.max_length) / 2
            self.num_batches = int(outer.total_num_frames / avg_chunk_length / outer.batch_size * proportion + 1)                    
        else:
            raise Exception("Either num_batches or proportion should be specified")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        chunk_length = random.randint(self.outer.min_length, self.outer.max_length)
        features_tensor = torch.zeros((self.outer.batch_size, self.outer.feat_dim, chunk_length))
        speakers_tensor = torch.zeros((self.outer.batch_size), dtype=torch.long)
        for i in range(self.outer.batch_size):
            while True:
                spk = random.choice(self.outer.speakers)
                if self.is_train:
                    utts = list(filter(lambda utt: self.outer.utt2num_frames[utt] >= chunk_length, self.outer.train_spk2utt[spk]))
                else:
                    utts = list(filter(lambda utt: self.outer.utt2num_frames[utt] >= chunk_length, self.outer.valid_spk2utt[spk]))
                if len(utts) == 0:
                    logging.debug(f"Speaker {spk} doesn't have any train utterances with length at least {chunk_length} frames, picking other speaker")
                else:
                    break

            utt = random.choice(utts)
            utt_features = self.get_utt_features(utt)
            start_pos = random.randint(0, utt_features.shape[0] - chunk_length)

            utt_features = utt_features[start_pos:start_pos+chunk_length]
            features_tensor[i] = torch.from_numpy(utt_features).T
            speakers_tensor[i] = self.outer.speakers2id[spk]
        return features_tensor, speakers_tensor


    def get_utt_features(self, utt):
        return self.outer.feats[utt]
        
class RandomChunkSubsetDatasetFactory:

    def __init__(self, datadir, min_length=200, max_length=400, num_valid_utts=100, batch_size=64):
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size

        utt2spk = {}
        
        for l in open(f"{datadir}/utt2spk"):
            ss = l.split()
            utt2spk[ss[0]] = ss[1]
            

        valid_utts = random.sample(utt2spk.keys(), num_valid_utts)
        #train_utt2spk = {key: utt2spk[key] for key in utt2spk if key not in valid_utts}

        self.train_spk2utt = {}
        self.valid_spk2utt = {}
        
        for utt, spk in utt2spk.items():
            if utt in valid_utts:
                self.valid_spk2utt.setdefault(spk, []).append(utt)
            else:    
                self.train_spk2utt.setdefault(spk, []).append(utt)
        self.speakers = list(sorted(set(utt2spk.values())))
        self.speakers2id = {speaker: i for i, speaker in enumerate(self.speakers)}
        
        self.feats = kaldiio.load_scp(f"{datadir}/feats.scp")
        
        self.feat_dim = self.feats[valid_utts[0]].shape[1]
        self.num_outputs = len(self.speakers)
        self.utt2num_frames = {}
        for l in open(f"{datadir}/utt2num_frames"):
            ss = l.split()
            self.utt2num_frames[ss[0]] = int(ss[1])
        self.total_num_frames = sum(self.utt2num_frames.values())

    def get_chunk_dataset(self, is_train, num_batches=None, proportion=None):
        return ChunkDataset(self, is_train=True, num_batches=num_batches, proportion=proportion)

    def get_train_dataset(self, num_batches=None, proportion=None):
        return self.get_chunk_dataset(is_train=True, num_batches=num_batches, proportion=proportion)

    def get_valid_dataset(self, num_batches):
        return self.get_chunk_dataset(is_train=False, num_batches=num_batches)



if __name__ == "__main__":
    g = RandomChunkSubsetDatasetFactory("../sre19/data/vox_tunisian_8k_combined_no_sil/", batch_size=512)
    train_dataset = g.get_train_dataset(1000)
    valid_dataset = g.get_valid_dataset(3)
    #breakpoint()
    dl = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=None, num_workers=3)
    it = iter(dl)
    for i in tqdm(range(100)):
        _ = next(it)
