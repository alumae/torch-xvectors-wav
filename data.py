import random
import torch
import kaldiio
import logging
import itertools
import numpy as np

import logging
from tqdm import tqdm


class ChunkDataset(torch.utils.data.Dataset):

    def __init__(self, outer, is_train, num_batches=None, proportion=None):
        super(ChunkDataset, self).__init__()
        if num_batches is not None and proportion is not None:
            raise Exception(
                "Both num_batches or proportion should not be specified")
        self.outer = outer
        self.is_train = True
        if num_batches is not None:
            self.num_batches = num_batches
        elif proportion is not None:
            avg_chunk_length = (outer.min_length + outer.max_length) / 2
            self.num_batches = int(
                outer.total_num_frames / avg_chunk_length / outer.batch_size * proportion + 1)
        else:
            raise Exception(
                "Either num_batches or proportion should be specified")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        chunk_length = random.randint(
            self.outer.min_length, self.outer.max_length)
        features_tensor = torch.zeros(
            (self.outer.batch_size, self.outer.feat_dim, chunk_length))
        labels_tensor = torch.zeros(
            (self.outer.batch_size), dtype=torch.long)

        if self.is_train:
            # pre-filter utts to those with length >= chunk_length
            train_label2utt_filtered = \
                {label: list(filter(lambda utt: self.outer.utt2num_frames[utt] >= chunk_length, utts))
                 for label, utts in self.outer.train_label2utt.items()}

        for i in range(self.outer.batch_size):
            while True:
                label = random.choice(self.outer.labels)
                if self.is_train:
                    utts = train_label2utt_filtered[label]
                else:
                    utts = list(filter(
                        lambda utt: self.outer.utt2num_frames[utt] >= chunk_length, self.outer.valid_label2utt[label]))
                if len(utts) == 0:
                    logging.debug(
                        f"Label {label} doesn't have any train utterances with length at least {chunk_length} frames, picking other speaker")
                else:
                    break

            utt = random.choice(utts)
            utt_features = self.get_utt_features(utt)
            start_pos = random.randint(0, utt_features.shape[0] - chunk_length)

            utt_features = utt_features[start_pos:start_pos+chunk_length]
            if isinstance(utt_features, np.ndarray):
                features_tensor[i] = torch.from_numpy(utt_features).T
            else:
                features_tensor[i] = utt_features.T
            labels_tensor[i] = self.outer.label2id[label]
        return features_tensor, labels_tensor

    def get_utt_features(self, utt):
        return self.outer.feats[utt]


class RandomChunkSubsetDatasetFactory:

    def __init__(self, datadir, min_length=200, max_length=400, num_valid_utts=100, batch_size=64, valid_utt_list_file=None, label_file="utt2lang"):
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size

        utt2label = {}

        for l in open(f"{datadir}/{label_file}"):
            ss = l.split()
            utt2label[ss[0]] = ss[1]

        if valid_utt_list_file is None:
            valid_utts = random.sample(utt2label.keys(), num_valid_utts)
        else:
            logging.info(f"Reading validation utterance list from {valid_utt_list_file}")
            valid_utts = [l.split()[0] for l in open(valid_utt_list_file)]
        #train_utt2spk = {key: utt2spk[key] for key in utt2spk if key not in valid_utts}

        self.train_label2utt = {}
        self.valid_label2utt = {}

        for utt, label in utt2label.items():
            if utt in valid_utts:
                self.valid_label2utt.setdefault(label, []).append(utt)
            else:
                self.train_label2utt.setdefault(label, []).append(utt)
        self.labels = list(sorted(set(utt2label.values())))
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        self.feats = kaldiio.load_scp(f"{datadir}/feats.scp")

        self.feat_dim = self.feats[valid_utts[0]].shape[1]
        self.num_labels = len(self.labels)
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


'''
Same as RandomChunkSubsetDatasetFactory but all features are kept in RAM.
'''


class FastRandomChunkSubsetDatasetFactory:

    def __init__(self, datadir, min_length=200, max_length=400, num_valid_utts=0, batch_size=64):
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size

        utt2spk = {}

        for l in open(f"{datadir}/utt2lang"):
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
        self.speakers2id = {speaker: i for i,
                            speaker in enumerate(self.speakers)}

        self.feats = {}
        logging.info(f"Reading features from {datadir}/feats.scp")
        num_lines = sum(1 for line in open(f"{datadir}/feats.scp"))
        pbar = tqdm(total=num_lines)
        for key, numpy_array in kaldiio.load_scp_sequential(f"{datadir}/feats.scp"):
            self.feats[key] = torch.tensor(numpy_array).float()
            pbar.update(1)

        pbar.close()
        logging.info("Reading features finished")

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


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, label2id, label_file="utt2lang"):
        
        self.utt2label = {}
        self.label2id = label2id

        for l in open(f"{datadir}/{label_file}"):
            ss = l.split()
            self.utt2label[ss[0]] = label2id[ss[1]]

        self.feats = {}
        logging.info(f"Reading features from {datadir}/feats.scp")
        num_lines = sum(1 for line in open(f"{datadir}/feats.scp"))
        pbar = tqdm(total=num_lines)
        self.keys = []
        for key, numpy_array in kaldiio.load_scp_sequential(f"{datadir}/feats.scp"):
            self.keys.append(key)
            self.feats[key] = torch.tensor(numpy_array).float().T
            pbar.update(1)

        pbar.close()
        logging.info("Reading features finished")
        # sort by audio length
        self.keys = sorted(self.keys, key=lambda k: len(self.feats[k]))

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        key = self.keys[index]
        return {"key": key, 
                "features": self.feats[key],
                "label": self.utt2label[key]}

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]

        len_max = max(feature_vecs.shape[1] for feature_vecs in features)
        f_dim = features[0].size(0)
        collated_features = features[0].new(
            len(features), f_dim, len_max).fill_(0.0)

        for i, v in enumerate(features):
            collated_features[i, :, : v.size(1)] = v

        batch = {
            "key": [s["key"] for s in samples],
            "features": collated_features,
            "features_length": torch.tensor([feature_vecs.shape[1] for feature_vecs in features]),
            "label": torch.tensor([s["label"] for s in samples])
        }
        return batch


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    g = FastRandomChunkSubsetDatasetFactory(
        "../sre19/data/vox_tunisian_8k_combined_no_sil/", batch_size=512)
    train_dataset = g.get_train_dataset(proportion=0.1)
    valid_dataset = g.get_valid_dataset(num_batches=3)
    # breakpoint()
    dl = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=None, num_workers=3)
    it = iter(dl)
    for i in tqdm(range(100)):
        _ = next(it)
