import math
import os

import pandas as pd
import torch
from danspeech.audio.resources import load_audio_wavPCM
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler, Sampler, BatchSampler, \
    DistributedSampler
import torch.distributed as dist


class DanSpeechDataset(Dataset):
    """
    Specifies a generator class for speech data
    located in a root directory. Speech data must be
    in .wav format and the root directory must include
    a .csv file with a list of filenames.

    Samples can be obtained my the __getitem__ method.
    Samples can be augmented by specifying a list of
    transform classes. DanSpeech offers a variety of
    premade transforms.
    """

    def __init__(self, root_dir, labels, audio_parser):

        print("Loading dataset from {}...".format(root_dir))
        self.audio_parser = audio_parser
        self.root_dir = root_dir
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])

        # Init csv file
        # Should only be single file with file,trans
        csv_files = [f for f in os.listdir(self.root_dir) if ".csv" in f]
        assert len(
            csv_files) == 1, "Multiple csv files present in specified data directory"
        csv_file = csv_files[0]

        # ToDO: Should not rely on pandas
        meta = pd.read_csv(os.path.join(root_dir, csv_file), encoding="utf-8")
        meta = list(zip(meta["file"].values, meta["trans"].values))

        # Check that all files exist
        files_not_found = False

        new_meta = []
        for f, trans in meta:
            if not os.path.isfile(os.path.join(self.root_dir, f)):
                files_not_found = True
            else:
                new_meta.append((f, trans))

        if files_not_found:
            print("Not all audio files in the found csv file were found.")

        keys = list(range(len(new_meta)))
        self.meta = dict(zip(keys, new_meta))

        self.size = len(self.meta)
        print("Length of dataset: {0}".format(self.size))

    def __len__(self):
        return self.size

    def path_gen(self, f):
        return os.path.join(self.root_dir, f)

    def __getitem__(self, idx):
        # ToDo: Consider rewriting load audio to use the SpeechFile audio loading setup and benchmark

        # ToDO: Not sure why we need this check. But we do for multi GPU setup.
        if type(idx) == Tensor:
            idx = idx.item()
        f, trans = self.meta[idx]

        trans = trans.lower()
        recording = load_audio_wavPCM(path=self.path_gen(f))
        recording = self.audio_parser.parse_audio(recording)

        trans = [self.labels_map.get(c) for c in trans]
        return recording, trans


def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes


class BatchDataLoader(DataLoader):
    def __init__(self, dataset, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(BatchDataLoader, self).__init__(dataset, *args, **kwargs)
        self.collate_fn = _collate_fn


class MultiDatasetBatchDataLoader(DataLoader):
    """
    Only relevant if not multi GPU.
    """
    def __init__(self, danspeech_multi_dataset, batch_size, *args, **kwargs):
        weighted_sampler = WeightedRandomSampler(danspeech_multi_dataset.final_weights, len(danspeech_multi_dataset))
        batch_sampler = BatchSampler(weighted_sampler, batch_size=batch_size, drop_last=True)
        super(MultiDatasetBatchDataLoader, self).__init__(danspeech_multi_dataset, batch_sampler=batch_sampler, *args, **kwargs)
        self.collate_fn = _collate_fn


class DistributedSamplerCustom(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    This is custom class that implements sorta grad i.e. first epoch is sorted
    assumes data handler sorts data first
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = list(torch.randperm(len(self.dataset), generator=g))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DanSpeechMultiDataset(ConcatDataset):
    def __init__(self, root_dir_list, weight_list, labels, audio_parser):
        datasets = []
        final_weights = []
        total_length = 0
        for root_dir in root_dir_list:
            ds = DanSpeechDataset(root_dir, labels, audio_parser)
            total_length += len(ds)
            datasets.append(ds)

        for ds, w in zip(datasets, weight_list):
            w_adjusted = float(w) / len(ds) * 1000
            final_weights += [w_adjusted] * len(ds)

        self.final_weights = final_weights

        super(DanSpeechMultiDataset, self).__init__(datasets)


class DistributedWeightedSamplerCustom(DistributedSampler):
    """
    Only relevant for multi-gpu
    """
    def __init__(self, danspeech_multi_dataset, num_replicas=None, rank=None):
        super(DistributedWeightedSamplerCustom, self).__init__(danspeech_multi_dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = WeightedRandomSampler(danspeech_multi_dataset.final_weights, len(danspeech_multi_dataset))

    def __iter__(self):
        # deterministically shuffle based on epoch
        self.sampler.generator = torch.manual_seed(self.epoch)

        indices = []
        while len(indices) < self.total_size:
            indices += list(self.sampler)

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
