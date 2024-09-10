#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package
#
"""
This is where the dataloader is defined for the SHD and SSC datasets.
"""
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or
    Spiking Speech Commands (SSC) dataset.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the dataset (h5py file).
    split : str
        Split of the SHD dataset, must be either "train" or "test".
    nb_steps : int
        Number of time steps for the generated spike trains.
    """

    def __init__(
        self,
        dataset_name,
        data_folder,
        split,
        nb_steps=100,
        labeled=True
    ):

        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)
        self.labeled = labeled

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)

        if self.labeled:
            y = self.labels[index]
            return x.to_dense(), y
        else:
            return x.to_dense()

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        return xs, xlens, ys
    

class CueAccumulationDataset(Dataset):
    """Adapted from the original TensorFlow e-prop implemation from TU Graz, available at https://github.com/IGITUGraz/eligibility_propagation

    Timing for cue_assignments[0] = [0,0,1,0,1,0,0]:
    t_silent (50ms) silence
    t_cue (100ms)   spikes on first 10 neurons (4% probability)
    t_silent (50ms) silence
    t_cue (100ms)   spikes on first 10 neurons (4% probability)
    t_silent (50ms) silence
    t_cue (100ms)   spikes on second 10 neurons (4% probability)
    ....
    until 2099ms    silence
    t_interval (150ms) spikes on third 10 neurons (4% probability) as recall cue
    """

    def __init__(self, seed=None, labeled=True):
        n_cues = 7
        f0 = 40
        t_cue = 100
        t_wait = 1200
        n_symbols = 4 # if 40 neurons: left cue (neurons 0-9), right cue (neurons 10-19), decision cue (neurons 20-29), noise (neurons 30-39)
        p_group = 0.3

        self.labeled = labeled
        self.dt = 1e-3
        self.t_interval = 150
        self.seq_len = n_cues*self.t_interval + t_wait
        self.n_in = 40
        self.n_out = 2    # This is a binary classification task, so using two output units with a softmax activation redundant
        n_channel = self.n_in // n_symbols
        prob0 = f0 * self.dt
        t_silent = self.t_interval - t_cue

        length = 200

        # Randomly assign group A and B
        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = np.random.choice([0, 1], length)
        probs = np.zeros((length, 2), dtype=np.float32)
        # Assign input spike probabilities
        probs[:, 0] = prob_choices[idx]
        probs[:, 1] = prob_choices[1 - idx]

        cue_assignments = np.zeros((length, n_cues), dtype=int)
        # For each example in batch, draw which cues are going to be active (left or right) -> e.g. cue_assignments[0]=[0,0,1,0,1,0,0]
        for b in range(length):
            cue_assignments[b, :] = np.random.choice([0, 1], n_cues, p=probs[b])

        # Generate input spikes
        input_spike_prob = np.zeros((length, self.seq_len, self.n_in))
        t_silent = self.t_interval - t_cue
        for b in range(length):
            for k in range(n_cues):
                # Input channels only fire when they are selected (left or right)
                c = cue_assignments[b, k]
                input_spike_prob[b, t_silent+k*self.t_interval:t_silent+k *
                                 self.t_interval+t_cue, c*n_channel:(c+1)*n_channel] = prob0

        # Recall cue and background noise
        input_spike_prob[:, -self.t_interval:, 2*n_channel:3*n_channel] = prob0
        input_spike_prob[:, :, 3*n_channel:] = prob0/4.
        input_spikes = self.generate_poisson_noise_np(input_spike_prob, seed)
        self.x = torch.tensor(input_spikes).float()

        # Generate targets
        target_nums = np.zeros((length, self.seq_len), dtype=int)
        target_nums[:, :] = np.transpose(
            np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (self.seq_len, 1)))
        self.y = torch.tensor(target_nums).long()

    def generate_poisson_noise_np(self, prob_pattern, freezing_seed=None):
        if isinstance(prob_pattern, list):
            return [self.generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

        shp = prob_pattern.shape
        rng = np.random.RandomState(freezing_seed)

        spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
        return spikes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.labeled:
            return self.x[index], self.y[index]
        else:
            return self.x[index]

def load_shd_or_ssc(
    dataset_name,
    data_folder,
    split,
    batch_size,
    nb_steps=100,
    shuffle=True,
    workers=0,
):
    """
    This function creates a dataloader for a given split of
    the SHD or SSC datasets.

    Arguments
    ---------
    dataset_name : str
        Name of the dataset, either shd or ssc.
    data_folder : str
        Path to folder containing the Heidelberg Digits dataset.
    split : str
        Split of dataset, must be either "train" or "test" for SHD.
        For SSC, can be "train", "valid" or "test".
    batch_size : int
        Number of examples in a single generated batch.
    shuffle : bool
        Whether to shuffle examples or not.
    workers : int
        Number of workers.
    """
    if dataset_name not in ["shd", "ssc"]:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")

    if dataset_name == "shd" and split == "valid":
        logging.info("SHD does not have a validation split. Using test split.")
        split = "test"

    dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps)
    logging.info(f"Number of examples in {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader
