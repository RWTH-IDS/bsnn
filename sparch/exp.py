#
# SPDX-FileCopyrightText: Copyright © 2022 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is part of the sparch package.
#
# Extension
# Date: 30-11-2024
# Author: Tim Stadtmann
# Content: Cleanup, simplification, new parameters
"""
This is to define the experiment class used to perform training and testing
of ANNs and SNNs on all speech command recognition datasets.
"""
import errno
import logging
import os
import time
import datetime
from datetime import timedelta
import random
import gc
import inspect
import warnings
import tqdm
import yaml
import git

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

from sparch.dataloaders.nonspiking_datasets import *
from sparch.dataloaders.spiking_datasets import * 
from sparch.models import anns
from sparch.models import snns
from sparch.models.anns import ANN
from sparch.models.snns import SNN

logger = logging.getLogger(__name__)


class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):
        
        if args.auto_encoder:
            if not args.fix_w_in or not args.fix_w_out or \
                not args.fix_w_rec or not args.fix_tau_out or not args.fix_tau_rec:
                    
                warnings.warn("Auto-encoder does not support learnable time constants or input/output/recurrent weights. Overwriting to False.")
            
            args.fix_w_in = True
            args.fix_w_out = True
            args.fix_w_rec = True
            args.fix_tau_out = True
            args.fix_tau_rec = True
            
        if args.balance_refit:
            if args.fix_w_rec and args.fix_w_in:
                raise ValueError("Cannot refit balance with fixed input and recurrent weights")

        # Unpack args into member variables
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.plot_classes = [0, 2, 4]
        self.plot_class_cnt = {p:0 for p in self.plot_classes}
        self.plot_class_cnt_max = 5
        self.plot_batch_id = 0

        self.outname = (self.dataset + "_" + self.model + "_" + \
                    str(self.n_layers) + "lay" + str(self.neurons) + \
                    "_drop" + str(self.dropout) + "_" + str(self.normalization) + \
                    "_nobias" + \
                    "_bdir" if self.bidirectional else "_udir" + \
                    "_reg" if self.use_regularizers else "_noreg" + \
                    "_lr" + str(self.lr) + \
                    "_repeat" + str(self.repeat) + \
                    "_singlespike" if self.single_spike else "").replace(".", "_")

        # Set seed
        self.seed = int(datetime.datetime.now().timestamp()) if self.seed == -1 else self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()

        logging.info(f"\nSaving results and trained model in {self.exp_folder}\n")

        # Set device
        if self.gpu != -1:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.train_loader, self.valid_loader, self.test_loader, self.n_inputs, self.n_outputs, t_crop = self.init_dataset()
        args.t_crop = t_crop
        if self.auto_encoder:
            self.n_outputs = self.n_inputs
        self.init_model(args)

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.balance_refit = args.balance_refit

        # Save commit ID in args & save args as yaml file
        repo = git.Repo(search_parent_directories=True)
        args.commit = str(repo.head.commit)
        with open(self.exp_folder+"/params.yml", "w") as f:
            yaml.dump(args, f)
            
        self.pruning_epochs = args.pruning_epochs
        self.pruning_threshold = args.pruning_threshold

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if self.auto_encoder:
            raise NotImplementedError("Training has not been implemented for auto-encoders. Use main.py::run_sample() instead.")

        e=0
        if not self.only_do_testing:
            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_acc, _ = self.valid_one_epoch(self.start_epoch, self.valid_loader, test=False)
                best_epoch = self.start_epoch
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            train_accs, train_frs, validation_accs, validation_frs = [], [], [], []
            if self.track_balance:
                train_balances_med, train_balances_low, valid_balances_med, valid_balances_low = [], [], [], []
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.n_epochs + 1):
                train_acc, train_fr = self.train_one_epoch(e)
                if self.track_balance:
                    train_balances_med.append(self.balance_val_med)
                    train_balances_low.append(self.balance_val_low)

                valid_acc, valid_fr = self.valid_one_epoch(e, self.valid_loader, test=False)
                if self.track_balance:
                    valid_balances_med.append(self.balance_val_med)
                    valid_balances_low.append(self.balance_val_low)
                self.scheduler.step(valid_acc) # Update learning rate
                if self.balance_refit:
                    print("Refitting network according to balance theory.")
                    self.net.snn[0].refit()

                # Update best epoch and accuracy
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_epoch = e

                    # Save best model
                    if self.save_best:
                        torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                        logging.info(f"\nBest model saved with valid acc={valid_acc}" + (f" and balance (lp)={valid_balances_low[-1]}" if self.track_balance else ""))
                        
                if e > self.pruning_epochs and valid_acc < self.pruning_threshold:
                    break

                train_accs.append(train_acc)
                train_frs.append(train_fr)
                validation_accs.append(valid_acc)
                validation_frs.append(valid_fr)
                logging.info("\n-----------------------------\n")
                gc.collect()

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                self.net = torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                )
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        logging.info("\n------ Begin Testing ------\n")
        if self.dataset in ["sc", "ssc"]:
            test_acc, test_fr = self.valid_one_epoch(e, self.test_loader, test=True)
        else:
            logging.info("\nThis dataset uses the same split for validation and testing.\n")
            test_acc, test_fr = self.valid_one_epoch(e, self.valid_loader, test=True)
        logging.info("\n-----------------------------\n")

        # Save results summary
        results = {}
        results["train_accs"] = train_accs
        results["train_frs"] = train_frs
        results["train_balances_med"] = train_balances_med
        results["train_balances_low"] = train_balances_low
        results["validation_accs"] = validation_accs
        results["validation_frs"] = validation_frs
        results["validation_balances_med"] = valid_balances_med
        results["validation_balances_low"] = valid_balances_low
        results["test_acc"] = test_acc
        results["test_fr"] = test_fr
        results["test_balance_med"] = self.balance_val_med
        results["test_balance_low"] = self.balance_val_low
        results["best_acc"] = best_acc
        results["best_epoch"] = best_epoch
        
        self.results_dict = results
        torch.save(results, f"{self.exp_folder}/results.pth")

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )
        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder
        # Generate a path for new model from chosen config
        else:
            exp_folder = "exp/" + self.outname + "/run0"
            while os.path.exists(exp_folder): # if run0 already exists, create run1 (and so on)
                exp_folder=exp_folder[:-1] + str(int(exp_folder[-1])+1)

        # For a new model check that out path does not exist
        if not self.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if self.plot:
            self.plot_dir = exp_folder + "/plots/"
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function creates a dataloaders for the given dataset.

        Arguments
        ---------
        dataset : str
            Name of the dataset, either shd or ssc.
        dataset_folder : str
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
        if self.dataset == "shd" or self.dataset == "hd":
            logging.info(f"{self.dataset} does not have a validation split. Using test split.")
        
        if self.dataset == "cue":
            logging.info(f"Cue accumulation not usable for training yet, just inferring individual samples")

        if self.dataset == "shd" or self.dataset == "ssc":
            if self.augment:
                raise NotImplementedError("Data augmentation not yet implemented for spiking datasets")

            trainset = SpikingDataset(self.dataset, self.dataset_folder, "train", 100, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            train_loader = DataLoader(
                trainset,
                batch_size=self.batch_size,
                collate_fn=trainset.generateBatch,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            valset = SpikingDataset(self.dataset, self.dataset_folder, "test", 100, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            val_loader = DataLoader(
                valset,
                batch_size=self.batch_size,
                collate_fn=valset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            testset = SpikingDataset(self.dataset, self.dataset_folder, "test", 100, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            test_loader = DataLoader(
                testset,
                batch_size=self.batch_size,
                collate_fn=testset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

        elif self.dataset == "hd" or self.dataset == "sc":
            if self.dataset == "hd":
                trainset = HeidelbergDigits(self.dataset_folder, "train", self.augment)
                valset = HeidelbergDigits(self.dataset_folder, "test", self.augment)
                testset = HeidelbergDigits(self.dataset_folder, "test", self.augment)
            else:
                trainset = SpeechCommands(self.dataset_folder, "training", self.augment)
                valset = SpeechCommands(self.dataset_folder, "validation", self.augment)
                testset = SpeechCommands(self.dataset_folder, "testing", self.augment)

            train_loader = DataLoader(
                trainset,
                batch_size=self.batch_size,
                collate_fn=trainset.generateBatch,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )
            val_loader = DataLoader(
                valset,
                batch_size=self.batch_size,
                collate_fn=valset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            test_loader = DataLoader(
                testset,
                batch_size=self.batch_size,
                collate_fn=testset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        elif self.dataset == "cue":
            trainset = CueAccumulationDataset(self.seed, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            train_loader = DataLoader(
                trainset,
                batch_size=self.batch_size,
                collate_fn=trainset.generateBatch,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            valset = CueAccumulationDataset(self.seed, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            val_loader = DataLoader(
                valset,
                batch_size=self.batch_size,
                collate_fn=valset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            testset = CueAccumulationDataset(self.seed, labeled=not self.auto_encoder, repeat=self.repeat, scale=self.dataset_scale)
            test_loader = DataLoader(
                testset,
                batch_size=self.batch_size,
                collate_fn=testset.generateBatch,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            
        if hasattr(trainset, "t_crop"):
            t_crop = trainset.t_crop
        else:
            t_crop = 0

        logging.info(f"Sample sizes: Training set = {len(trainset)}; validation set = {len(valset)}; test set = {len(testset)}")
        return train_loader, val_loader, test_loader, trainset.n_units, trainset.n_classes, t_crop

    def init_model(self, args):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        args.input_shape = (self.batch_size, None, self.n_inputs)
        args.layer_sizes = self.n_layers*[self.neurons] + [self.n_outputs]

        if self.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=self.device)
            logging.info(f"\nLoaded model at: {self.load_path}")
        elif self.model in [name.split('Layer')[0] for name, _ in inspect.getmembers(snns) if 'Layer' in name and 'Readout' not in name]:
            self.net = SNN(args).to(self.device)
        elif self.model in [name.split('Layer')[0] for name, _ in inspect.getmembers(anns) if 'Layer' in name and 'Readout' not in name]:
            self.net = ANN(args).to(self.device)

        self.n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.input_shape = args.input_shape
        self.layer_sizes = args.layer_sizes
        logging.info(f"Model {self.net } has {self.n_params - (1 if self.auto_encoder else 0)} trainable parameters")

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        Returns: training accuracy (float), spike rate average (float)
        """
        start = time.time()
        self.net.train()
        losses, accs = [], []
        if self.track_balance:
            balances_med, balances_low, firing_rates_arr = [], [], []
        epoch_spike_rate = 0

        pbar = tqdm.tqdm(total = len(self.train_loader), desc="Training: acc=0.0000, fr=0.0000" + (", balance=0.0000" if self.track_balance else ""))
        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output, firing_rates = self.net(x)
            
            if self.track_balance:
                balances_med.append(self.net.balance_val_med)
                balances_low.append(self.net.balance_val_low)
                firing_rates_arr.append(torch.mean(firing_rates).detach().cpu().numpy())

            # Compute loss
            loss_val = self.loss_fn(output, y)
            losses.append(loss_val.item())

            # Spike activity
            if self.net.__class__.__name__ == "SNN":
                epoch_spike_rate += torch.mean(firing_rates)

                if self.use_regularizers:
                    reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                    loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

            # Compute accuracy with labels
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

            # Plot if necessary
            label=int(y[self.plot_batch_id])
            if self.plot and (e-1) % self.plot_epoch_freq == 0 and label in self.plot_classes and self.plot_class_cnt[label] < self.plot_class_cnt_max:
                self.net.plot(self.plot_dir+f"epoch{e}_class{label}_{self.plot_class_cnt[label]}.png")
                self.plot_class_cnt[label]+=1
                
            pbar.set_description(f"Training: acc={acc:.4f}, fr={torch.mean(firing_rates):.4f}" + (f", balance (lp)={self.net.balance_val_low:.4f}" if self.track_balance else ""), refresh=False)

            pbar.update(1)
            
        pbar.close()

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: Train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: Train acc={train_acc}")

        # Train spike activity of whole epoch
        if self.net.__class__.__name__ == "SNN":
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: Train mean act rate={epoch_spike_rate}")
        
        if self.track_balance:
            if self.plot:
                plt.clf()
                _, ax = plt.subplots(figsize=(10, 5))
                t = np.arange(len(balances_low))
                ax.plot(t, balances_low, label="balance lowpass")
                ax.set_xlabel("Batch")
                ax.set_ylabel("Balance")
                ax2 = ax.twinx()
                ax2.plot(t, firing_rates_arr, label="Firing rate", color="orange")
                ax2.set_ylabel("Firing rate")
                plt.legend()
                plt.savefig(self.plot_dir+f"epoch{e}_balance_fr.png", bbox_inches='tight', pad_inches=0.3)
                
            
            self.balance_val_med = np.mean(balances_med)
            self.balance_val_low = np.mean(balances_low)
            logging.info(f"Epoch {e}: Train balance (lp)={self.balance_val_low}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: Train elapsed time={elapsed}")

        return train_acc, epoch_spike_rate.cpu().detach().numpy()

    def valid_one_epoch(self, e, dataloader, test=False):
        """
        This function tests the model with a single pass over the
        validation/test set (given via dataloader)
        Returns: validation accuracy (float), spike rate average (float)
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            if self.track_balance:
                balances_med, balances_low = [], []
            epoch_spike_rate = 0
            
            pbar = tqdm.tqdm(total = len(dataloader), desc="Validation")

            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(dataloader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates = self.net(x)
                
                if self.track_balance:
                    balances_med.append(self.net.balance_val_med)
                    balances_low.append(self.net.balance_val_low)

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.__class__.__name__ == "SNN":
                    epoch_spike_rate += torch.mean(firing_rates)
                    
                pbar.update(1)
                    
            pbar.close()

            # Validation loss of whole epoch
            mean_acc = np.mean(accs)
            if test:
                logging.info(f"Test loss={np.mean(losses)}")
                logging.info(f"Test acc={mean_acc}")
                if self.track_balance:
                    logging.info(f"Test balance med={np.mean(balances_med)}")
                    logging.info(f"Test balance low={np.mean(balances_low)}")
            else:
                logging.info(f"Epoch {e}: Valid loss={np.mean(losses)}")
                logging.info(f"Epoch {e}: Valid acc={mean_acc}")
                if self.track_balance:
                    logging.info(f"Epoch {e}: Valid balance med={np.mean(balances_med)}")
                    logging.info(f"Epoch {e}: Valid balance low={np.mean(balances_low)}")

            # Validation spike activity of whole epoch
            if self.net.__class__.__name__ == "SNN":
                epoch_spike_rate /= step
                if test:
                    logging.info(f"Test mean act rate={epoch_spike_rate}")
                else:
                    logging.info(f"Epoch {e}: valid mean act rate={epoch_spike_rate}")
                    
            if self.track_balance:
                self.balance_val_med = np.mean(balances_med)
                self.balance_val_low = np.mean(balances_low)

            return mean_acc, epoch_spike_rate.cpu().detach().numpy()
