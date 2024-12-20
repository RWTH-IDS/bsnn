# -----------------------------------------------------------------------------
# File Name : parser.py
# Purpose:
#
# Author: Tim Stadtmann
#
# Creation Date : 28-08-2024
#
# Copyright : (c) Tim Stadtmann
# License : BSD-3-Clause
# -----------------------------------------------------------------------------

import inspect
from sparch.models import snns, anns

def add_model_options(parser):
    parser.add_argument(
        "--model",
        type=str,
        choices=[name.split('Layer')[0] for name, _ in inspect.getmembers(snns)+inspect.getmembers(anns) if 'Layer' in name and 'Readout' not in name],
        default="LIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--balance",
        action='store_true',
        default=False,
        help="If True, the network will be initialized according to the balance theory.",
    )
    parser.add_argument(
        "--fix-w-out",
        action='store_true',
        default=False,
        help="If True, the readout layer will have fixed weights.",
    )
    parser.add_argument(
        "--fix-w-in",
        action='store_true',
        default=False,
        help="If True, the input layer will have fixed weights.",
    )
    parser.add_argument(
        "--fix-w-rec",
        action='store_true',
        default=False,
        help="If True, the recurrent weights will be fixed.",
    )
    parser.add_argument(
        "--fix-tau-out",
        action='store_true',
        default=False,
        help="If True, the readout layer will have fixed time constants.",
    )
    parser.add_argument(
        "--fix-tau-rec",
        action='store_true',
        default=False,
        help="If True, the recurrent layer will have fixed time constants.",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of layers (including readout layer).",
    )
    parser.add_argument(
        "--neurons", "--n",
        type=int,
        default=128,
        dest="neurons",
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="batchnorm",
        help="Type of normalization, Every string different from batchnorm "
        "and layernorm will result in no normalization.",
    )
    parser.add_argument(
        "--bidirectional",
        action='store_true',
        default=False,
        help="If True, a bidirectional model that scans the sequence in both "
        "directions is used, which doubles the size of feedforward matrices. ",
    )
    parser.add_argument(
        "--track-balance",
        action='store_true',
        default=False,
        help="If True, input currents to individual neurons are tracked and saved - will reduce performance drastically.",
    )
    parser.add_argument(
        "--single-spike",
        action='store_true',
        default=False,
        help="If True, only spike per timestep is allowed, chosen randomly.",
    )
    parser.add_argument(
        "--auto-encoder",
        action='store_true',
        default=False,
        help="If True, network will be initialized as an auto-encoder.",
    )    
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=0.9999,
        help="Leaky integration factor.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Linear cost term (penalize high number of spikes)"
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.0,
        help="Quadratic cost term (penalize non-equally distributed spikes)"
    )
    parser.add_argument(
        "--V-scale",
        type=float,
        default=1.0,
        help="Scaling factor for the recurrent weights."
    )
    parser.add_argument(
        "--V-slow-scale",
        type=float,
        default=1.0,
        help="Scaling factor for the slow recurrent weights."
    )
    parser.add_argument(
        "--balance-refit",
        action='store_true',
        default=False,
        help="Should the weights/threshold be refit after each epoch according to balance theory?"
    )
    parser.add_argument(
        "--slow-dynamics",
        action='store_true',
        default=False,
        help="If True, the network will add recurrent currents that use a filtered version of the spikes and new, trainable weights"
    )
    parser.add_argument(
        "--reset",
        default="default",
        choices=["default", "voltage", "threshold"],
        help="What to reset after each spike. Default will either substract 1 or use the diagonal of V_rec (depending on balance), voltage will subtract the voltage, threshold will subtract the threshold."
    )
    parser.add_argument(
        "--w-in-init",
        default="bernoulli",
        choices = ["bernoulli", "uniform"],
        help="How to initialize the input weights"
    )
    parser.add_argument(
        "--pruning-epochs",
        type=int,
        default=2,
        help="Number of epochs to prune the network"
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=0.12,
        help="Accuracy threshold for pruning"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="",
        help="Perform fixed-point quantization on weights if set to BITS.FRAC"
    )
    parser.add_argument(
        "--quantize-adc",
        type=str,
        default="",
        help="Perform fixed-point quantization on output of Wx multiplication if set to BITS.FRAC (to model ADC)"
    )
    parser.add_argument(
        "--gauss",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--gauss-mul",
        action='store_true',
        default=False,
        help="If True, the gaussian noise will be multiplied by output of Wx multiplication, otherwise added to it"
    )
    return parser

def add_training_options(parser):
    parser.add_argument(
        "--use-pretrained-model",
        action='store_true',
        default=False,
        help="Whether to load a pretrained model or to create a new one.",
    )
    parser.add_argument(
        "--only-do-testing",
        action='store_true',
        default=False,
        help="If True, will skip training and only perform testing of the "
        "loaded model.",
    )
    parser.add_argument(
        "--load-exp-folder",
        type=str,
        default=None,
        help="Path to experiment folder with a pretrained model to load. Note "
        "that the same path will be used to store the current experiment.",
    )
    parser.add_argument(
        "--new-exp-folder",
        type=str,
        default=None,
        help="Path to output folder to store experiment.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["shd", "ssc", "hd", "sc", "cue"],
        default="shd",
        help="Dataset name (shd, ssc, hd, sc or cue).",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="data/shd/",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--dataset-scale",
        type=float,
        default=1.0,
        help="Each spike in the dataset is multiplied by this factor"
    )
    parser.add_argument(
        "--log",
        action='store_true',
        default=False,
        help="Whether to print experiment log in an dedicated file or "
        "directly inside the terminal.",
    )
    parser.add_argument(
        "--save-best",
        action='store_true',
        default=True,
        help="If True, the model from the epoch with the highest validation "
        "accuracy is saved, if False, no model is saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of input examples inside a single batch.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5,
        help="Number of training epochs (i.e. passes through the dataset).",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="Epoch number to start training at. Will be 0 if no pretrained "
        "model is given. First epoch will be start_epoch+1.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate for training. The default value of 0.01 "
        "is good for SHD and SC, but 0.001 seemed to work better for HD and SC.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=1,
        help="Number of epochs without progress before the learning rate "
        "gets decreased.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.7,
        help="Factor between 0 and 1 by which the learning rate gets "
        "decreased when the scheduler patience is reached.",
    )
    parser.add_argument(
        "--use-regularizers",
        action='store_true',
        default=False,
        help="Whether to use regularizers in order to constrain the "
        "firing rates of spiking neurons within a given range.",
    )
    parser.add_argument(
        "--reg-factor",
        type=float,
        default=0.5,
        help="Factor that scales the loss value from the regularizers.",
    )
    parser.add_argument(
        "--reg-fmin",
        type=float,
        default=0.01,
        help="Lowest firing frequency value of spiking neurons for which "
        "there is no regularization loss.",
    )
    parser.add_argument(
        "--reg-fmax",
        type=float,
        default=0.5,
        help="Highest firing frequency value of spiking neurons for which "
        "there is no regularization loss.",
    )
    parser.add_argument(
        "--augment",
        action='store_true',
        default=False,
        help="Whether to use data augmentation or not. Only implemented for "
        "nonspiking HD and SC datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for RNGs (if -1, use time).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials to run with different seeds.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="CUDA ID of GPU to use"
    )
    parser.add_argument(
        "--plot",
        action='store_true',
        default=False,
        help="Activate plotting of spikes etc",
    )
    parser.add_argument(
        "--plot-epoch-freq",
        type=int,
        default=2,
        help="CUDA ID of GPU to use"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="How many times to repeat each spike"
    )
    parser.add_argument(
        "--balance-metric",
        type=str,
        default="median",
        choices=["median", "lowpass", "raw"],
        help="The balance metric will be calculated on a median or lowpass filtered version of the neuronal input currents, or the raw ones"
    )
    return parser
