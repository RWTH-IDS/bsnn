import ConfigSpace as CS
import argparse
import os
import tqdm
import pandas as pd
import numpy as np
from filelock import FileLock
import time
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from sparch.helpers.parser import add_model_options
from sparch.helpers.parser import add_training_options
from sparch.exp import Experiment
from bayesian import sample_constrained_bayesian

def build_exp(config, seed, results_folder, batch_size, dataset):
    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args([])
    args.seed = seed
    args.new_exp_folder = results_folder+str(seed)+"/"
    args.model = "RLIF"
    args.dataset = dataset
    args.n_layers = 1
    args.neurons = 100 if dataset=="cue" else 128
    args.dropout = 0
    args.normalization = "none"
    args.track_balance = True
    args.balance_metric = "lowpass"
    args.plot = False
    args.batch_size = batch_size
    args.auto_encoder = False
    args.single_spike = True
    args.repeat = 4
    args.dataset_scale = 1
    args.bidirectional = False
    args.balance = True
    args.n_epochs = 10
    
    if dataset == "cue":
        args.fix_w_in = config["fix_w_in"]
        args.fix_w_rec = config["fix_w_rec"]
        args.fix_w_out = config["fix_w_out"]
        args.fix_tau_rec = config["fix_tau_rec"]
        args.fix_tau_out = config["fix_tau_out"]
        args.V_scale = config["V_scale"] 
        args.slow_dynamics = config["slow_dynamics"]
        args.V_slow_scale = config["V_slow_scale"]
        args.balance_refit = config["refit"]
    elif dataset == "shd":
        args.dataset_folder = "/mnt/data40tb/paessens/datasets/shd/"
        args.fix_w_in = config["fix_w_in"]
        args.fix_w_rec = config["fix_w_rec"]
        args.fix_w_out = config["fix_w_out"]
        args.fix_tau_rec = config["fix_tau_rec"]
        args.fix_tau_out = config["fix_tau_out"]
        args.V_scale = config["V_scale"] 
        args.slow_dynamics = config["slow_dynamics"]
        args.V_slow_scale = config["V_slow_scale"]
        args.balance_refit = config["refit"]
    else:
        raise NotImplementedError("Dataset not implemented")
    
    return Experiment(args)

def sample_config(space, data, hpo_args):
    if hpo_args.method == "random":
        return space.sample_configuration()
    elif hpo_args.method == "constrained-bayesian":
        return sample_constrained_bayesian(space, data, hpo_args.balance_constraint, hpo_args.fr_constraint, hpo_args.n_challengers)
    else:
        raise NotImplementedError("Method not implemented")


def get_configspace(name, seed, dataset):
    
    if dataset in ["cue", "shd"]:
        configspace = CS.ConfigurationSpace(
            name = name,
            seed = seed,
            space = {
                "identifier" : CS.Integer("identifier", bounds=(0,2**32-1)),
                "V_scale": CS.Float("V_scale", bounds=(1e-5, 10), log=True, default=1),
                "slow_dynamics": CS.CategoricalHyperparameter("slow_dynamics", [True, False], default_value=False),
                "V_slow_scale": CS.Float("V_slow_scale", bounds=(1e-5, 10), log=True, default=1),
                "refit" : CS.CategoricalHyperparameter("refit", [True, False], default_value=False),
                "fix_w_in" : CS.CategoricalHyperparameter("fix_w_in", [True, False], default_value=False),
                "fix_w_rec" : CS.CategoricalHyperparameter("fix_w_rec", [True, False], default_value=False),
                "fix_w_out" : CS.CategoricalHyperparameter("fix_w_out", [True, False], default_value=False),
                "fix_tau_rec" : CS.CategoricalHyperparameter("fix_tau_rec", [True, False], default_value=False),
                "fix_tau_out" : CS.CategoricalHyperparameter("fix_tau_out", [True, False], default_value=False),
                "mu" : CS.Float("mu", bounds=(1e-4, 1e-2), log=True, default=1e-3),
                "nu" : CS.Float("nu", bounds=(1e-4, 1e-2), log=True, default=1e-3),
            }
        )
        
        forbidden_clause = CS.ForbiddenAndConjunction(*[CS.ForbiddenEqualsClause(configspace[fix_], True) for fix_ in ["fix_w_in", "fix_w_rec", "fix_w_out", "fix_tau_rec", "fix_tau_out"]])
        configspace.add(forbidden_clause)
        
        forbidden_clause2 = CS.ForbiddenAndConjunction(*[CS.ForbiddenEqualsClause(configspace[fix_], True) for fix_ in ["fix_w_in", "fix_w_rec", "refit"]])
        configspace.add(forbidden_clause2)
    
    else:
        raise NotImplementedError("Dataset not implemented")
        
    return configspace


def main(hpo_args):
    N = hpo_args.n
    results_folder = "/mnt/data40tb/paessens/sparch_hpo/"+ hpo_args.opt_name +"/"
    lock = FileLock(results_folder+"results.csv.lock", timeout=3)
    random = np.random.RandomState(time.time_ns() % 2**32)
    
    configspace = get_configspace(hpo_args.opt_name, hpo_args.config_seed, hpo_args.dataset)
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        with lock:
            data = pd.DataFrame(columns=["seed"] + list(configspace.keys()) + ["accuracy", "firing_rate", "balance_lowpass", "balance_median"])
            data.to_csv(results_folder+"results.csv", index=False)
        
    for config in tqdm.tqdm(range(N)):
        data = pd.read_csv(results_folder+"results.csv")
        
        seed = random.randint(0, 2**32)
        while os.path.exists(results_folder+str(seed)):
            seed = random.randint(0, 2**32)
            
        config = sample_config(configspace, data, hpo_args)
        exp = build_exp(config, seed, results_folder, hpo_args.batch_size, hpo_args.dataset)
        
        print()
        print("CONFIG: ", config)
        
        exp.forward()
        
        acc = exp.results_dict["test_acc"]
        fr = exp.results_dict["test_fr"]
        balance_low = exp.results_dict["test_balance_low"]
        balance_med = exp.results_dict["test_balance_med"]
        
        with lock:
            data = pd.read_csv(results_folder+"results.csv")  
            index = len(data)     
            data.loc[index] = [seed] + list(config.values()) + [acc, fr, balance_low, balance_med]
            data.to_csv(results_folder+"results.csv", index=False)        
        
if __name__ == "__main__":
    hpo_parser = argparse.ArgumentParser(description="HPO for balanced SNN")
    hpo_parser.add_argument("--config-seed", type=int, required=True, help="Seed to use for config space sampling")
    hpo_parser.add_argument("--opt-name", type=str, required=True, help="Name of the optimization")
    hpo_parser.add_argument("--batch-size", type=int, default=30, help="Batch size")
    hpo_parser.add_argument("--method", type=str, default="random", choices=["random", "constrained-bayesian"], help="HPO method")
    hpo_parser.add_argument("--dataset", type=str, default="cue", choices=["cue", "shd"], help="Dataset to use")
    hpo_parser.add_argument("--balance-constraint", type=float, default=0.5, help="Balance constraint (only considered for constrained-bayesian)")
    hpo_parser.add_argument("--fr-constraint", type=float, default=0.5, help="Firing rate constraint (only considered for constrained-bayesian)")
    hpo_parser.add_argument("--n-challengers", type=int, default=1000, help="Number of challengers to sample (only considered for constrained-bayesian)")
    hpo_parser.add_argument("--n", type=int, default=50, help="Number of configurations to sample")
    hpo_args = hpo_parser.parse_args()
    main(hpo_args)
        