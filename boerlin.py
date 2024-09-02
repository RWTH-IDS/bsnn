import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import os
from sparch.dataloaders.spiking_datasets import SpikingDataset

BALANCE_EPS=0.005  # = mean dist between i_exc and -i_inh


def parse_args():
  parser = argparse.ArgumentParser(description='Simulate spiking integrator')
  parser.add_argument('--n', type=int, default=400, help='Number of recurrent units')
  parser.add_argument('--h', type=int, default=0.0001, help='Simulaton time step (s)')
  parser.add_argument('--w-init', type=str, default='boerlin-fix', choices = ['boerlin-fix', 'boerlin-rand', 'rand'], help='Choice of the w-out initialization')
  parser.add_argument('--lds', type=str, default='1d', choices = ['1d', '2d'], help='Choice of the dynamical system to mimic/train on')
  parser.add_argument('--lambda-d', type=float, default=10, help='Leak term of read out (Hz)')
  parser.add_argument('--lambda-v', type=float, default=20, help='Leak term of membrane voltage (Hz)')
  parser.add_argument('--sigma-v', type=float, default=0.001, help='Standard deviaton of noise injected each time step into membrane voltage v')
  parser.add_argument('--lambda-s', type=float, default=0, help='Leak term of sensory integrator (Hz)')
  parser.add_argument('--sigma-s', type=float, default=0.01, help='Standard deviaton of noise injected each time step into sensory input c')
  parser.add_argument('--rescale', action='store_true', help='Deactivate rescaling of weights')
  parser.add_argument('--mu', type=float, default=0, help='Linear cost term (penalize high number of spikes)')
  parser.add_argument('--nu', type=float, default=0, help='Quadratic cost term (penalize non-equally distributed spikes)')
  parser.add_argument('--v-rest', type=float, default=0, help='Resting voltage')
  parser.add_argument('--v-thresh', type=float, default=0.5, help='Threshold voltage')
  parser.add_argument('--seed', type=int, default=0, help='Random seed (if -1: use default seed (system time I think?))')
  parser.add_argument('--alemi', action='store_true', help='Train w_slow using Alemi-rule instead of fixed init + use error feedback')
  parser.add_argument('--k', type=float, default=0.5, help='feedback scale')
  parser.add_argument('--eta', type=float, default=0.01, help='learn rate')
  parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
  parser.add_argument('--track-balance', action='store_true', help='trace input inh/exc currents to neurons (slows down simulation)')
  parser.add_argument('--auto-encoder', action='store_true', help='Implement auto-encoder instead of function encoder (aka set W_s = 0)')
  parser.add_argument('--plot-neuron', type=int, default=0, help='ID of neuron whose currents will be plotted')
  parser.add_argument('--plot', action='store_true', help="Visualize plot")
  parser.add_argument('--plot-dim', type=int, default=4, help='Maximum dimension of the input signal to be plotted')
  parser.add_argument('--plot-input-raster', action='store_true', help='(only for SHD) Plot input data as raster plot')
  parser.add_argument('--save', type=str, default="", help='Save plot in given path as png file')
  parser.add_argument('--save-path', type=str, default="plots", help="Folder to store plots in")
  parser.add_argument('--data', type=str, default="1d", help="Dataset to use for training")
  return parser.parse_args()

def main(args):
  ## solve linear dynamical system (LDS) ẋ = Ax + c and formally-equivalent balanced network (EBN)
  print("### Balanced Spiking Neural Network ###")
  print("# Arguments")
  print(''.join(f' {k}={v}\n' for k, v in vars(args).items()))

  # define constants for leaky integrator example & unpack args for convenience
  N = args.n
  h = args.h
  lambda_d = args.lambda_d
  lambda_v = args.lambda_v
  sigma_v = args.sigma_v
  lambda_s = args.lambda_s
  sigma_s = args.sigma_s
  mu = args.mu
  nu = args.nu
  rescale = args.rescale
  
  ## define and solve LDS
  # define LDS of form: ẋ = Ax + c(t)
  if args.data != "shd": # random data generation
    J = int(args.data.split("d")[0])
    t = 20000
    for dim in range(J):
      if dim==0: # default boerlin example
        c_orig = np.zeros([t, J])
        c_orig[0:2000, 0] = 0
        c_orig[2000:5000, 0] = 50
        c_orig[5000:7000, 0] = 0
        c_orig[7000:9000, 0] = -100
        c_orig[9000:-1, 0] = 0
      else:
        num_slices = random.randint(3, 8)
        slice_lengths = [random.randint(1, t) for _ in range(num_slices)]
        slice_lengths = [int(l / sum(slice_lengths) * t) for l in slice_lengths]

        start = 0
        for length in slice_lengths:
            end = start + length
            c_orig[start:end, dim] = random.randint(-100, 100)
            start = end
        if start < t: # If there's any remaining length, fill it with the last value
            c_orig[start:, dim] = 0
  else:
    t = 100
    dataset = SpikingDataset("shd", "SHD", "train", t, False)
    c_orig = dataset[0].cpu().numpy()
    J = dataset.nb_units
 
  # solve LDS with forward Euler and exact solution
  c       = np.zeros([t, J])
  x_euler = np.zeros([t, J])
  #x_explicit = np.zeros([t, J])
  if args.auto_encoder:
    A = -lambda_d*np.ones(J)
  else:
    A = -lambda_s * np.ones(J)

  for k in tqdm(range(t-1), desc="# Euler Integration"):
    c[k] = c_orig[k] + sigma_s * np.random.randn(*c_orig[k].shape) * (1/h)

    x_euler[k+1] = (1+h*A)*x_euler[k] + h*c[k]  # explicit euler
    #x_explicit[k+1] = np.exp(-h*A[0][0])*x_explicit[k] + ((1-np.exp(-h*A[0][0]))/A[0][0])*c[k]  # exact

  ## define and solve EBN with forward Euler
  print("")
  print("# Weight initialization")
  w_out = np.zeros([J, N])  # relation to paper: output kernel Γ_i = w_out_i = w_out[:, i]
  if args.w_init == 'boerlin-fix':
    w_out[:,0:int(w_out.shape[1]/2)] = 0.1
    w_out[:,int(w_out.shape[1]/2):int(w_out.shape[1])] = -0.1
  elif args.w_init == 'boerlin-rand':
    n=int(w_out.shape[1]/2)
    w_out[:,0:n] = np.random.binomial(1, 0.7, size=(J,n)) * np.random.uniform(0.06, 0.1, size=(J, n))
    w_out[:,n:int(w_out.shape[1])] = np.random.binomial(1, 0.7, size=(J,int(w_out.shape[1])-n)) * np.random.uniform(-0.1, -0.06, size=(J,int(w_out.shape[1])-n))
  elif args.w_init == 'rand':
    w_out = np.random.binomial(1, 0.7, size=(J,N)) * np.random.uniform(-0.1, 0.1, size=(J, N))

  # set other weights
  w_in   = w_out.T                                          # NxJ
  w_fast = w_out.T @ w_out + mu * (lambda_d**2) * np.eye(N) # NxN
  w_slow = w_out.T @ (A + lambda_d*np.eye(J)) @ w_out       # NxN

  # rescaling
  T = 0.5*(nu * lambda_d + mu * lambda_d**2 + np.diagonal(w_fast)) # np.linalg.norm(w_out,axis=0)
  v_rest = np.full(N, args.v_rest, dtype=float)
  if rescale:
    v_thresh = np.full(N, args.v_thresh)
    b = v_rest
    a = v_thresh - b

    w_in = (a*np.divide(w_in.T, T)).T
    w_out = a*np.divide(w_out, T)
    w_fast = a*np.divide(w_fast, T)
    w_slow = a*np.divide(w_slow, T)
  else:
    v_thresh = T

  w_fast = -w_fast
  if args.alemi:
    w_slow = np.zeros_like(w_fast)

  if args.track_balance:
    w_fast_neg = np.where(w_fast<0, w_fast, 0)
    w_fast_pos = np.where(w_fast>=0, w_fast, 0)
    w_slow_neg = np.where(w_slow<0, w_slow, 0)
    w_slow_pos = np.where(w_slow>=0, w_slow, 0)
    w_in_neg = np.where(w_in<0, w_in, 0)
    w_in_pos = np.where(w_in>=0, w_in, 0)

  print("  Summary:")
  print(f"    w_out {w_out.shape}: min={np.min(w_out):4f}, max={np.max(w_out):.4f}, mean={np.mean(w_out):.4f}, std={np.std(w_out):.4f}")
  print(f"    w_in {w_in.shape}: min={np.min(w_in):.4f}, max={np.max(w_in):.4f}, mean={np.mean(w_in):.4f}, std={np.std(w_in):.4f}")
  print(f"    w_fast {w_fast.shape}: min={np.min(w_fast):.4f}, max={np.max(w_fast):.4f}, mean={np.mean(w_fast):.4f}, std={np.std(w_fast):.4f}")
  print(f"    w_slow {w_slow.shape}: min={np.min(w_slow):.4f}, max={np.max(w_slow):.4f}, mean={np.mean(w_slow):.4f}, std={np.std(w_slow):.4f}")
  print(f"    v_thresh {v_thresh.shape}: min={np.min(v_thresh):.4f}, max={np.max(v_thresh):.4f}, mean={np.mean(v_thresh):.4f}, std={np.std(v_thresh):.4f}")
  if args.data == "shd":
    print(f"    c {c.shape}: max={np.max(c):.4f}, highest avg fr/neuron={(c.sum(axis=0)/dataset.max_time).max():.4f} (neuron={(c.sum(axis=0)/dataset.max_time).argmax()})")
  else:
    print(f"    c {c.shape}: min={np.min(c):.4f}, max={np.max(c):.4f}, mean={np.mean(c):.4f}, std={np.std(c):.4f}")
  print(f"    ")

  # c.sum()/(dataset.max_time*J) = avg firing rate dataset
  # np.sum(np.any(c_orig > 0, axis=0)==False) = silent input neurons

  for epoch in range(args.epochs):
    # solve EBN with forward euler
    x_snn = np.zeros([t, J])
    v     = np.full([t, N], v_rest)
    r     = np.zeros([t, N])
    o     = np.zeros([t, N])
    i_slow= np.zeros([t, N])
    i_fast= np.zeros([t, N])
    i_in  = np.zeros([t, N])
    i_e   = np.zeros([t, N])

    i_inh = np.zeros([t, N])
    i_exc = np.zeros([t, N])
    for k in tqdm(range(t-1), desc="# Simulation"):
      if args.track_balance:
        i_fast_inh = np.matmul(w_fast_neg, o[k])
        i_fast_exc = np.matmul(w_fast_pos, o[k])
        i_slow_inh = np.matmul(w_slow_neg, r[k]) if not args.auto_encoder else np.zeros_like(i_fast_exc)
        i_slow_exc = np.matmul(w_slow_pos, r[k]) if not args.auto_encoder else np.zeros_like(i_fast_inh)
        i_in_inh = np.matmul(w_in_neg, np.where(c[k]>=0, c[k], 0))  # c can be negative, so we need to use np.where for it to make sure we only use negative weights
        i_in_inh += np.matmul(w_in_pos, np.where(c[k]<0, c[k], 0))
        i_in_exc = np.matmul(w_in_neg, np.where(c[k]<0, c[k], 0))
        i_in_exc += np.matmul(w_in_pos, np.where(c[k]>=0, c[k], 0))
        i_inh[k] = i_slow_inh + i_fast_inh + i_in_inh
        i_exc[k] = i_slow_exc + i_fast_exc + i_in_exc

        i_slow[k] = i_slow_exc + i_slow_inh
        i_fast[k] = i_fast_exc + i_fast_inh
        i_in[k]   = i_in_exc + i_in_inh
      else:
        # update synaptic currents
        if not args.auto_encoder:
          i_slow[k] = np.matmul(w_slow, r[k])
        i_fast[k] = np.matmul(w_fast, o[k])
        i_in[k]   = np.matmul(w_in, c[k])

      # calculate error
      e = x_snn[k] - x_euler[k] if args.alemi else np.zeros(J)
      i_e[k] = args.k * np.matmul(w_in, e)

      # update weights
      if args.alemi:
        dw_slow = args.eta * (np.expand_dims(r[k],1)@(np.expand_dims(w_in @ e, 1)).T)  # eta * r * (w_in*e)^T
        #np.fill_diagonal(dw_slow, 0)
        w_slow += dw_slow

      # update membrane voltage
      v[k+1] = (1-h*lambda_v) * v[k] + h * (lambda_v * v_rest + i_in[k] + i_e[k] + i_slow[k] + i_fast[k]) + sigma_v * np.random.randn(*v[k].shape) * np.sqrt(h)

      # update rate
      r[k+1] = (1-h*lambda_d) * r[k] + h*o[k]

      # update output
      x_snn[k+1] = np.matmul(w_out, r[k+1])

      # spikes
      spike_ids = np.asarray(np.argwhere(v[k+1] > v_thresh))
      if len(spike_ids) > 0:
        spike_id = np.random.choice(spike_ids[:, 0])  # spike_ids.shape = (Nspikes, 1) -> squeeze away second dimension (cant use np.squeeze() for arrays for (1,1) though)
        o[k+1][spike_id] = 1/h

    plot(args, t, epoch, c_orig, x_euler, x_snn, o, i_slow, i_fast, i_in, i_e, v, i_inh, i_exc)
  return c_orig, x_euler, x_snn, o, i_slow, i_fast, i_in, i_e, v, i_inh, i_exc

def plot(args, seq_len, epoch, c, x_euler, x_snn, o, i_slow, i_fast, i_in, i_e, v, i_inh, i_exc):
  # define colors
  RED = "#D17171"
  YELLOW = "#F3A451"
  GREEN = "#7B9965"
  BLUE = "#5E7DAF"
  DARKBLUE = "#3C5E8A"
  DARKRED = "#A84646"
  VIOLET = "#886A9B"
  GREY = "#636363"


  # create plots
  t = list(range(0,seq_len))
  if args.track_balance:
    fig, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 3, 2]})
  else:
    fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 2, 3]})
  fig.subplots_adjust(hspace=0)

  # plot inputs 
  data_dim = min(c.shape[1], args.plot_dim) 
  ls = ['solid', 'dashed', 'dotted', 'dashdot']
  if args.data == "shd" and args.plot_input_raster:
    neurons_min, neurons_max = 0, c.shape[1]
    neurons_ticks = 100 if neurons_max > 150 else 10 if neurons_max > 20 else 1
    spikes = np.argwhere(c[:,neurons_min:neurons_max]>0)
    x_axis = spikes[:,0] # x-axis: spike times
    y_axis = spikes[:,1] # y-axis: spiking neuron ids
    colors = len(x_axis)*[BLUE]
    axs[0].scatter(x_axis, y_axis, c=colors, marker = "o", s=10)
    yticks = list(range(neurons_min,neurons_max,neurons_ticks))
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels(yticks, fontsize=12)
  else:
    for dim in range(data_dim):
      axs[0].plot(t, c[:, dim], color=GREY, label=f"c_{dim}", linestyle=ls[dim%len(ls)])
    axs[0].legend()

  # plot outputs
  for dim in range(data_dim):
    axs[1].plot(t, x_euler[:, dim], color=GREY, label=f"x_euler_{dim}", linestyle=ls[dim%len(ls)])
    axs[1].plot(t, x_snn[:, dim], color=YELLOW, label=f"x_snn_{dim}", linestyle=ls[dim%len(ls)])
  axs[1].legend()

  # axs[2].plot(t, i_slow[:, 0], color=BLUE, label="i_slow", linestyle='dashed')
  # axs[2].plot(t, i_fast[:, 0], color=BLUE, label="i_fast", linestyle='dotted')
  #axs[2].plot(t_array, i_in_array[:, 0], color=BLUE, label="i_in")
  #axs[2].plot(t_array, i_e_array[:, 0], color=BLUE, label="i_e", linestyle='dashdot')
  # axs[2].plot(t, i_slow[:, int(args.n/2)], color=RED, label="i_slow", linestyle='dashed')
  # axs[2].plot(t, i_fast[:, int(args.n/2)], color=RED, label="i_fast", linestyle='dotted')
  #axs[2].plot(t_array, i_in_array[:, 200], color=RED, label="i_in")
  #axs[2].plot(t_array, i_e_array[:, 200], color=RED, label="i_e", linestyle='dashdot')
  # axs[2].legend()

  # axs[3].plot(t, v[:, 0], color=BLUE, label="V")
  # axs[3].plot(t, v[:, int(args.n/2)], color=RED, label="V")
  # axs[3].legend()

  # plot spike raster
  neurons_min, neurons_max = 0, o.shape[1]
  neurons_ticks = 100 if neurons_max > 150 else 10 if neurons_max > 20 else 1

  spikes = np.argwhere(o[:,neurons_min:neurons_max]>0)
  x_axis = spikes[:,0] # x-axis: spike times
  y_axis = spikes[:,1]# y-axis: spiking neuron ids
  colors = len(spikes[:,0])*[BLUE]
  axs[2].scatter(x_axis, y_axis, c=colors, marker = "o", s=10)
  yticks=list(range(neurons_min,neurons_max,neurons_ticks))
  axs[2].set_yticks(yticks)
  axs[2].set_yticklabels(yticks, fontsize=12)

  balanced_str = "unknown"
  if args.track_balance:
    b, a = butter(4, 0.2, btype='low', analog=False)
    i_exc_plot = i_exc[:, args.plot_neuron]
    i_inh_plot = i_inh[:, args.plot_neuron]
    i_exc_plot = np.array(filtfilt(b, a, i_exc_plot))
    i_inh_plot = np.array(filtfilt(b, a, i_inh_plot))
    axs[3].plot(t, i_exc_plot, color=BLUE, label="i_exc")
    axs[3].plot(t, -i_inh_plot, color=RED, label="-i_inh")
    axs[3].legend()

    balanced = (-i_inh_plot-i_exc_plot)[1500:].mean() < BALANCE_EPS
    balanced_str = "balanced" if balanced else "not balanced"
    print("# Analysis")
    print("Network is " + balanced_str)

  plt.xlabel('Timesteps')
  if args.plot:
    plt.show()
  if args.save != "":
    if not os.path.exists(args.save_path):
      os.makedirs(args.save_path)
    plt.savefig(args.save_path + "/" + args.save + "_balancestate_" + balanced_str + ".png", dpi=250)
  plt.close()

if __name__ == '__main__': 
  args = parse_args()

  if args.seed != -1:
    np.random.seed(args.seed)
    random.seed(args.seed)
  main(args)

  # i = 0
  # for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
  #   for eta in [0.001, 0.01, 0.1]:
  #     print("Run %d: k = %f, eta = %f" %(i, k, eta))
  #     args.k = k
  #     args.eta = eta
  #     main(args)

  #     i += 1