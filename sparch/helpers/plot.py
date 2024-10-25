import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from scipy.signal import butter, filtfilt

BALANCE_EPS=0.005  # = mean dist between i_exc and -i_inh

def plot_network(inputs, spikes, layer_sizes, balance, currents_exc, currents_inh, voltages, show, lowpass=False, filename=""):
    # define colors
    RED = "#D17171"
    YELLOW = "#F3A451"
    GREEN = "#7B9965"
    BLUE = "#5E7DAF"
    DARKBLUE = "#3C5E8A"
    DARKRED = "#A84646"
    VIOLET = "#886A9B"
    GREY = "#636363"

    # constants for plotting
    LAYER = 0
    BATCH = 0
    NEURON = 0
    NEURONS_TO_PLOT=[50]
    N_NEURONS_TO_PLOT=len(NEURONS_TO_PLOT)

    # cast data lists to torch tensors
    spikes = spikes[LAYER, BATCH, :, :].cpu()  # layers x batch x time x neurons

    t = list(range(0,spikes.shape[0]))
    # setup spike raster plot
    neurons_min, neurons_max = 0, layer_sizes[LAYER]
    neurons_ticks = 100 if neurons_max > 150 else 10 if neurons_max > 20 else 1

    spikes = torch.argwhere(spikes[:,neurons_min:neurons_max]>0)
    x_axis = spikes[:,0].cpu().numpy()
    y_axis = spikes[:,1].cpu().numpy()
    colors = len(spikes[:,0])*[BLUE]

    idx=torch.isin(spikes[:,1], torch.tensor(NEURONS_TO_PLOT))
    x_axis_2 = spikes[idx][:,0].cpu().numpy()
    y_axis_2 = spikes[idx][:,1].cpu().numpy()
    colors_2 = len(spikes[idx][:,0])*[RED]

    # create plots
    plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=8) #fontsize of the y tick labels
    fig, axs = plt.subplots(2+N_NEURONS_TO_PLOT*2, 1, sharex=True, gridspec_kw={'height_ratios': [3] + 2*N_NEURONS_TO_PLOT*[5] + [5]}, figsize=(30,15))
    fig.subplots_adjust(hspace=0)

    # plot
    ## plot inputs
    input_i = inputs[BATCH, :, :]
    inputs_min, inputs_max = 0, input_i.shape[1]
    inputs_ticks = 100 if inputs_max > 150 else 10 if inputs_max > 20 else 1
    input_spikes = torch.argwhere(input_i[:,inputs_min:inputs_max]>0)
    input_x_axis = input_spikes[:,0].cpu().numpy() # x-axis: spike times
    input_y_axis = input_spikes[:,1].cpu().numpy() # y-axis: spiking neuron ids
    input_colors = len(input_x_axis)*[BLUE]
    axs[0].scatter(input_x_axis, input_y_axis, c=input_colors, marker = "o", s=10)
    input_yticks = list(range(inputs_min,inputs_max,inputs_ticks))
    axs[0].set_yticks(input_yticks)
    axs[0].set_yticklabels(input_yticks, fontsize=12)
    axs[0].legend()

    ## plot rest
    if balance:
        for i in range(1, 1+2*N_NEURONS_TO_PLOT, 2):
            neuron = NEURONS_TO_PLOT[int(i/2)]
            currents_exc_i = currents_exc[LAYER, BATCH, :, neuron].cpu()
            currents_inh_i = currents_inh[LAYER, BATCH, :, neuron].cpu()
            v = voltages[LAYER, BATCH, :, neuron].cpu()
            if lowpass:
                b, a = butter(4, 0.05, btype='low', analog=False) # 0.005/(0.5*spikes.shape[0])
                currents_exc_i = np.array(filtfilt(b, a, currents_exc_i))
                currents_inh_i = np.array(filtfilt(b, a, currents_inh_i))
            axs[i].plot(t, currents_exc_i, color=BLUE, label="i_exc")
            axs[i].plot(t, -currents_inh_i, color=RED, label="-i_inh")
            axs[i+1].plot(t, v, color=GREY, label="v")
            axs[i].set_title("neuron " + str(neuron), y=0.5)
            if i==0:
                axs[i].legend()

    axs[1+2*N_NEURONS_TO_PLOT].scatter(x_axis, y_axis, c=colors, marker = "o", s=8)
    axs[1+2*N_NEURONS_TO_PLOT].scatter(x_axis_2, y_axis_2, c=colors_2, marker = "o", s=8)
    yticks=list(range(neurons_min,neurons_max,neurons_ticks))
    axs[1+2*N_NEURONS_TO_PLOT].set_yticks(yticks)
    axs[1+2*N_NEURONS_TO_PLOT].set_yticklabels(yticks, fontsize=8)

    plt.xlabel('Timesteps')
    if filename!="":
        plt.savefig(filename, dpi=600)
    if show:
        plt.show()
    plt.close()