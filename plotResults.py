#!/bin/python3

# Plot functions for neuronal and synaptic variables as well as spike raster plots

# Copyright 2021-2024 Jannik Luboeinski
# License: Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# Contact: mail[at]jlubo.net

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from outputUtilities import getPresynapticId
	
#####################################
# plotResults
# Plots the results of a simulation of synaptic weights changing based on calcium dynamics
# config: configuration parameters in JSON format
# data_stacked: two-dimensional array containing the values of the membrane potential, weights, calcium amount, etc. over time
# timestamp: timestamp of the current trial
# dataset: number of the dataset (usually one neuron/synapse pair) that shall be considered
# mem_dyn_data [optional]: specifies if membrane potential and current shall be plotted
# neuron [optional]: number of the neuron that shall be considered [default: 0]
# pre_neuron [optional]: number of the presynaptic neuron that shall be considered for the synapse [default: 0]
# store_path [optional]: path to store resulting graphics file
# figure_fmt [optional]: format of resulting graphics file
def plotResults(config, data_stacked, timestamp, dataset, mem_dyn_data = False, neuron=0, pre_neuron=0, store_path = ".", figure_fmt = 'png'):

	h_0 = config["synapses"]["syn_exc_calcium_plasticity"]["h_0"] # reference value (initial value) of synaptic weight
	time = data_stacked[:,0] / 60 / 1000 # time values, in min
	xlim_0 = 0 # lower limit of time axis, in min
	xlim_1 = config["simulation"]["runtime"] / 60 # upper limit of time axis, in min
	xlim_auto = False # auto-adjustment of x-axis
	
	# adapt depending on whether membrane dynamics shall be plotted or not
	if mem_dyn_data:
		num_rows = 3 # number of rows in figure array
		data_ptr = 3+dataset*7 # pointer to the first data column for the particular synapse
	else:
		num_rows = 2 # number of rows in figure array
		data_ptr = 1+dataset*5 # pointer to the first data column for the particular synapse
		
	fig, axes = plt.subplots(nrows=num_rows, ncols=1, sharex=False, figsize=(10, 10)) # create figure with 'num_rows' subfigures

	# set axis labels for plasticity dynamics
	axes[0].set_xlabel("Time (min)")
	axes[0].set_ylabel("Synaptic weight (%)")
	axes[0].set_xlim(xmin=xlim_0, xmax=xlim_1, auto=xlim_auto)
	
	# plot data for plasticity dynamics
	axes[0].plot(time, data_stacked[:,data_ptr]/h_0*100, color="#800000", label='Early-phase', marker='None', zorder=10)
	axes[0].plot(time, (data_stacked[:,data_ptr+1]/h_0+1)*100, color="#1f77b4", label='Late-phase', marker='None', zorder=9)
	axes[0].axhline(y=(config["synapses"]["syn_exc_calcium_plasticity"]["theta_pro"]/h_0+1)*100, label='Protein thresh.', linestyle='-.', color="#dddddd", zorder=5)
	axes[0].axhline(y=(config["synapses"]["syn_exc_calcium_plasticity"]["theta_tag"]/h_0+1)*100, label='Tag thresh.', linestyle='dashed', color="#dddddd", zorder=4)
	
	# create legend for plasticity dynamics
	axes[0].legend(loc="center left")

	if mem_dyn_data:
		# set axis labels for membrane potential and current plots
		axes[1].set_xlabel("Time (min)")
		axes[1].set_ylabel("Membrane potential (mV)")
		axes[1].set_xlim(xmin=xlim_0, xmax=xlim_1, auto=xlim_auto)
		axes[1].set_ylim(-71, -54)
		#axes[1].set_ylim(-67, -63)
		ax1twin = axes[1].twinx() # create twin axis for axes[1]
		ax1twin.set_ylabel("Current (nA)")
		ax1twin.set_ylim(-2, 4)
		
		# plot data for membrane potential and current data
		ax1g1 = axes[1].plot(time, data_stacked[:,data_ptr-2], color="#ff0000", label='Membrane pot.', marker='None', zorder=10)
		ax1g2 = ax1twin.plot(time, data_stacked[:,data_ptr-1], color="#ffee00", label='Membrane curr.', marker='None', zorder=9)
		
		# create common legend for axes[1] and ax1twin
		handles, labels = axes[1].get_legend_handles_labels()
		handles_twin, labels_twin = ax1twin.get_legend_handles_labels()
		axes[1].legend(handles + handles_twin, labels + labels_twin, loc="center left")

	# set axis labels for calcium and protein plots
	axes[num_rows-1].set_xlabel("Time (min)")
	axes[num_rows-1].set_ylabel("Protein concentration (µM)")
	axes[num_rows-1].set_xlim(xmin=xlim_0, xmax=xlim_1, auto=xlim_auto)
	#axes[num_rows-1].set_ylim(-0.1, 15.1)
	axLtwin = axes[num_rows-1].twinx() # create twin axis for axes[num_rows-1]
	axLtwin.set_ylabel("Calcium amount")
	#axLtwin.set_ylabel("SPS amount")
	
	# plot data for sps and protein dynamics
	axLtwin.plot(time, data_stacked[:,data_ptr+2], color="#c8c896", label='Calcium', marker='None', zorder=10)
	axLtwin.plot(time, data_stacked[:,data_ptr+3], color="purple", label='SPS', marker='None', zorder=10)
	axes[num_rows-1].plot(time, data_stacked[:,data_ptr+4], color="#008000", label='Protein', marker='None', zorder=9)
	axLtwin.axhline(y=config["synapses"]["syn_exc_calcium_plasticity"]["theta_p"], label='LTP thresh.', linestyle='dashed', color="#969664", zorder=8)
	axLtwin.axhline(y=config["synapses"]["syn_exc_calcium_plasticity"]["theta_d"], label='LTD thresh.', linestyle='dashed', color="#969696", zorder=7)
	
	# create legend for sps and protein plots
	handles, labels = axes[num_rows-1].get_legend_handles_labels()
	handles_twin, labels_twin = axLtwin.get_legend_handles_labels()
	axes[num_rows-1].legend(handles + handles_twin, labels + labels_twin, loc="center left")
	#axes[num_rows-1].legend(loc="center left")

	# save figure in given format (e.g., 'png' or 'svg')
	fig.savefig(os.path.join(store_path, f"{timestamp}_traces_neuron_{neuron}_synapse_{pre_neuron}to{neuron}.{figure_fmt}"))
	plt.close()

#####################################
# plotRaster
# Plots a spike raster, highlighting different neuronal subpopulations by different colors
# config: configuration parameters in JSON format
# spikes_stacked: two-dimensional array containing the neuron and time of each spike
# timestamp: timestamp of the current trial
# store_path [optional]: path to store resulting graphics file
# figure_fmt [optional]: format of resulting graphics file
def plotRaster(config, spikes_stacked, timestamp, store_path = ".", figure_fmt = 'png'):
	N_CA = int(config["populations"]["N_CA"])
	N_exc = int(config["populations"]["N_exc"]) 
	N_inh = int(config["populations"]["N_inh"])
	N = N_exc + N_inh # total number of neurons

	xlim_0 = 0 # lower limit of x-axis
	xlim_1 = config["simulation"]["runtime"]*1000 # upper limit of x-axis
	
	#plt.axhspan(0, N_CA, color='k', alpha=0.5) # background color
	#plt.axhspan(N_exc, N, color='r', alpha=0.5) # background color
	
	mask_CA = (spikes_stacked[1] < N_CA)
	mask_exc = np.logical_and(spikes_stacked[1] >= N_CA, spikes_stacked[1] < N_exc)
	mask_inh = (spikes_stacked[1] >= N_exc)
	
	marker_type = '.' # ',' 
	marker_size = 1
	plt.plot(spikes_stacked[0][mask_CA], spikes_stacked[1][mask_CA], marker_type, color='blue', markersize=marker_size)
	plt.plot(spikes_stacked[0][mask_exc], spikes_stacked[1][mask_exc], marker_type, color='blue', markersize=marker_size)
	plt.plot(spikes_stacked[0][mask_inh], spikes_stacked[1][mask_inh], marker_type, color='red', markersize=marker_size)
	
	#print("Spikes in the first 500 ms:", len(spikes_stacked[0][spikes_stacked[0] < 500]))
	
	plt.xlim(xlim_0, xlim_1)
	plt.ylim(0, N)
	plt.xlabel('Time (ms)', fontsize=12)
	plt.ylabel('Neuron index', fontsize=12)
	yticklabels=np.linspace(0, N, num=5, endpoint=False)
	plt.yticks(yticklabels, yticklabels, rotation='horizontal')
	plt.gca().set_yticklabels(['{:.0f}'.format(y) for y in yticklabels])	
	plt.tight_layout()
	plt.savefig(os.path.join(store_path, timestamp + '_spike_raster.' + figure_fmt), dpi=800)
	plt.close()
'''
# customized for smallnet2:
def plotRaster(config, spikes_stacked, timestamp, store_path = ".", figure_fmt = 'png'):
	N_CA = int(config["populations"]["N_CA"])
	N_exc = int(config["populations"]["N_exc"]) 
	N_inh = int(config["populations"]["N_inh"])
	N = N_exc + N_inh # total number of neurons

	xlim_0 = 9900 # lower limit of x-axis
	xlim_1 = 11300 #config["simulation"]["runtime"]*1000 # upper limit of x-axis
	
	#plt.axhspan(0, N_CA, color='k', alpha=0.5) # background color
	#plt.axhspan(N_exc, N, color='r', alpha=0.5) # background color
	
	mask_CA = (spikes_stacked[1] < N_CA)
	mask_exc = np.logical_and(spikes_stacked[1] >= N_CA, spikes_stacked[1] < N_exc)
	mask_inh = (spikes_stacked[1] >= N_exc)
	
	marker_type = '.' # ',' 
	marker_size = 1
	plt.plot(spikes_stacked[0][mask_CA], spikes_stacked[1][mask_CA], marker_type, color='blue', markersize=marker_size)
	plt.plot(spikes_stacked[0][mask_exc], spikes_stacked[1][mask_exc], marker_type, color='blue', markersize=marker_size)
	plt.plot(spikes_stacked[0][mask_inh], spikes_stacked[1][mask_inh], marker_type, color='red', markersize=marker_size)
	
	print("Spikes in the first 500 ms:", len(spikes_stacked[0][spikes_stacked[0] < 500]))
	
	plt.xlim(xlim_0, xlim_1)
	plt.ylim(-0.5, N-0.5)
	plt.xlabel('Time (ms)', fontsize=12)
	plt.ylabel('Neuron index', fontsize=12)
	yticklabels=np.linspace(0, N, num=4, endpoint=False)
	plt.yticks(yticklabels, yticklabels, rotation='horizontal')
	plt.gca().set_yticklabels(['{:.0f}'.format(y) for y in yticklabels])	
	plt.tight_layout()
	plt.savefig(os.path.join(store_path, timestamp + '_spike_raster.' + figure_fmt), dpi=800)
	plt.close()
'''

#####################################
# plotDendriticProteinDist
# Plots the distribution of the protein concentration across dendrites
# config: configuration parameters in JSON format
# data_stacked: two-dimensional array containing the values of the membrane potential, weights, calcium amount, etc. over time
# timestamp: timestamp of the current trial
# protein_dist_gid: number of the neuron whose dendritic protein distribution shall be plotted
# phase: simulation phase from which the considered data has been retrieved
# store_path [optional]: path to store resulting graphics file
# figure_fmt [optional]: format of resulting graphics file
def plotDendriticProteinDist(config, data_stacked, timestamp, protein_dist_gid, phase, store_path = ".", figure_fmt = 'png'):

	# assign data arrays
	times = data_stacked[:, 0] / 3600
	data_soma_pc = data_stacked[:, 1]
	data_pss_spsV = data_stacked[:, 2]
	data_pss_pc = data_stacked[:, 3]

	num_dendrite_probes = config["simulation"]["num_dendrite_probes"] # number of probes to set along a dendrite
	fs = np.linspace(0, 1, num_dendrite_probes)
	data_pc_dendrite_A = np.zeros((num_dendrite_probes, len(times)))
	data_pc_dendrite_B = np.zeros((num_dendrite_probes, len(times)))

	for i in range(num_dendrite_probes):
		data_pc_dendrite_A[i] = data_stacked[:, 4+i]
	for i in range(num_dendrite_probes):
		data_pc_dendrite_B[i] = data_stacked[:, 4+num_dendrite_probes+i]

	data_total_spsV = data_stacked[:, 4+num_dendrite_probes*2]
	data_total_pcV = data_stacked[:, 4+num_dendrite_probes*2+1]

	# plot overview over dendritic protein distribution
	fig, axes = plt.subplots(4, sharex=True, figsize=(10, 10))

	axes[0].plot(times, data_pss_pc, label='Protein', color='green', zorder=9)

	ax_twin0 = axes[0].twinx()
	ax_twin0.plot(times, data_total_spsV, label='Signal (summed)', color='red', linestyle="-.", zorder=10)
	ax_twin0.plot(times, data_pss_spsV, label='Signal (estimated)', color='purple', zorder=9)

	ax_twin0.axhline(y=config["synapses"]["syn_exc_calcium_plasticity"]["theta_pro"], label='Norm. PS thresh.', linestyle='dashed', color="#111111", zorder=8)

	axes[1].plot(times, data_soma_pc, label='Protein', color='green')

	ax_twin1 = axes[1].twinx()
	ax_twin1.plot(times, data_total_pcV, label='Protein (summed)', color='black', linestyle="-.", zorder=10)

	for data, f in zip(data_pc_dendrite_A, fs):
		axes[2].plot(times, data, color=plt.cm.viridis(f))

	for data, f in zip(data_pc_dendrite_B, fs):
		axes[3].plot(times, data, color=plt.cm.viridis(f))

	# create legend for sps and protein plots
	handles, labels = axes[0].get_legend_handles_labels()
	handles_twin, labels_twin = ax_twin0.get_legend_handles_labels()
	axes[0].legend(handles + handles_twin, labels + labels_twin, loc="center right")

	handles, labels = axes[1].get_legend_handles_labels()
	handles_twin, labels_twin = ax_twin1.get_legend_handles_labels()
	axes[1].legend(handles + handles_twin, labels + labels_twin, loc="center right")

	axes[0].set_title("Soma, protein synthesis segment")
	axes[1].set_title("Soma, basal end")
	axes[2].set_title("Dendrite A")
	axes[3].set_title("Dendrite B")

	axes[3].set_xlabel("Time (h)")

	axes[0].set_ylabel("Protein concentration (µmol/l)")
	ax_twin0.set_ylabel("Signal amount (1e-21 mol)")
	axes[1].set_ylabel("Protein concentration (µmol/l)")
	ax_twin1.set_ylabel("Protein amount (1e-21 mol)")
	axes[2].set_ylabel("Protein concentration (µmol/l)")
	axes[3].set_ylabel("Protein concentration (µmol/l)")

	#if phase == 2:
	#	axes[0].set_ylim(-0.2, 10.2)
	#	axes[1].set_ylim(-0.2, 10.2)
	#	axes[2].set_ylim(-0.2, 10.2)

	#ax_twin.set_ylim(-0.0005, 0.0045)

	norm1 = matplotlib.colors.Normalize(vmin=0, vmax=config["neuron"]["length_dendrite_1"])
	norm2 = matplotlib.colors.Normalize(vmin=0, vmax=config["neuron"]["length_dendrite_2"])
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=None), ax=axes[0])
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=None), ax=ax_twin0)
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=None), ax=axes[1])
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=None), ax=ax_twin1)
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm1), ax=axes[2])
	fig.colorbar(matplotlib.cm.ScalarMappable(cmap="viridis", norm=norm2), ax=axes[3])
	plt.tight_layout()
	fig.savefig(os.path.join(store_path, f"protein_dist_neuron_{protein_dist_gid}_phase_{phase}.{figure_fmt}"), dpi=800)
	plt.close()
	#plt.show()
	
#####################################
if __name__ == '__main__':
	rawpaths = Path(".")
	timestamp = None

	for path in sorted(rawpaths.iterdir()):

		tpath = os.path.split(str(path))[1] # take tail

		if not path.is_dir() and "_traces.txt" in tpath:
			timestamp = tpath.split("_traces")[0]
			
			config = json.load(open(timestamp + "_config.json", "r"))
			
			data_stacked = np.loadtxt(timestamp + '_traces.txt')
			spikes_stacked = np.loadtxt(timestamp + '_spikes.txt').transpose()
			
			#plotResults(config, data_stacked, timestamp, 0, mem_dyn_data = True, figure_fmt = 'svg')

			sample_gid_list = config['simulation']['sample_gid_list'] # list of the neurons that are to be probed (given by number/gid)
			sample_pre_list = config['simulation']['sample_pre_list'] # list of the presynaptic neurons for probing synapses (one value for each value in `sample_gid`; -1: no synapse probing)
			protein_dist_gid = config['simulation']['protein_dist_gid'] # neuron whose dendritic protein distribution shall be plotted

			for i in range(len(sample_gid_list)):
				sample_gid = sample_gid_list[i]
				sample_presyn_gid = getPresynapticId(sample_pre_list, i)
				plotResults(config, data_stacked, timestamp, i, mem_dyn_data = True, neuron=sample_gid, pre_neuron=sample_presyn_gid, figure_fmt = 'svg')
	
			if spikes_stacked.size != 0:
				plotRaster(config, spikes_stacked, timestamp, figure_fmt = 'png')
			else:
				print("No spike data to plot...")

			for phase in [1, 2, 3]:
				data_stacked = np.loadtxt(f"{timestamp}_protein_dist_neuron_{protein_dist_gid}_phase_{phase}.txt")
				plotDendriticProteinDist(config, data_stacked, timestamp, protein_dist_gid, phase, figure_fmt = 'svg')
