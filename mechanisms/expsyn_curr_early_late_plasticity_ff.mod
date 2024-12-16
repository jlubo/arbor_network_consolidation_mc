: Synapse similar to 'expsyn_curr_early_late_plasticity', 
: but for fast-forward computation
: with late-phase dynamics only

NEURON {
	POINT_PROCESS expsyn_curr_early_late_plasticity_ff
	RANGE pc, h_0, theta_tag, h_init, z_init, area
	NONSPECIFIC_CURRENT I
	USEION pc_ READ pc_d : common pool of plasticity-related proteins
	USEION sps_ WRITE sps_d : signal triggering protein synthesis
}

PARAMETER {
	: RANGE parameters
	h_0         =   4.20075      : initial weight (in mV)
	theta_tag   =   0.840149     : tagging threshold (in mV)
	R_mem       =   10.0         : membrane resistance (in MOhm)

	: GLOBAL parameters
	tau_h       =   688400       : early-phase time constant (in ms)
	tau_z       =   3600000      : late-phase time constant (in ms)
	f_int       =   0.1          : late-phase integration factor (in l/µmol)

	h_init      =   4.20075      : parameter to set state variable h (in mV)
	z_init      =   0            : parameter to set state variable z

	diam                         : CV diameter (internal variable, in µm)
	area                         : CV area of effect (internal variable in newer Arbor versions, in µm^2)
}

ASSIGNED {
	volume                       : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
	w                            : total synaptic weight (in mV)
	h_diff_abs_prev              : relative early-phase weight in the previous timestep (absolute value)
}

STATE {
	h                            : early-phase weight (in mV)
	z                            : late-phase weight
	Ca                           : calcium concentration (only here so that probing does not fail)
	pc                           : common pool of plasticity-related proteins (concentration, in µmol/l or mmol/l or ...)
}

INITIAL {
	volume = area*diam/4 : volume of the CV (= area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h)
	h = h_init
	z = z_init
	Ca = 0
	sps_d = 0

	h_diff_abs_prev = 0
	pc = pc_d
}

BREAKPOINT {
	:SOLVE state METHOD cnexp : is not compatible with late-phase equation
	SOLVE state METHOD sparse : is not compatible with using 'pc_d' directly in DERIVATIVE block
	LOCAL h_diff_abs

	I = 0

	: Update the signal triggering protein synthesis
	h_diff_abs = fabs(h - h_0) : current relative early-phase weight (absolute value)
	sps_d = sps_d + (h_diff_abs - h_diff_abs_prev) * area / volume / 1000 : for a description of the normalization, see https://github.com/arbor-sim/arbor/issues/2084#issuecomment-1498037774
	h_diff_abs_prev = h_diff_abs

	: Get the protein concentration
	pc = pc_d
}

DERIVATIVE state {
	LOCAL h_diff : relative early-phase weight (in mV)

	: Get relative early-phase weight
    h_diff = h - h_0

	: Early-phase decay
	h' = (- 0.1 * h_diff) / tau_h
	
	: Late-phase dynamics
	z' = pc * f_int * ((1 - z) * step_right(h_diff - theta_tag) - (z + 0.5) * step_right(- h_diff - theta_tag)) / tau_z
}

NET_RECEIVE(weight) {

}

POST_EVENT(time) {

}
