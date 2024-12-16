: Current-based synapse with exponential postsynatic potential; featuring
: early-phase plasticity (transient) based on the postsynaptic calcium amount, and
: late-phase plasticity (permanent/line attractor) with synaptic tagging and capture

NEURON {
	POINT_PROCESS expsyn_curr_early_late_plasticity
	RANGE pc, h_0, tau_syn, Ca_pre, Ca_post, theta_p, theta_d, theta_tag, R_mem, sigma_pl, h_init, z_init, c_morpho, area
	NONSPECIFIC_CURRENT I
	USEION pc_ READ pc_d : common pool of plasticity-related proteins
	USEION sps_ WRITE sps_d : signal triggering protein synthesis
}

PARAMETER {
	: RANGE parameters
	h_0         =   4.20075      : initial weight (in mV)
	tau_syn     =   5.0          : synaptic time constant (in ms)
	Ca_pre      =   1.0          : pre-synaptic calcium contribution
	Ca_post     =   0.2758       : post-synaptic calcium contribution
	theta_p     =   3.0          : calcium threshold for potentiation
	theta_d     =   1.2          : calcium threshold for depression
	theta_tag   =   0.840149     : tagging threshold (in mV)
	R_mem       =   10.0         : membrane resistance (in MOhm)
	sigma_pl    =   2.90436      : standard deviation for plasticity fluctuations (in mV)

	: GLOBAL parameters
	tau_h       =   688400       : early-phase time constant (in ms)
	tau_z       =   3600000      : late-phase time constant (in ms)
	tau_Ca      =   48.8         : calcium time constant (in ms)
	gamma_p     =   1645.6       : potentiation rate
	gamma_d     =   313.1        : depression rate
	t_Ca_delay  =   18.8         : delay of postsynaptic calcium increase after presynaptic spike (in ms)
	f_int       =   0.1          : late-phase integration factor (in l/µmol)

	h_init      =   4.20075      : parameter to set state variable h (in mV)
	z_init      =   0            : parameter to set state variable z

	diam                         : CV diameter in µm (internal variable)
	area                         : CV area of effect in µm^2 (internal variable in newer Arbor versions)

	c_morpho    = 1              : parameter to adjust the impact of EPSPs occurring far away from the soma
}

ASSIGNED {
	volume                       : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
	w                            : total synaptic weight (in mV)
	h_diff_abs_prev              : relative early-phase weight in the previous timestep (absolute value, in mV)
}

STATE {
	psp                          : instant value of postsynaptic potential (in mV)
	h                            : early-phase weight (in mV)
	z                            : late-phase weight
	Ca                           : calcium concentration
	pc                           : common pool of plasticity-related proteins (concentration, in µmol/l or mmol/l or ...)
}

WHITE_NOISE {
	zeta
}

INITIAL {
	volume = area*diam/4 : volume of the CV (= area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h)
	psp  = 0
	h = h_init
	z = z_init
	Ca = 0
	sps_d = 0

	w = h_init + z_init*h_0
	h_diff_abs_prev = 0
	pc = pc_d
}

BREAKPOINT {
	SOLVE state METHOD stochastic
	LOCAL h_diff_abs

	: Set postsynaptic current
	w = (h + z*h_0) * c_morpho
	I = -psp / R_mem

	: Update the signal triggering protein synthesis
	h_diff_abs = fabs(h - h_0) : current relative early-phase weight (absolute value)
	sps_d = sps_d + (h_diff_abs - h_diff_abs_prev) * area / volume / 1000 : for a description of the normalization, see https://github.com/arbor-sim/arbor/issues/2084#issuecomment-1498037774
	h_diff_abs_prev = h_diff_abs

	: Get the protein concentration
	pc = pc_d
}

DERIVATIVE state {
	LOCAL h_diff : relative early-phase weight (in mV)
	LOCAL xi : additive noise (in mV)
	LOCAL ltp_switch, ltd_switch

	: Exponential decay of postsynaptic potential
	psp' = - psp / tau_syn

	: Get the relative early-phase weight and other helper variables
    h_diff = h - h_0
	xi = sqrt(tau_h * (ltp_switch + ltd_switch)) * sigma_pl * zeta
	ltp_switch = step_right(Ca - theta_p)
	ltd_switch = step_right(Ca - theta_d)
	
	: Early-phase dynamics
	h' = (- 0.1 * h_diff + gamma_p * (10 - h) * ltp_switch - gamma_d * h * ltd_switch + xi) / tau_h
	
	: Late-phase dynamics
	z' = pc * f_int * ((1 - z) * step_right(h_diff - theta_tag) - (z + 0.5) * step_right(- h_diff - theta_tag)) / tau_z
	
	: Exponential decay of calcium concentration
	Ca' = - Ca / tau_Ca
}

NET_RECEIVE(weight) {
	if (weight >= 0) {
		: Start of postsynaptic potential
		psp = psp + w
	}
	else {
		: Increase of calcium amount by presynaptic spike
		Ca = Ca + Ca_pre
	}
}

POST_EVENT(time) {
	: Increase of calcium amount by postsynaptic spike
	Ca = Ca + Ca_post
}

