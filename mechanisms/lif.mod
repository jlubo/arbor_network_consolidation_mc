: Leaky-integrate and fire (LIF) neuron with protein pool
: based on the LIF mechanism written by Sebastian Schmitt (https://github.com/tetzlab/FIPPA/blob/main/STDP/mechanisms/lif.mod)

NEURON {
	SUFFIX lif
	RANGE R_reset, R_leak, V_reset, V_rev, V_th, I_0, pc_init, i_factor, t_ref
	NONSPECIFIC_CURRENT i
	USEION pc_ WRITE pc_d : common pool of plasticity-related proteins
	USEION sps_ READ sps_d : signal triggering protein synthesis
}

PARAMETER {
	R_leak    =   10              : membrane resistance during leakage (standard case, in MOhm)
	R_reset   =   1e-10           : membrane resistance during refractory period (very small to yield fast reset, in MOhm)
	I_0       =   0               : constant input current (in nA)
	i_factor  =   1               : current to current density conversion factor (nA to mA/cm^2; for point neurons)
	tau_p     =   3600000         : time constant for protein dynamics (in ms)

	V_reset   =  -70              : reset potential (in mV)
	V_rev     =  -65              : reversal potential (in mV)
	V_th      =  -55              : threshold potential to be crossed for spiking (in mV)
	t_ref     =   2               : refractory period (in ms)
	
	pc_init   =   0               : parameter to set protein concentration (in µmol/l or mmol/l or ...)
}

ASSIGNED {
}

STATE {
	refractory_counter            : counter for refractory period
}

INITIAL {
	refractory_counter = t_ref + 1 : start not refractory

	: set the initial protein concentration (in µmol/l or mmol/l or ...)
	pc_d = pc_init

	: set the initial concentration of the signal to trigger protein synthesis (in µmol/l or mmol/l or ...) 
    : - is initialized in synapse mechanisms
	:sps_d = 0
}

BREAKPOINT {
	SOLVE state METHOD cnexp

	LOCAL R_mem
	LOCAL E
		
	: threshold crossed -> start refractory counter
	if (v > V_th) {
	   refractory_counter = 0
	}

	: choose between leak and reset potential
	if (refractory_counter <= t_ref) { : in refractory period - strong drive of v towards V_reset
		R_mem = R_reset
		E = V_reset
	} else { : outside refractory period
		R_mem = R_leak
		E = V_rev
	}

	: current density in units of mA/cm^2
	i = (((v - E) / R_mem) - I_0) * i_factor
}

DERIVATIVE state {
	if (refractory_counter <= t_ref) {
		refractory_counter' = 1
	}
	pc_d' = -pc_d / tau_p
}

