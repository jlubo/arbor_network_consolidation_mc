NEURON {
    SUFFIX protein_synthesis
	RANGE theta, alpha, area
	USEION pc_ WRITE pc_d : common pool of plasticity-related proteins
	USEION sps_ READ sps_d : signal triggering protein synthesis
}

PARAMETER {
	theta     =   1         : threshold for protein synthesis (refers to SPS concentration, in µmol/l or mmol/l or ...)
	alpha     =   10        : protein synthesis scaling constant (in 1e-18 mol if concentration is in mmol/l)
	tau_p     =   3600000   : time constant for protein dynamics (in ms)

	area                    : surface area of the CV (internal variable in newer Arbor versions, in µm^2)
	diam                    : CV diameter (internal variable, in µm)
}

ASSIGNED {
	volume                  : volume of the CV (conversion factor between concentration and particle amount, in µm^3)
}

INITIAL {
	volume = area*diam/4 : volume of the CV (= area*r/2 = 2*pi*r*h*r/2 = pi*r^2*h)

	: set the initial protein concentration and amount
	:pc_d = pc_init : not done here because concentration is already set in overpainted 'lif' mechanism
}

BREAKPOINT {
	SOLVE state METHOD cnexp
}

DERIVATIVE state {
	pc_d' = alpha/volume * step_right( sps_d - theta ) / tau_p : equation describing the concentration (an by that, the amount) of protein
}
