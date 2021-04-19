# -*- coding: utf-8 -*-
""" Import user inputs and initialize cantera objects, dictionaries for
parameters, pointers, etc, and set up the initial solution vector arrays"""
"-------------------------------------------------------------------------"

# Import modules:
import numpy as np
from matplotlib import pyplot as plt
import cantera as ct

# Import user inputs:
from sei_1d_inputs import *

"""----------Create Cantera objects----------"""
CE, elyte, sei, sei_conductor, WE = ct.import_phases(ctifile, \
    [CE_phase, elyte_phase, sei_phase, sei_conductorphase, WE_phase])
WE_elyte = ct.Interface(ctifile, WE_elyte_surfphase, [WE, elyte, sei])
WE_sei =  ct.Interface(ctifile, WE_sei_surfphase, \
    [WE, sei, sei_conductor])
sei_elyte = ct.Interface(ctifile, sei_elyte_surfphase, \
    [sei, elyte, sei_conductor])
CE_elyte = ct.Interface(ctifile, CE_surfphase, [CE, elyte])

print('\n     Cantera phases created. \n')

"""----------Set phase properties for cantera objects----------"""
TP_o = T_0, P_0
elyte.TP = TP_o
sei.TP = TP_o
WE.TP = TP_o
CE.TP = TP_o
WE_elyte.TP = TP_o
WE_sei.TP = TP_o
sei_elyte.TP = TP_o
sei_conductor.TP = TP_o
CE_elyte.TP = TP_o

elyte.electric_potential = phi_elyte_0
sei.electric_potential = phi_SEI_dl_0
sei_conductor.electric_potential = phi_SEI_dl_0
WE.electric_potential = phi_0

# Store these objects in a common dictionary:
objs = {'WE':WE, 'SEI':sei, 'elyte':elyte, 'CE':CE, 'conductor':sei_conductor, \
    'WE_SEI':WE_sei, 'WE_elyte':WE_elyte, 'SEI_elyte':sei_elyte, \
    'CE_Elyte':CE_elyte}

"""----------Define electrolyte species----------"""

# ---------------------------------------------------------------------------
# The electrolyte information is brought in from an input file using Cantera.
# ---------------------------------------------------------------------------
C_k_elyte = elyte.density_mole*elyte.X
print("The species in the electrolyte are:")
print('\n'.join(elyte.species_names))
print('\n')
"""----------Define SEI species----------"""

# ---------------------------------------------------------------------------
# The SEI information is brought in from an input file using Cantera.
# ---------------------------------------------------------------------------

num_SEI_species = sei.n_species
C_k_sei = sei.concentrations
#C_k_sei[sei.species_index('LEDC[SEI]')] = 1e-6
#C_k_sei[sei.species_index('Li2O[SEI]')] = 1e-6
#C_k_sei[sei.species_index('Li2CO3[SEI]')] = 1e-6

print("The species in the SEI are:")
print('\n'.join(sei.species_names))

# %% Initializing solution vector

# Set preliminary parameters for the anode voltage sweep function
phi_bounds = np.array([phi_1, phi_2])  # Upper and lower voltage bounds
R = sweep_rate                            # Sweep rate [V/s]

# Times for discontinuities (sweep sign change) in anode voltage
t_event0 = -sweep_dirn_0*(phi_0 - phi_bounds[int(0.5*(1. + sweep_dirn_0))])/(R)

if init_hold == 1:
    t_event0 = t_event0 + t_hold_0

dt = (phi_bounds[1] - phi_bounds[0])/R
t_events = np.arange(t_event0,t_event0+dt*(2*n_cycles+1),dt)
times = np.concatenate((np.array([0.]),t_events),)

t_hold_init = times[-1] + 1e-6
t_hold_final = times[-1] + t_hold

times = np.concatenate((times,(t_hold_init,),(t_hold_final,)))

voltage_array = np.zeros_like(times)
voltage_array[0] = phi_0

if init_hold == 1:
    voltage_array[1] = phi_0

direction = sweep_dirn_0
for i, t in enumerate(t_events):

    if init_hold == 1:
        voltage_array[i + 2] = voltage_array[i+1] + direction * (t - times[i+1]) * R
    else:
        voltage_array[i + 1] = voltage_array[i] + direction * (t - times[i]) * R
    direction *= -1

voltage_array[-2:] = phi_hold

#print(voltage_array)

t = np.linspace(0,times[-1],500)
v = np.interp(t,times,voltage_array)

if check_profile:
    print('Check that voltage profile is correct.  Cancel simulation, if not. \n')
    plt.figure()
    plt.plot(t,v)
    plt.xlabel('time (s)')
    plt.ylabel(r"$\Phi_{WE} - \Phi_{CE}$" ' (V)')
    plt.show()


"""----------Set up solution vector pointer SVptr----------"""
SVptr = {}

if mode == 'detailed':
    """----------Geometry calcs----------"""
    dx = x/N_x      # USER INPUT length of step in x direction
    dy = d_sei
    N_y  = int(y/dy)     # USER INPUT length of step in y direction

    """     sei/elyte domain    """
    # 1 variable for sei volume fraction
    # 1 variable for sei electric potential
    # 1 variable for elyte electric potential
    # 1 variable for each species in sei and elyte
    nvars_node = 3 + sei.n_species + elyte.n_species

    SVptr['phi sei'] = np.arange(0,nvars_node*N_y,nvars_node,dtype='int')
    SVptr['phi elyte'] = np.arange(1,nvars_node*N_y,nvars_node,dtype='int')
    SVptr['eps sei'] = np.arange(2,1+nvars_node*N_y,nvars_node,dtype='int')
    SVptr['Ck sei'] = np.ndarray(shape=(N_y,sei.n_species),dtype='int')
    SVptr['Ck elyte'] = np.ndarray(shape=(N_y,elyte.n_species),dtype='int')
    for i in range(N_y):
        SVptr['Ck sei'][i,:] = np.arange(3+i*nvars_node,\
            3+i*nvars_node+sei.n_species,dtype=int)
        SVptr['Ck elyte'][i,:] = range(3+i*nvars_node+sei.n_species,\
            3+i*nvars_node+sei.n_species+elyte.n_species)


    # ---------------------------------------------------------------------------
    # Now use the total number of tracked variables per node to calculate the
    # length of the solution vector based on discretization and tracked variables
    # ---------------------------------------------------------------------------
    nvars_tot = N_x*(1 + N_y*nvars_node)

    eps_0 += 1e-6

    # ---------------------------------------------------------------------------
    # Initialize solution vector using np.zeros based on number of the variables
    # ---------------------------------------------------------------------------
    # SV_node = np.concatenate((np.array((phi_SEI_dl_0, phi_elyte_0, eps_0)), C_k_sei, C_k_elyte))
    SV_node = np.concatenate((np.array((phi_0+phi_SEI_dl_0, phi_elyte_0, float(eps_0))), C_k_sei*eps_0, C_k_elyte))
    SV_0 = np.tile(SV_node, N_y)
    SV_dot_0 = np.zeros_like(SV_0)
    res = np.zeros_like(SV_0)

    algvar = np.ones_like(SV_0)
    algvar[SVptr['phi elyte']] = 0.

    D_SEI = np.zeros((num_SEI_species,num_SEI_species))
    # Any way to do this with a pointer to the Li+ species?
    D_SEI[3:] = D_Li_I

    # Trying to get species charge in params
   # zk_elyte = np.zeros_like(elyte.X)
   # for i in elyte.n_species:
  #      zk_elyte[i]=elyte.species(i).charge


    params = {'phi bounds':phi_bounds, 'Rate':R, 'Ny':N_y, 'dyInv':1./dy, \
        'd_sei':d_sei, 'TP':TP_o, 'C_dl WE_sei':C_dl_WE_SEI, 'sigma sei':sigma_el, 'D_SEI':D_SEI}

elif mode == 'homogeneous':
    SVptr['phi sei-we'] = 0
    SVptr['phi sei-elyte'] = 1
    SVptr['thickness'] = 2
    SVptr['Ck sei'] = np.arange(3,3+sei.n_species)

    SV_0 = np.concatenate((np.array((phi_SEI_dl_0, phi_0+phi_SEI_dl_0, t_0)), C_k_sei))
    SV_dot_0 = np.zeros_like(SV_0)
    res = np.zeros_like(SV_0)

    params = {'phi bounds':phi_bounds, 'Rate':R, 'TP':TP_o, \
        'C_dl WE_sei':C_dl_WE_SEI, 'sigma sei':sigma_el}

elif mode == 'reduced':
    SVptr['thickness'] = 0
    SVptr['Ck sei'] = np.arange(1,1+sei.n_species)

    SV_0 = np.concatenate((np.array((t_0,)), C_k_sei))
    SV_dot_0 = np.zeros_like(SV_0)
    res = np.zeros_like(SV_0)

    params = {'phi bounds':phi_bounds, 'Rate':R, 'TP':TP_o}

#vol_k_sei = sei.molecular_weights/rho_k_SEI

voltage_lookup = {'time':times, 'voltage':voltage_array}
