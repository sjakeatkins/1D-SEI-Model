# -*- coding: utf-8 -*-
"""
User inputs for 1-D SEI model.
"""
"-------------------------------------------------------------------------"
import numpy as np

# Choose 'detailed', 'homogeneous', or 'reduced':
mode = 'detailed'

"Identify cti file:"
#ctifile = 'W_anode_chem_07242019.cti'
ctifile = 'W_anode_chem_Carelli_detailed.cti'

# Phase names in cti file
elyte_phase = 'electrolyte'
sei_phase = 'SEI'
sei_conductorphase = 'conductor'
WE_phase = 'tungsten'
WE_elyte_surfphase = 'tungsten_electrolyte_surf'
WE_sei_surfphase = 'tungsten_SEI_surf'
sei_elyte_surfphase = 'SEI_electrolyte_surf'
CE_phase = 'Lithium'
CE_surfphase = 'Li_surf'

"Optional: give a name to the output folder"
save_name = 'DOE_report_March312021'

"""----------Define grid dimensions----------"""

# A (2,10) grid will be indexed as shown below with each node having the
# number of tracked variables stored for it. The solution vector will be
# a row vector progressing through each tracked variable for each cell, then
# progressing in increasing column number for increasing row number.
#
#          |--> electrolyte
#          |
#          |-----------------------------------------------------------|
#          | 0,0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 |
# electrode|-----------------------------------------------------------|
#          | 1,0 | 1,1 | 1,2 | 1,3 | 1,4 | 1,5 | 1,6 | 1,7 | 1,8 | 1,9 |
#          |-----------------------------------------------------------|
#
# The properties of species will be stored in the order shown below for each
# cell.
#
"""----------Initial state variables----------"""
T_0 = np.array([300.])      # USER INPUT initial temperature [K], assumed uniform
P_0 = 101325.               # USER INPUT defines initial pressure.

if mode == 'detailed':
    N_x = 1         # USER INPUT number of grids in plane of electrode
    x = 1           # USER INPUT x length of domain [m]
    y = 2.0e-8#2.4e-8        # USER INPUT y length of domain [m]

    #   DON'T TOUCH
    d_sei = 1e-9    # USER INPUT d_SEI representative diameter of SEI grain [m]

    eps_0 = np.array([0.])      # initial volume fraction of SEI
elif mode == 'homogeneous' or mode == 'reduced':

    t_0 = 1e-11     # Initial thickness [m]



"""----------Define CV parameters:----------"""
sweep_rate = 0.01  #...Voltage sweep rate [V/s]
#sweep_rate = 0.005  #...Voltage sweep rate [V/s]
sweep_dirn_0 = -1
#phi_0 = 1.0
phi_0 = 2.65
#phi_1 = 0.05
phi_1 = 0.25
phi_2 = 1.5
##phi_1 = 2.65
##phi_2 = 0.25
#n_cycles = 10.
n_cycles = 0.
#n_cycles = 1

t_hold_0 = 20*3600
t_hold = 20*3600
phi_hold = 0.25

# If you want to verify that the electric potential input looks correct before
#     running the simulation, switch this to '1'
check_profile = 0

# Toggle initial voltage hold by switching this to '1'

init_hold = 0

#...Initial electric potential of SEI, relative to WE:
phi_SEI_dl_0 = 0.
#...Initial electric potential of electrolyte:
phi_elyte_0 = 0.

"""----------SEI properties:----------"""
# Mass density [kg of k per m3 of k] requires knowing order of species in cti
#   file.
rho_k_SEI = [2110, 2013, 1321, 534]
# Electrical Conductivity [S/m]
sigma_scale = 1e-4
sigma_el = [sigma_scale*2*1.5e-8, sigma_scale*8e-8, sigma_scale*13e-8, 0] #[1.5e-8, 4e-8, 9e-8]

# Double layer capacitances
C_dl_WE_SEI = 2e-7  # F/m2

# Diff coeffs for Li+ in SEI

D_Li_I = [10e-12, 3.3e-16, 4.01e-16, 0]