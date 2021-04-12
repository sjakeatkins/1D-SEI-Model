# -*- coding: utf-8 -*-

import numpy as np
import cantera as ct


"""----------Residual function for IDA solver----------"""
def residual_detailed(t, SV, SV_dot):
    from sei_1d_init import objs, params, voltage_lookup, SVptr

    "TEMPORARY"
    eps_sei_max = 0.9
    # Initialize residual equal to all zeros:
    res = SV_dot

    # Read out cantera objects:
    WE = objs['WE']
    sei = objs['SEI']
    elyte = objs['elyte']
    sei_elyte = objs['SEI_elyte']
    sei_conductor = objs['conductor']
    WE_sei = objs['WE_SEI']
    WE_elyte = objs['WE_elyte']

    phi_WE = np.interp(t,voltage_lookup['time'],voltage_lookup['voltage'])

    i_sei = np.zeros(params['Ny']+1,)
    i_Far = np.zeros(params['Ny'],)

    # Set anode and anode/electrolyte interface potentials
    WE.electric_potential = phi_WE

    # Start in the volume adjacent to the working electrode:
    j=0

    # Electrolyte electric potential:
    phi_elyte_loc = SV[SVptr['phi elyte'][j]]
    # SEI electric potential:
    phi_sei_loc = phi_elyte_loc + SV[SVptr['phi sei'][j]]
    sei.electric_potential = phi_sei_loc
    sei_conductor.electric_potential = phi_sei_loc
    
    # Electrolyte electric potential assumed to be zero:
    #elyte.electric_potential = 0.


    # SEI volume fraction:
    eps_sei_loc = SV[SVptr['eps sei'][j]]
    eps_elyte_loc = 1. - eps_sei_loc

    # The current in the SEI entering this voluem is that produced by
    #   charge-transfer reactions at the WE-SEI interface:
    i_sei[j] = eps_sei_loc*WE_sei.get_net_production_rates(WE)*ct.faraday

    # sei-electrolyte area per unit volume.  This is scaled by
    #   (1 - eps_sei)*eps_sei so that available area goes to zero
    #   as the sei volume fraction approaches either zero or one:
    sei_APV = (eps_sei_max - eps_sei_loc) * \
        (params['dyInv'] + 4.*eps_sei_loc/params['d_sei'])

    # Array of molar fluxes (kmol/m2/s) and ionic current (A/m2) at WE/elyte BC
    N_k_in = eps_elyte_loc*WE_elyte.get_net_production_rates(elyte)

    # Initialize array of molar fluxes (kmol/m2/s) out of the volume:
    N_k_out = np.zeros_like(N_k_in)

    i_io = np.zeros(params['Ny'] + 1, )

    # Loop through the remaining volumes (except for the very last one):
    for j in range(params['Ny']-1):

        # Import diffusion coefficient calculator:
     #   if params['transport'] == 'cst':
     #       from functions.diffusion_coeffs import cst as diff_coeffs
     #   elif params['transport'] == 'dst':
     #       from functions.diffusion_coeffs import dst as diff_coeffs
     #   else:
     #       raise Exception('Please specify a valid transport model: cst or dst')

        # Read out local SEI composition and set Cantera object:
        Ck_sei_loc = SV[SVptr['Ck sei'][j]]
        Ck_elyte_loc = SV[SVptr['Ck elyte'][j]]
        Ck_elyte_next = SV[SVptr['Ck elyte'][j+1]]
        #rho_sei_loc = abs(np.dot(Ck_sei_loc,sei.molecular_weights))

        #Xk_sei_loc = Ck_sei_loc / sum(Ck_sei_loc)
        # NAN again
        if sum(Ck_sei_loc)>0.:
            Xk_sei_loc = Ck_sei_loc/sum(Ck_sei_loc)
        else:
            Xk_sei_loc = np.ones_like(Ck_sei_loc)*1e-12
        # TRY SMALL NUMBER INSTEAD OF ZERO

        #Xk_elyte_loc = Ck_elyte_loc / sum(Ck_elyte_loc)
        if sum(Ck_elyte_loc) > 0.:
            Xk_elyte_loc = Ck_elyte_loc/sum(Ck_elyte_loc)
            Xk_elyte_next = Ck_elyte_next/sum(Ck_elyte_next)
        else:
            Xk_elyte_loc = np.ones_like(Ck_elyte_loc)*1e-12
            Xk_elyte_next = np.ones_like(Ck_elyte_next)*1e-12

        sei.X = Xk_sei_loc
        elyte.X = Xk_elyte_loc

        # Electrolyte electric potential assumed to be zero:
        # elyte.electric_potential = 0.
        # Replacing...

        phi_elyte_loc = SV[SVptr['phi elyte'][j]]
        phi_elyte_next = SV[SVptr['phi elyte'][j + 1]]
        elyte.electric_potential = phi_elyte_loc

        # SEI electric potential:
        # phi_sei_loc = SV[SVptr['phi sei'][j]] + phi_elyte_loc
        sei.electric_potential = phi_sei_loc
        sei_conductor.electric_potential = phi_sei_loc

        # Production rates from chemical reactions at sei-electrolyte interface:
        Rates_sei_elyte = sei_elyte.get_net_production_rates(sei)*sei_APV
        Rates_elyte_sei = sei_elyte.get_net_production_rates(elyte)*sei_APV

        # Production rates from homogeneous chemical reactions (NOT IMPLEMENTED):
        Rates_elyte = np.zeros_like(SV_dot[SVptr['Ck elyte'][j]])
        Rates_sei = np.zeros_like(SV_dot[SVptr['Ck sei'][j]])
        # Rates_elyte = elyte.get_net_production_rates(elyte)
        # Rates_sei = sei.get_net_production_rates(sei)

        # sei electric potential at next volume:
        phi_sei_next = SV[SVptr['phi sei'][j+1]] + phi_elyte_next

        # Sei & elyte volume fractions in next volume:
        eps_sei_next = SV[SVptr['eps sei'][j+1]]
        eps_elyte_next = 1. - eps_sei_next

        # Concentration and volume fracion at interface between nodes:
        C_k_elyte_int = 0.5 * (Ck_elyte_loc + Ck_elyte_next)
        eps_elyte_int = 0.5 * (eps_elyte_loc + eps_elyte_next)
        eps_sei_int = 0.5 * (eps_sei_loc + eps_sei_next)

        # Elyte species transport
        brugg = 2.0#1.5
        # TODO add (phi_elyte_loc - phi_elyte_next)*Ck*zk*F/R/T to "grad_Ck_elyte" (also a product with dyInv) and rename
        # appropriately.  The expression for N_k_out will then be electro-diffusive flux
        # resolve phi_elyte using i_dl and phi_sei. dont forget to remove phi_elyte=0 line
        D_o = np.ones_like(SV_dot[SVptr['Ck elyte'][j]])*(10.**-10.)
        D_o[3] *= 0.01
        delta_Ck_elyte = (Ck_elyte_loc/eps_elyte_loc 
            - Ck_elyte_next/eps_elyte_next)
        delta_phi = phi_elyte_loc - phi_elyte_next
        D_eff_elyte = D_o*eps_elyte_int**brugg
        # migr_coeff = (np.multiply(C_k_elyte_int,elyte.charges)
        #     *ct.faraday/ct.gas_constant/elyte.T)
        migr_coeff = (np.multiply(C_k_elyte_int,[0,0,1,-1,0,0,0,0,0])
            *ct.faraday/ct.gas_constant/elyte.T)
        flux_gradient = (delta_Ck_elyte + migr_coeff*delta_phi)*params['dyInv']

        N_k_out = np.multiply(D_eff_elyte,flux_gradient)

        # --Li interstitial diffusion flux--
        # c_Li_I_Vo is the interstitial concentration at the anode-sei interface for an anode potential of 0 vs Li/Li+
        # ... it is a model parameter
        c_Li_I_Vo = np.ones_like(SV_dot[SVptr['Ck sei'][j]])*8.7e-7
        c_Li_I_o = np.multiply(c_Li_I_Vo,SV[SVptr['eps sei'][0]])*np.exp(-1*ct.faraday*WE.electric_potential/ct.gas_constant/WE.T)
        # Assumption: Li interstitials are consumed by fast SEI formation rxns @ sei-elyte interface
        c_Li_I_loc = np.zeros_like(c_Li_I_o)
        D_Li_I = np.ones_like(SV_dot[SVptr['Ck sei'][j]])*10e-12
        N_k_in[2] = N_k_in[2] - ct.faraday * np.dot(D_Li_I,(c_Li_I_loc-c_Li_I_o)/(j+1)/params['d_sei']/2)


        # # Elyte species transport
        # brugg = 1.5
        # # TODO add (phi_elyte_loc - phi_elyte_next)*Ck*zk*F/R/T to "grad_Ck_elyte" (also a product with dyInv) and rename
        # # appropriately.  The expression for N_k_out will then be electro-diffusive flux
        # # resolve phi_elyte using i_dl and phi_sei. dont forget to remove phi_elyte=0 line
        # # grad_Ck_elyte = (Ck_elyte_loc - Ck_elyte_next)*params['dyInv']
        # # ed_term = (phi_elyte_loc - phi_elyte_next)*np.multiply(C_k_elyte_int,elyte.charges)*ct.faraday/ct.gas_constant/elyte.T*params['dyInv']
        # # D_scale = 1
        # # Deff_elyte = np.ones_like(SV_dot[SVptr['Ck elyte'][j]])*(D_scale*10.**-10.)*(eps_elyte_int**brugg)
        # # no_coeff = grad_Ck_elyte + ed_term

        # # N_k_out = np.multiply(Deff_elyte,no_coeff)

        grad_Flux_elyte = (N_k_in - N_k_out)*params['dyInv']

        # i_io[j+1] = ct.faraday*np.dot(elyte.charges,N_k_out)
        i_io[j+1] = ct.faraday*np.dot([0,0,1,-1,0,0,0,0,0],N_k_out)

        # Calculate residual for chemical molar concentrations:
        dSVdt_ck_sei = Rates_sei_elyte + Rates_sei
        res[SVptr['Ck sei'][j]] = SV_dot[SVptr['Ck sei'][j]] - dSVdt_ck_sei

        #Test the git add function
        # Calculate residual for sei volume fraction:
        dSVdt_eps_sei = np.dot(dSVdt_ck_sei, sei.partial_molar_volumes)
        #rint(dSVdt_eps_sei)
        #fds
        res[SVptr['eps sei'][j]] = SV_dot[SVptr['eps sei'][j]] - dSVdt_eps_sei

        dSVdt_ck_elyte = Rates_elyte_sei + Rates_elyte + grad_Flux_elyte
        res[SVptr['Ck elyte'][j]] = SV_dot[SVptr['Ck elyte'][j]] - dSVdt_ck_elyte

        # Calculate faradaic current density due to charge transfer at SEI-elyte
        #   interface, in A/m2 total.
        Rates_sei_conductor = sei_elyte.get_net_production_rates(sei_conductor)
        i_Far[j] = Rates_sei_conductor*sei_APV*ct.faraday/params['dyInv']

        # Current = (Conductivity)*(volume fraction)*(-grad(Phi))
        dPhi = phi_sei_loc - phi_sei_next
        vol_k = sei.X * sei.partial_molar_volumes
        vol_tot = np.dot(sei.X, sei.partial_molar_volumes)
        vol_fracs = vol_k / vol_tot
        sigma_sei = np.dot(params['sigma sei'],vol_fracs)
        i_sei[j+1] = eps_sei_int*sigma_sei*dPhi*params['dyInv']

        # sei-electrolyte area per unit volume.  This is scaled by
        #   (1 - eps_sei)*eps_sei so that available area goes to zero
        #   as the sei volume fraction approaches either zero or one:
        # sei_APV = ((1. - eps_sei_next) * 
        #     (eps_sei_loc*params['dyInv'] + 4.*eps_sei_int/params['d_sei']))
        sei_APV = ((eps_sei_max - eps_sei_next) * 
            (params['dyInv'] + 4.*eps_sei_int/params['d_sei']))

            

        eps_sei_loc = eps_sei_next
        eps_elyte_loc = eps_elyte_next
        N_k_in = N_k_out
        phi_sei_loc = phi_sei_next
        phi_elyte_loc = phi_elyte_next

    # Repeat calculations for final node, where the boundary condition is
    #   that i_sei = 0 at the interface with the electrolyte:
    j = int(params['Ny']-1)
    Ck_sei_loc = SV[SVptr['Ck sei'][j]]
    #Xk_sei_loc = Ck_sei_loc / sum(Ck_sei_loc)
    if sum(Ck_sei_loc) > 0.:
        Xk_sei_loc = Ck_sei_loc / sum(Ck_sei_loc)
    else:
        Xk_sei_loc = np.ones_like(Ck_sei_loc) * 1e-12
    Ck_elyte_loc = SV[SVptr['Ck elyte'][j]]
    #Xk_elyte_loc = Ck_elyte_loc / sum(Ck_elyte_loc)
    if sum(Ck_elyte_loc) > 0.:
        Xk_elyte_loc = Ck_elyte_loc / sum(Ck_elyte_loc)
    else:
        Xk_elyte_loc = np.ones_like(Ck_elyte_loc) * 1e-12

    sei.X = Xk_sei_loc
    elyte.X = Xk_elyte_loc

    #elyte.electric_potential = 0.
    # phi_elyte_loc = SV[SVptr['phi elyte'][j]]
    elyte.electric_potential = phi_elyte_loc

    # phi_sei_loc =  SV[SVptr['phi sei'][j]] + phi_elyte_loc
    sei.electric_potential = phi_sei_loc
    sei_conductor.electric_potential = phi_sei_loc

    # SEI surface Area Per unit Volume (APV)
    # sei_APV = 4.*eps_sei_loc*(1-eps_sei_loc)**2/params['d_sei']
    Rates_sei_elyte = sei_elyte.get_net_production_rates(sei)*sei_APV
    #vv
    Rates_sei = np.zeros_like(SV_dot[SVptr['Ck sei'][j]])*SV[SVptr['eps sei'][j]]
    # Rates_sei = sei.get_net_production_rates(sei)*SV[SVptr['eps sei'][j]]
    #^^ check proper implementation of multiplication by volume fraction of phase
    dSVdt_ck_sei = Rates_sei_elyte + Rates_sei
    res[SVptr['Ck sei'][j]] = SV_dot[SVptr['Ck sei'][j]] - dSVdt_ck_sei

    # Repeat calculations for elyte chemistry in final volume (has caused problems)
    Rates_elyte_sei = sei_elyte.get_net_production_rates(elyte) * sei_APV
    #vv
    Rates_elyte = np.zeros_like(SV_dot[SVptr['Ck elyte'][j]])
    # Rates_elyte = elyte.get_net_production_rates(elyte)
    #^^ need to multiply by volume fraction of elyte phase? (this is not yet in SV)
    N_k_out = np.zeros_like(N_k_in)
    N_k_out[2] = i_io[j-1]/ct.faraday
    grad_Flux_elyte = (N_k_in - N_k_out)/100e-6# * params['dyInv']
    dSVdt_ck_elyte = Rates_elyte_sei + Rates_elyte + grad_Flux_elyte

    # Infinite reservoir:
    res[SVptr['Ck elyte'][j]] = SV_dot[SVptr['Ck elyte'][j]] - dSVdt_ck_elyte

    # Calculate faradaic current density due to charge transfer at SEI-elyte
    #   interface, in A/m2 total.
    Rates_sei_conductor = sei_elyte.get_net_production_rates(sei_conductor)
    i_Far[j] = Rates_sei_conductor*sei_APV*ct.faraday/params['dyInv']

    dSVdt_eps_sei = np.dot(dSVdt_ck_sei, sei.partial_molar_volumes)
    res[SVptr['eps sei'][j]] = SV_dot[SVptr['eps sei'][j]] - dSVdt_eps_sei

    i_dl = i_Far - i_sei[:-1] + i_sei[1:]
    dSVdt_phi_dl = -i_dl/params['C_dl WE_sei']
    res[SVptr['phi sei']] = SV_dot[SVptr['phi sei']] - dSVdt_phi_dl
    #check signs--option 1:
    #res[SVptr['phi elyte']] = i_io_in - i_io_out + i_dl + i_Far
    #check signs--option 2:
    res[SVptr['phi elyte']] = i_io[:-1] - i_io[1:] + i_sei[:-1] - i_sei[1:]
    res[SVptr['phi elyte'][-1]] = SV[SVptr['phi elyte'][-1]]

    return res

def residual_homogeneous(t, SV, SV_dot):
    from sei_1d_init import objs, params, voltage_lookup, SVptr

    res = SV_dot - np.zeros_like(SV_dot)

    # Read out cantera objects:
    WE = objs['WE']
    sei = objs['SEI']
    elyte = objs['elyte']
    sei_elyte = objs['SEI_elyte']
    sei_conductor = objs['conductor']
    WE_sei = objs['WE_SEI']

    phi_WE = np.interp(t,voltage_lookup['time'],voltage_lookup['voltage'])

    # Set anode and anode/electrolyte interface potentials
    WE.electric_potential = phi_WE

    # SEI electric potential at anode interface:
    phi_sei_WE = phi_WE + SV[SVptr['phi sei-we']]

    # SEI Chemical composition:
    X_sei = SV[SVptr['Ck sei']] / sum(SV[SVptr['Ck sei']])

    # SEI electric potential at electrolyte interface:
    phi_sei_elyte = SV[SVptr['phi sei-elyte']]

    # SEI thickness:
    t_SEI = SV[SVptr['thickness']]

    sei.electric_potential = phi_sei_WE
    sei_conductor.electric_potential = phi_sei_WE
    sei.X = X_sei

    # The current into the sei at the WE interface equals the rate of production
    #   of electrons in the WE:
    i_far_WE = WE_sei.get_net_production_rates(WE)*ct.faraday

    # Calculate the current through the sei, which is Ohmic in nature:
    vol_k = sei.X * sei.partial_molar_volumes
    vol_tot = np.dot(sei.X, sei.partial_molar_volumes)
    vol_fracs = vol_k / vol_tot
    sigma_sei = np.dot(params['sigma sei'],vol_fracs)

    i_sei = sigma_sei*(phi_sei_WE - phi_sei_elyte)/t_SEI

    sei.electric_potential = phi_sei_elyte
    sei_conductor.electric_potential = phi_sei_elyte
    elyte.electric_potential = 0.

    # The current into the electolyte at the sei interface equals the rate of
    #   production of electrons in the sei conductor phase:
    i_far_elyte = sei_elyte.get_net_production_rates(sei_conductor)*ct.faraday

    # Molar production rate for sei species due to reactions at the sei-elyte
    #   interface:
    sdot_sei_elyte = sei_elyte.get_net_production_rates(sei)

    # Double layer current at the sei-WE interface:
    i_dl_WE = i_sei - i_far_WE

    # Double layer current at the sei-WE interface:
    i_dl_elyte = i_sei - i_far_elyte


    dSVdt_phi_sei_we = -i_dl_WE/params['C_dl WE_sei']
    res[SVptr['phi sei-we']] = SV_dot[SVptr['phi sei-we']] - dSVdt_phi_sei_we

    dSVdt_phi_sei_elyte = i_dl_elyte/params['C_dl WE_sei']
    res[SVptr['phi sei-elyte']] = SV_dot[SVptr['phi sei-elyte']] - dSVdt_phi_sei_elyte

    dSVdt_ck_sei = sdot_sei_elyte/t_SEI
    res[SVptr['Ck sei']] = SV_dot[SVptr['Ck sei']] - dSVdt_ck_sei

    dSVdt_t_sei = np.dot(sdot_sei_elyte,sei.partial_molar_volumes)
    res[SVptr['thickness']] = SV_dot[SVptr['thickness']] - dSVdt_t_sei

    return res


def residual_reduced(t, SV, SV_dot):
    from sei_1d_init import objs, params, voltage_lookup, SVptr

    res = SV_dot - np.zeros_like(SV_dot)

    # Read out cantera objects:
    sei = objs['SEI']
    elyte = objs['elyte']
    sei_elyte = objs['SEI_elyte']
    sei_conductor = objs['conductor']

    phi_WE = np.interp(t,voltage_lookup['time'],voltage_lookup['voltage'])

    # SEI Chemical composition:
    X_sei = SV[SVptr['Ck sei']] / sum(SV[SVptr['Ck sei']])

    # SEI thickness:
    t_SEI = SV[SVptr['thickness']]

    sei.electric_potential = phi_WE
    sei_conductor.electric_potential = phi_WE
    sei.X = X_sei

    # The model assumes that the electrolyte electric potential = 0 V:
    elyte.electric_potential = 0.

    # Molar production rate for sei species due to reactions at the sei-elyte
    #   interface.  These are scaled by the inverse sei thickness:
    sdot_sei_elyte = sei_elyte.get_net_production_rates(sei)/t_SEI

    # Time derivative of the change in the molar concentration (kmol/m3) of
    #   electrolyte species:
    dSVdt_ck_sei = sdot_sei_elyte/t_SEI
    res[SVptr['Ck sei']] = SV_dot[SVptr['Ck sei']] - dSVdt_ck_sei

    # Time derivative of the SEI thickness:
    dSVdt_t_sei = np.dot(sdot_sei_elyte,sei.partial_molar_volumes)
    res[SVptr['thickness']] = SV_dot[SVptr['thickness']] - dSVdt_t_sei

    return res
