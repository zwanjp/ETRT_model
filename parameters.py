import numpy as np
"""
ETRT-MODEL PARAMETERS EN CONSTANTEN
"""
# =============================================================================
# 1. Fysische parameters
# =============================================================================
XI_E = 0.323      
XI_Z = 0.5  
Y_VAC = 10.0
Y_ASSYM = 0.001

# =============================================================================
# 2. Modelconstanten & Deuterium / Macro-koppeling
# =============================================================================
# Y_UP en Y_DOWN worden hier dynamisch afgeleid uit Y_VAC en Y_ASSYM:
Y_UP = Y_VAC - Y_ASSYM
Y_DOWN = Y_VAC + Y_ASSYM

DUMMY_XI_E = XI_E
LAMBDA_ELEKTRON = XI_Z  
BETA_COUPLE = 1.0 

J_SPIN = np.sqrt(3.0)/2.0 
SPIN_FACTOR = 2.0    
      
GRID_SHAPE_ELECTRON = (41, 41, 41) 
DX_ELECTRON = 3.5                  
RELAX_ITERATIONS = 30     

GRID_SHAPE_PHOTON = (101, 101, 101) 
DX_PHOTON = 0.1                     
DT_PHOTON = 0.04

K_TR = 1.0                
RHO_Z = 1.0 

R_ELECTRON = 1000

# Kernfysica & Massfield (Tijd-Behouds Heuvelrug) parameters
BAG_ENERGY_DENSITY = 0.5
D_DEUTERIUM = 14.0              # Rooster-uitgelijnde bindingsafstand (4 * DX_ELECTRON ~ e * r_bag)
OVERLAP_CANCEL_FACTOR = 0.0     # Geen kunstmatige fase-afvlakking nodig (100% via massfield.py)
OMEGA_MACRO_SCALE = 0.00012288  # Zuivere TR-heuvelrug kalibratie voor E_bind = 2.2246 MeV

# =============================================================================
# 3. Natuurconstanten & Empirische Referenties
# =============================================================================
M_E = 9.1093837e-31       
E_CHARGE = 1.60217663e-19 
C = 299792458.0           
HBAR = 1.054571817e-34    
ALPHA = 1.0 / 137.035999  
R_E_FM = 2.8179403227  

M_E_MEV = (M_E * C**2) / (E_CHARGE * 1e6)
G_ELECTRON_EMP = 2.002319
M_PROTON_EMP_MEV = 938.272
G_PROTON_EMP = 2.792847
M_NEUTRON_EMP_MEV = 939.565
E_BIND_DEUTERIUM_EMP_MEV = 2.22457

# Empirische referenties Waterstofatoom (1H)
A0_BOHR_EMP_M = 5.291772e-11
M_ATOM_H_EMP_MEV = 938.783
E_BIND_ATOM_H_EMP_EV = 13.5984