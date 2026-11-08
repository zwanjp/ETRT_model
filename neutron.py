import numpy as np

try:
    from zgrid import ZGrid
    from parameters import *
except ImportError:
    from .zgrid import ZGrid
    from .parameters import *

def get_spherical_envelope(X, Y, Z, radius):
    r2 = X**2 + Y**2 + Z**2
    sigma = radius * 0.8  
    return np.exp(-r2 / (2.0 * sigma**2))

def run_neutron_model():
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    
    r_q_up = Y_UP / 2.0
    r_q_down = Y_DOWN / 2.0
    r_bag = (1.0 * r_q_up + 2.0 * r_q_down) / 3.0 # Gezamenlijke Confinement Straal
    
    # 1. Gezamenlijke ruimtelijke wolk voor de quarks (Bag confinement)
    shell_bag = get_spherical_envelope(z_grid.X, z_grid.Y, z_grid.Z, r_bag)
    
    # 2. Topologische Druk (MASSA-opbouw afhankelijk van microscopische Y-waarden)
    tr_up = ((Y_VAC / Y_UP)**3) * (((r_q_up) / ((r_q_up) + XI_Z))**2)
    tr_down = ((Y_VAC / Y_DOWN)**3) * (((r_q_down) / ((r_q_down) + XI_Z))**2)
    
    amp_up = abs(tr_up - 1.0) 
    amp_down = abs(tr_down - 1.0) 
    
    # Neutron = UDD (Absoluut opgeteld binnen de bag)
    total_pressure = (1.0 * amp_up + 2.0 * amp_down) * shell_bag
    
    # 3. Lading Opbouw (Halfway Bag: Balans tussen eigen aard en opsluiting)
    r_center = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_center <= R_ELECTRON, 1.0, 0.0)

    # De effectieve ladings-straal is halverwege de vrije straal en de bag-straal
    r_charge_up = (r_q_up + r_bag) / 2.0
    r_charge_down = (r_q_down + r_bag) / 2.0
    
    shell_charge_up = get_spherical_envelope(z_grid.X, z_grid.Y, z_grid.Z, r_charge_up)
    shell_charge_down = get_spherical_envelope(z_grid.X, z_grid.Y, z_grid.Z, r_charge_down)

    vol_up = np.sum(shell_charge_up * cutoff_envelope) * z_grid.dV
    vol_down = np.sum(shell_charge_down * cutoff_envelope) * z_grid.dV
    
    rho_up = (shell_charge_up * cutoff_envelope) * ((2.0/3.0) / vol_up) if vol_up > 1e-10 else 0.0
    rho_down = (shell_charge_down * cutoff_envelope) * ((-1.0/3.0) / vol_down) if vol_down > 1e-10 else 0.0

    # Neutron Lading = 1x Up + 2x Down
    rho_em_total = (1.0 * rho_up) + (2.0 * rho_down)
    Q_total = np.sum(rho_em_total) * z_grid.dV

    z_grid.relax(external_pressure=total_pressure, iterations=RELAX_ITERATIONS)
    
    # 4. Fysieke Massa & Spin
    e_grid = z_grid.get_energy_density()
    E_veld = np.abs(total_pressure) + e_grid
    
    E_bag = np.where(r_center <= r_bag, 0.5, 0.0)
    
    E_total_density = (E_veld + E_bag) * cutoff_envelope
    M0_model = np.sum(E_total_density) * z_grid.dV
    
    r_perp = np.sqrt(z_grid.X**2 + z_grid.Y**2)
    I_eff = np.sum(E_total_density * (r_perp**2)) * z_grid.dV

    omega_0 = J_SPIN / I_eff if I_eff > 1e-10 else 0.0
    
    j_phi = SPIN_FACTOR * rho_em_total * (omega_0 * r_perp)
    mu_z = 0.5 * np.sum(r_perp * j_phi) * z_grid.dV

    print("\n" + "=" * 60)
    print("GEÜNIFICEERDE 3D TENSOR ANALYSE: NEUTRON")
    print("=" * 60)
    print(f"Geïntegreerde Lading (Q0):         {Q_total:.6f} [Model Units] (Verwacht: 0.0)")
    print(f"Geünificeerde Rustmassa (M0):      {M0_model:.4f} [Model Units]")
    print(f"Effectief Traagheidsmoment (I_eff): {I_eff:.4f}")
    print(f"Magnetisch Moment (mu_z):          {mu_z:.6f} (Verwacht: Negatief!)")
    print("=" * 60)
    return M0_model

if __name__ == "__main__":
    run_neutron_model()
