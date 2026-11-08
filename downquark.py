import numpy as np

try:
    from zgrid import ZGrid
    from electron import HopfionElectron
    from parameters import *
    from visualize import plot_1d_slice
except ImportError:
    from .zgrid import ZGrid
    from .electron import HopfionElectron
    from .parameters import *
    from .visualize import plot_1d_slice

class DownQuark(HopfionElectron):
    def __init__(self, X, Y, Z, dx, xi=None, e_charge=(-1.0/3.0)):
            self.xi = xi if xi is not None else XI_E
            self.e_charge = e_charge
            self._X = X
            self._Y = Y
            self._Z = Z
        
    def get_torsion_density(self):
        tr_local = ((Y_VAC / Y_DOWN)**3) * (((Y_DOWN / 2.0) / ((Y_DOWN / 2.0) + XI_Z))**2)
        delta_tr = tr_local - 1.0
        r2 = self._X**2 + self._Y**2 + self._Z**2
        r_q_down = Y_DOWN / 2.0
        sigma_expansie = r_q_down * 0.8 
        torsion_envelope = np.exp(-r2 / (2.0 * sigma_expansie**2))
        raw_pressure = abs(delta_tr) * abs(self.e_charge) * torsion_envelope
        return raw_pressure

    def get_charge_density(self):
        tr_local = ((Y_VAC / Y_DOWN)**3) * (((Y_DOWN / 2.0) / ((Y_DOWN / 2.0) + XI_Z))**2)
        delta_tr = tr_local - 1.0
        r2 = self._X**2 + self._Y**2 + self._Z**2
        r_q_down = Y_DOWN / 2.0
        sigma_expansie = r_q_down * 0.8
        volume_envelope = np.exp(-r2 / (2.0 * sigma_expansie**2))
        rho_em_raw = abs(delta_tr) * volume_envelope
        return rho_em_raw

def run_downquark_model(verbose=True):
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    down = DownQuark(X=z_grid.X, Y=z_grid.Y, Z=z_grid.Z, dx=DX_ELECTRON)
    
    t_density = down.get_torsion_density()
    rho_em_raw = down.get_charge_density()
    
    r_center = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_center <= R_ELECTRON, 1.0, 0.0)

    Q_raw = np.sum(rho_em_raw * cutoff_envelope) * z_grid.dV
    rho_em_calibrated = (rho_em_raw * cutoff_envelope) * (down.e_charge / Q_raw) if Q_raw > 1e-10 else np.zeros_like(rho_em_raw)
    Q_total = np.sum(rho_em_calibrated) * z_grid.dV
    
    z_grid.relax(external_pressure=t_density, iterations=RELAX_ITERATIONS)
    
    e_grid = z_grid.get_energy_density()
    E_total = (np.abs(t_density) + e_grid) * cutoff_envelope
    M0_raw = np.sum(E_total) * z_grid.dV
    
    r_perp = np.sqrt(z_grid.X**2 + z_grid.Y**2)
    I_eff = np.sum(E_total * (r_perp**2)) * z_grid.dV
    R_rms_sq = 1.5 * (I_eff / M0_raw) if (M0_raw > 1e-10 and I_eff > 1e-10) else 1.0
    M0_phys = M0_raw / R_rms_sq
    
    omega = J_SPIN / I_eff if I_eff > 1e-10 else 0.0
    j_phi = SPIN_FACTOR * rho_em_calibrated * (omega * r_perp)
    mu_z = 0.5 * np.sum(r_perp * j_phi) * z_grid.dV
    g = (mu_z / J_SPIN) * (2.0 * M0_raw / down.e_charge) if abs(down.e_charge) > 1e-10 else 0.0

    if verbose:
        print(f"Geïntegreerde Lading (Q0):         {Q_total:.6f} [Model Units]")
        print(f"Geünificeerde Rustmassa (M0):      {M0_raw:.4f} [Model Units]")
        print(f"Effectief Traagheidsmoment (I_eff): {I_eff:.4f}")
        print(f"Magnetisch Moment (mu_z):          {mu_z:.6f}")
        print(f"COMPOSIET SPIN G-FACTOR (g):       {g:.6f}")

    return {
        "Q0": Q_total,
        "M0_raw": M0_raw,
        "I_eff": I_eff,
        "R_rms_sq": R_rms_sq,
        "M0_phys": M0_phys,
        "omega_0": omega,
        "mu_z": mu_z,
        "g": g,
        "z_grid": z_grid,
        "torsion_density": t_density
    }

if __name__ == "__main__":
    run_downquark_model()