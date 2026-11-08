import numpy as np

try:
    from zgrid import ZGrid
    from parameters import *
    from visualize import plot_1d_slice
except ImportError:
    from .zgrid import ZGrid
    from .parameters import *
    from .visualize import plot_1d_slice

class HopfionElectron:
    """
    Het topologische defect (Spin-1/2 Hopfion).
    Gedefinieerd door de characteristic transition length (xi).
    """
    def __init__(self, X, Y, Z, dx, xi=0.5, k_torsion=1.0, e_charge=1.0):
        self.dx = dx
        self.xi = xi
        self.k_torsion = k_torsion
        self.e_charge = e_charge
        
        # 1. Director veld via zuivere Spinor-projectie (afhankelijk van xi)
        r_sq = X**2 + Y**2 + Z**2
        denom = (r_sq + self.xi**2)**2

        Z0_re = r_sq - self.xi**2
        Z0_im = 2.0 * self.xi * Z
        Z1_re = 2.0 * self.xi * X
        Z1_im = 2.0 * self.xi * Y

        self.nx = 2.0 * (Z0_re * Z1_re + Z0_im * Z1_im) / denom
        self.ny = 2.0 * (Z0_re * Z1_im - Z0_im * Z1_re) / denom
        self.nz = (Z0_re**2 + Z0_im**2 - Z1_re**2 - Z1_im**2) / denom

        # 2. Ruimtelijke Gradiënten (De twist van de cellen)
        self.dnx_dx, self.dnx_dy, self.dnx_dz = np.gradient(self.nx, dx, dx, dx)
        self.dny_dx, self.dny_dy, self.dny_dz = np.gradient(self.ny, dx, dx, dx)
        self.dnz_dx, self.dnz_dy, self.dnz_dz = np.gradient(self.nz, dx, dx, dx)

    def get_torsion_density(self):
        """Berekent de lokale topologische torsiedruk."""
        grad_sq = (self.dnx_dx**2 + self.dnx_dy**2 + self.dnx_dz**2 +
                   self.dny_dx**2 + self.dny_dy**2 + self.dny_dz**2 +
                   self.dnz_dx**2 + self.dnz_dy**2 + self.dnz_dz**2)
        return 0.5 * self.k_torsion * grad_sq

    def get_charge_density(self):
        """Berekent de exacte 3D Pontryagin-tensordichtheid."""
        Bx = (self.nx * (self.dny_dy * self.dnz_dz - self.dnz_dy * self.dny_dz) +
              self.ny * (self.dnz_dy * self.dnx_dz - self.dnx_dy * self.dnz_dz) +
              self.nz * (self.dnx_dy * self.dny_dz - self.dny_dy * self.dnx_dz)) / (4.0 * np.pi)

        By = (self.nx * (self.dny_dz * self.dnz_dx - self.dnz_dz * self.dny_dx) +
              self.ny * (self.dnz_dz * self.dnx_dx - self.dnx_dz * self.dnz_dx) +
              self.nz * (self.dnx_dz * self.dny_dx - self.dny_dz * self.dnx_dx)) / (4.0 * np.pi)

        Bz = (self.nx * (self.dny_dx * self.dnz_dy - self.dnz_dx * self.dny_dy) +
              self.ny * (self.dnz_dx * self.dnx_dy - self.dnx_dx * self.dnz_dy) +
              self.nz * (self.dnx_dx * self.dny_dy - self.dny_dx * self.dnx_dy)) / (4.0 * np.pi)

        B_top_norm = np.sqrt(Bx**2 + By**2 + Bz**2)
        
        # Genormaliseerde lading
        dV = self.dx**3
        Q_raw = np.sum(B_top_norm) * dV
        
        if Q_raw > 1e-10:
            return self.e_charge * (B_top_norm / Q_raw)
        return np.zeros_like(B_top_norm)

def run_unified_model():
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    
    print(f"1. Injecteren van het Hopfion defect met overgangslengte xi = {DUMMY_XI_E}...")
    electron = HopfionElectron(X=z_grid.X, Y=z_grid.Y, Z=z_grid.Z, dx=DX_ELECTRON, xi=DUMMY_XI_E)

    print("2. Bepalen van torsiedruk en autonome vacuümrelaxatie...")
    torsion_density = electron.get_torsion_density()
    z_grid.relax(external_pressure=torsion_density, iterations=RELAX_ITERATIONS)

    print(f"3. Spin-eigenschappen, lading en MASSA integreren (Vacuümgrens R = {R_ELECTRON})...")
    
    # 1. Harde vacuümgrens op basis van R_ELECTRON
    r_center = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_center <= R_ELECTRON, 1.0, 0.0)

    # 2. Lading en Energiedichtheid over exact hetzelfde begrensde volume
    rho_em = electron.get_charge_density() * cutoff_envelope
    Q_total = np.sum(rho_em) * z_grid.dV

    e_grid = z_grid.get_energy_density()
    E_total_density = (torsion_density + e_grid) * cutoff_envelope
    
    # 3. Massa en Traagheidsmoment
    M0_model = np.sum(E_total_density) * z_grid.dV
    r_perp = np.sqrt(z_grid.X**2 + z_grid.Y**2)
    I_eff_model = np.sum(E_total_density * (r_perp**2)) * z_grid.dV
    
    # 4. Hoeksnelheid en Zuivere Ringstroom
    omega_0 = J_SPIN / I_eff_model if I_eff_model > 1e-10 else 0.0
    j_phi = SPIN_FACTOR * rho_em * (omega_0 * r_perp)

    # 5. Magnetisch Moment & G-Factor
    mu_z_model = 0.5 * np.sum(r_perp * j_phi) * z_grid.dV
    g_factor = (mu_z_model / J_SPIN) * (2.0 * M0_model / Q_total) if Q_total > 1e-10 else 0.0

    # Conversie naar SI
    Q_physical_Coulomb = Q_total * E_CHARGE
    kg_per_model_unit = M_E / M0_model if M0_model > 1e-10 else 0.0
    M0_physical_kg = M0_model * kg_per_model_unit
    M0_physical_MeV = (M0_physical_kg * C**2) / (E_CHARGE * 1e6)

    print("\n" + "=" * 65)
    print("GEÜNIFICEERDE 3D TENSOR ANALYSE: ELEKTRON IN Z-GRID")
    print("=" * 65)
    print("--- MODEL EENHEDEN ---")
    print(f"Geïntegreerde Lading (Q0):         {Q_total:.6f} [Model Units]")
    print(f"Oppervlakte Rustmassa (M0):        {M0_model:.4f} [Model Units]")
    print(f"Effectief Traagheidsmoment (I_eff): {I_eff_model:.4f}")
    print(f"Hoeksnelheid (omega_0):            {omega_0:.8e}")
    print(f"Magnetisch Moment (mu_z):          {mu_z_model:.6f}")
    print("\n--- FYSISCHE REALITY CHECK (CONVERSIE) ---")
    print(f"Lading in Coulomb:                 {Q_physical_Coulomb:.6e} C")
    print(f"Massa (gekalibreerd) in kg:        {M0_physical_kg:.6e} kg")
    print(f"Massa (gekalibreerd) in MeV/c^2:   {M0_physical_MeV:.6f} MeV/c^2")
    print("-" * 65)
    print(f"DEFINITIEVE SPIN G-FACTOR (g):      {g_factor:.6f}")
    print("=" * 65)
    
    return M0_model

if __name__ == "__main__":
    run_unified_model()