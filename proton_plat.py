import numpy as np

try:
    from zgrid import ZGrid
    from upquark import UpQuark
    from downquark import DownQuark
    from parameters import *
    from visualize import plot_1d_slice
except ImportError:
    from .zgrid import ZGrid
    from .upquark import UpQuark
    from .downquark import DownQuark
    from .parameters import *
    from .visualize import plot_1d_slice

def run_proton_model():
    print("1. Z-Grid initialiseren voor de 3-Quark composiet (Proton)...")
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    
    r_q = 50.0
    
    # 2. Definiëren van de drie Quarks (U-U-D)
    u1_X = z_grid.X
    u1_Y = z_grid.Y - r_q
    
    u2_X = z_grid.X - r_q * np.cos(np.pi/6)
    u2_Y = z_grid.Y + r_q * np.sin(np.pi/6)
    
    d_X = z_grid.X + r_q * np.cos(np.pi/6)
    d_Y = z_grid.Y + r_q * np.sin(np.pi/6)
    
    print("2. Injecteren van de Modulaire Topologische Knooppunten...")
    
    up1 = UpQuark(X=u1_X, Y=u1_Y, Z=z_grid.Z, dx=DX_ELECTRON)
    up2 = UpQuark(X=u2_X, Y=u2_Y, Z=z_grid.Z, dx=DX_ELECTRON)
    down = DownQuark(X=d_X, Y=d_Y, Z=z_grid.Z, dx=DX_ELECTRON)
    
    # De klassen regelen nu zelf het teken en de Cosserat-saturatie!
    total_pressure = up1.get_torsion_density() + up2.get_torsion_density() + down.get_torsion_density()
    
    rho_em_total = up1.get_charge_density() + up2.get_charge_density() + down.get_charge_density()
    Q_total = np.sum(rho_em_total) * z_grid.dV

    print("4. Vacuüm autonoom laten relaxeren rondom het Proton-defect...")
    z_grid.relax(external_pressure=total_pressure, iterations=RELAX_ITERATIONS)

    print("5. Fysieke spin-eigenschappen en composiet-massa berekenen...")
    e_grid = z_grid.get_energy_density()
    E_total_density = np.abs(total_pressure) + e_grid
    M0 = np.sum(E_total_density) * z_grid.dV

    r_perp = np.sqrt(z_grid.X**2 + z_grid.Y**2)
    I_eff = np.sum(E_total_density * (r_perp**2)) * z_grid.dV
    omega_0 = J_SPIN / I_eff

    j_isovector = SPIN_FACTOR * rho_em_total * (omega_0 * r_perp)
    j_TR_halo = VACUUM_POL_FACTOR * (1.0 - z_grid.TR) * (omega_0 * r_perp)
    j_phi = j_isovector + j_TR_halo

    mu_z = 0.5 * np.sum(r_perp * j_phi) * z_grid.dV
    g_factor = (mu_z / J_SPIN) * (2.0 * M0 / Q_total)

    print("\n" + "=" * 60)
    print("GEÜNIFICEERDE 3D TENSOR ANALYSE: PROTON (U-U-D COMPOSIET)")
    print("=" * 60)
    print(f"Geïntegreerde Totale Lading (Q0):   {Q_total:.6f} (Verwacht: +1.0)")
    print(f"Totale Rustmassa (M0):             {M0:.4f}")
    print(f"Effectief Traagheidsmoment (I_eff): {I_eff:.4f}")
    print(f"Hoeksnelheid (omega_0):            {omega_0:.8e}")
    print(f"Magnetisch Moment (mu_z):          {mu_z:.6f}")
    print("-" * 60)
    print(f"COMPOSIET SPIN G-FACTOR (g):        {g_factor:.6f}")
    print("=" * 60)

    plot_1d_slice(z_grid, total_pressure, title="Proton (U-U-D) - Cosserat Saturation")
    
    return M0

if __name__ == "__main__":
    run_proton_model()