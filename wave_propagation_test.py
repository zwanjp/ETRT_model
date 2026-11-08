import numpy as np

try:
    from zgrid import ZGrid
    from parameters import *
except ImportError:
    from .zgrid import ZGrid
    from .parameters import *

def get_propagation_speed(verbose=False):
    z_grid = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_PHOTON, k_tr=K_TR, rho_z=RHO_Z)
    c_theorie = np.sqrt(z_grid.k_tr / z_grid.rho_z)
    
    stappen = 120
    dt = DT_PHOTON
    centrum = GRID_SHAPE_PHOTON[0] // 2
    
    # Initiële 3D Gaussische verstoring rond het rust-vacuüm
    r2_3d = z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2
    amplitude = 0.1
    sigma = 0.2
    TR_nu = amplitude * np.exp(-r2_3d / (2.0 * sigma**2))
    TR_oud = np.copy(TR_nu)
    
    # Meet de positie van het 3D golffront op twee tijdstippen (t1=50 en t2=100)
    # zodat de constante 3D Huygens fase-verschuiving (+sigma) exact wegvalt.
    t1, t2 = 50, 100
    r_t1, r_t2 = 0.0, 0.0
    
    for t in range(1, stappen + 1):
        TR_nieuw = z_grid.propagate_wave(TR_oud, TR_nu, dt=dt)
        TR_oud = np.copy(TR_nu)
        TR_nu = np.copy(TR_nieuw)
        
        if t in (t1, t2):
            slice_2d = TR_nu[:, :, centrum]
            midden_doorsnede = slice_2d[centrum, centrum:]
            piek_index = np.argmax(midden_doorsnede[5:]) + 5
            r_piek = piek_index * DX_PHOTON
            if t == t1:
                r_t1 = r_piek
            elif t == t2:
                r_t2 = r_piek

    gemeten_v = (r_t2 - r_t1) / ((t2 - t1) * dt)
    
    if verbose:
        print("=" * 65)
        print("ETRT 3D GOLFPROPAGATIE TEST (FOTON IN Z-GRID VACUÜM)")
        print("=" * 65)
        print(f"Theoretische roostersnelheid (c_grid): {c_theorie:.4f} [ModelUnit/t]")
        print(f"Piekpositie op t1 = {t1:3d} ({t1*dt:.2f} t0):     r1 = {r_t1:.4f} (ct1 + sigma)")
        print(f"Piekpositie op t2 = {t2:3d} ({t2*dt:.2f} t0):     r2 = {r_t2:.4f} (ct2 + sigma)")
        print(f"Gemeten voortplantingssnelheid (v):    {gemeten_v:.4f} [ModelUnit/t]")
        print(f"Fysieke lichtsnelheid (v * c):         {gemeten_v * C:.4e} m/s")
        print("=" * 65)
        
    return float(gemeten_v)

if __name__ == "__main__":
    v = get_propagation_speed(verbose=True)
    print(f"Propagation Speed: {v:.4f}")