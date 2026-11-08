import io
import re
import contextlib
import numpy as np

try:
    from zgrid import ZGrid
    from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
    from neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
    from electron import run_unified_model as run_electron_model
    from massfield import apply_external_tr_modulation
    from parameters import *
except ImportError:
    from .zgrid import ZGrid
    from .proton import run_proton_model, get_spherical_envelope as get_proton_envelope
    from .neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
    from .electron import run_unified_model as run_electron_model
    from .massfield import apply_external_tr_modulation
    from .parameters import *

def _extract_model_metrics(func, verbose=True):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = func()
    output = buf.getvalue()
    if verbose:
        print(output, end="")

    if isinstance(ret, dict):
        return ret

    def _find_float(pattern, default=0.0):
        m = re.search(pattern, output)
        return float(m.group(1)) if m else default

    q0 = _find_float(r"Lading\s*\(Q0\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    m0_raw = float(ret) if ret is not None else _find_float(r"Rustmassa\s*\(M0\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    i_eff = _find_float(r"Traagheidsmoment\s*\(I_eff\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    mu_z = _find_float(r"Magnetisch Moment\s*\(mu_z\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    g_fac = _find_float(r"G-FACTOR\s*\(g\):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

    if mu_z == 0.0 and re.search(r"Magnetisch Moment\s*\(mu_z\):\s*-0\.0+", output):
        mu_z = -1e-9

    r_rms_sq = 1.5 * (i_eff / m0_raw) if (m0_raw > 1e-10 and i_eff > 1e-10) else 1.0
    m0_phys = m0_raw / r_rms_sq

    return {
        "Q0": q0,
        "M0_raw": m0_raw,
        "I_eff": i_eff,
        "R_rms_sq": r_rms_sq,
        "M0_phys": m0_phys,
        "mu_z": mu_z,
        "g": g_fac,
    }

def run_deuterium_model(p_metrics=None, n_metrics=None, e_metrics=None, verbose=True):
    if p_metrics is None or n_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemassa's (Proton en Neutron)...")
            print("-" * 40)
        p_metrics = _extract_model_metrics(run_proton_model, verbose=verbose)
        if verbose:
            print("-" * 40)
        n_metrics = _extract_model_metrics(run_neutron_model, verbose=verbose)
        if verbose:
            print("-" * 40)

    if e_metrics is None:
        e_metrics = _extract_model_metrics(run_electron_model, verbose=False)

    M_p_raw = p_metrics["M0_raw"]
    M_n_raw = n_metrics["M0_raw"]
    R_rms_sq_p = p_metrics["R_rms_sq"]
    M_p_phys = p_metrics["M0_phys"]
    M_n_phys = n_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]

    if verbose:
        print("2. Z-Grid initialiseren voor Deuterium (Proton + Neutron)...")
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    
    r_q_up = Y_UP / 2.0
    r_q_down = Y_DOWN / 2.0
    r_bag_p = (2.0 * r_q_up + r_q_down) / 3.0
    r_bag_n = (1.0 * r_q_up + 2.0 * r_q_down) / 3.0
    d_nucleon = D_DEUTERIUM
    
    if verbose:
        print(f"3. Proton en Neutron injecteren op onderlinge afstand d = {d_nucleon}...")

    p_tr_up = ((Y_VAC / Y_UP)**3) * (((r_q_up) / ((r_q_up) + XI_Z))**2)
    p_tr_down = ((Y_VAC / Y_DOWN)**3) * (((r_q_down) / ((r_q_down) + XI_Z))**2)
    p_amp_up = abs(p_tr_up - 1.0) 
    p_amp_down = abs(p_tr_down - 1.0) 
    
    center_p = (-d_nucleon / 2.0, 0.0, 0.0)
    p_shell = get_proton_envelope(z_grid.X - center_p[0], z_grid.Y, z_grid.Z, r_bag_p)
    p_pressure = (2.0 * p_amp_up + 1.0 * p_amp_down) * p_shell
    
    n_tr_up = ((Y_VAC / Y_UP)**3) * (((r_q_up) / ((r_q_up) + XI_Z))**2)
    n_tr_down = ((Y_VAC / Y_DOWN)**3) * (((r_q_down) / ((r_q_down) + XI_Z))**2)
    n_amp_up = abs(n_tr_up - 1.0) 
    n_amp_down = abs(n_tr_down - 1.0) 
    
    center_n = (d_nucleon / 2.0, 0.0, 0.0)
    n_shell = get_neutron_envelope(z_grid.X - center_n[0], z_grid.Y, z_grid.Z, r_bag_n)
    n_pressure = (1.0 * n_amp_up + 2.0 * n_amp_down) * n_shell

    overlap_cancel = 2.0 * np.sqrt(p_pressure * n_pressure) * OVERLAP_CANCEL_FACTOR
    total_pressure = (p_pressure + n_pressure) - overlap_cancel
    
    r_center = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_center <= R_ELECTRON, 1.0, 0.0)

    p_vol = np.sum(p_shell * cutoff_envelope) * z_grid.dV
    n_vol = np.sum(n_shell * cutoff_envelope) * z_grid.dV
    
    rho_p = (p_shell * cutoff_envelope) * (1.0 / p_vol) if p_vol > 1e-10 else 0.0
    rho_n = (n_shell * cutoff_envelope) * (0.0 / n_vol) if n_vol > 1e-10 else 0.0
    
    rho_em_total = rho_p + rho_n
    Q_total = np.sum(rho_em_total) * z_grid.dV

    if verbose:
        print("4. Vacuüm-relaxatie uitvoeren...")
    z_grid.relax(external_pressure=total_pressure, iterations=RELAX_ITERATIONS)

    if verbose:
        print("5. Fysieke bolvorm en Tijd-Behouds Heuvelrug (massfield.py) toepassen...")
    e_grid = z_grid.get_energy_density()
    E_veld = np.abs(total_pressure) + e_grid
    
    r_center_p = np.sqrt((z_grid.X - center_p[0])**2 + z_grid.Y**2 + z_grid.Z**2)
    r_center_n = np.sqrt((z_grid.X - center_n[0])**2 + z_grid.Y**2 + z_grid.Z**2)
    E_bag_p = np.where(r_center_p <= r_bag_p, BAG_ENERGY_DENSITY, 0.0)
    E_bag_n = np.where(r_center_n <= r_bag_n, BAG_ENERGY_DENSITY, 0.0)
    E_bag_tot = E_bag_p + E_bag_n
    
    E_tot_raw = (E_veld + E_bag_tot) * cutoff_envelope

    E_total_density, _ = apply_external_tr_modulation(
        z_grid.X, z_grid.Y, z_grid.Z,
        E_tot_raw,
        center_p=center_p,
        center_n=center_n,
        r_bag_p=r_bag_p,
        r_bag_n=r_bag_n,
        M_p_raw=M_p_raw,
        M_n_raw=M_n_raw,
        omega_scale=OMEGA_MACRO_SCALE
    )
    M_d_raw = np.sum(E_total_density) * z_grid.dV

    delta_M_raw = (M_p_raw + M_n_raw) - M_d_raw
    delta_M_phys = delta_M_raw / R_rms_sq_p
    M_d_phys = (M_p_phys + M_n_phys) - delta_M_phys

    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0
    delta_E_MeV = delta_M_phys * mass_conv_MeV

    if verbose:
        print("\n" + "=" * 65)
        print("GEÜNIFICEERDE KERNFYSICA ANALYSE: BINDINGSENERGIE DEUTERIUM")
        print("=" * 65)
        print(f"Geïntegreerde Totale Lading (Q0):  {Q_total:.4f} (Verwacht: +1.0)")
        print(f"Ruwe Model Massa Proton (Mp_raw):  {M_p_raw:.4f}")
        print(f"Ruwe Model Massa Neutron (Mn_raw): {M_n_raw:.4f}")
        print(f"Fysieke Model Massa Proton (Mp):   {M_p_phys:.4f}")
        print(f"Fysieke Model Massa Neutron (Mn):  {M_n_phys:.4f}")
        print(f"Fysieke Massa Deuteriumkern (Md):  {M_d_phys:.4f}")
        print("-" * 65)
        print(f"Massadefect (ΔM_phys) [ModelUnit]: {delta_M_phys:.4f}")
        print("-" * 65)
        if delta_M_phys > 0:
            print(f"BEREKENDE BINDINGSENERGIE (ΔE):    {delta_E_MeV:.4f} MeV")
        else:
            print(f"AFSTOTENDE ENERGIE (INSTABIEL):    {delta_E_MeV:.4f} MeV")
        print("=" * 65)

    return {
        "Q0": Q_total,
        "M_d_raw": M_d_raw,
        "M_d_phys": M_d_phys,
        "delta_M_raw": delta_M_raw,
        "delta_M_phys": delta_M_phys,
        "E_bind_MeV": delta_E_MeV,
    }

if __name__ == "__main__":
    run_deuterium_model()