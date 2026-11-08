import io
import re
import contextlib
import numpy as np

from zgrid import ZGrid
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from massfield import get_macroscopic_tr_field, apply_tr_to_energy
from parameters import *
from visualize import plot_1d_slice


def run_atom_h_model(p_metrics=None, e_metrics=None, verbose=True):
    """
    ETRT 3D Model voor het Waterstofatoom (1H: Proton + 1s Elektron-schil).
    Combineert proton.py, electron.py en massfield.py op het atomaire Z-Grid
    (geschaald met de elektromagnetische fijnstructuur-factor 1 / ALPHA^2).
    """
    if p_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton en Elektron)...")
            print("-" * 55)
        p_metrics = _extract_model_metrics(run_proton_model, verbose=verbose)
        if verbose:
            print("-" * 55)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=verbose)
        if verbose:
            print("-" * 55)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    R_rms_p_model = np.sqrt(p_metrics["R_rms_sq"])

    M_e_raw = e_metrics["M0_raw"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    atomic_scale_factor = 1.0 / (ALPHA**2)
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)

    r_bohr_grid = R_rms_e_model / red_mass_factor
    R_bohr_model = r_bohr_grid * atomic_scale_factor

    r_e_m = R_E_FM * 1e-15
    m_per_unit = r_e_m / R_rms_e_model if R_rms_e_model > 1e-12 else 0.0
    a_0_phys_m = R_bohr_model * m_per_unit

    if verbose:
        print("2. Z-Grid initialiseren op atomaire schaal (1 / ALPHA^2)...")
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_grid = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    if verbose:
        print(f"3. Proton-kern in (0,0,0) en 1s Elektron-wolk (r_Bohr = {r_bohr_grid:.2f}) injecteren...")

    r_core_p = DX_ELECTRON
    proton_core_shell = get_proton_envelope(z_grid.X, z_grid.Y, z_grid.Z, r_core_p)
    electron_1s_shell = np.exp(-2.0 * r_grid / r_bohr_grid)

    vol_p = np.sum(proton_core_shell * cutoff_envelope) * z_grid.dV
    vol_e = np.sum(electron_1s_shell * cutoff_envelope) * z_grid.dV
    rho_p = (proton_core_shell * cutoff_envelope) * (1.0 / vol_p) if vol_p > 1e-10 else 0.0
    rho_e = (electron_1s_shell * cutoff_envelope) * (-1.0 / vol_e) if vol_e > 1e-10 else 0.0
    rho_em_total = rho_p + rho_e
    Q_total = np.sum(rho_em_total) * z_grid.dV

    p_tr_up = ((Y_VAC / Y_UP)**3) * (((Y_UP / 2.0) / ((Y_UP / 2.0) + XI_Z))**2)
    p_tr_down = ((Y_VAC / Y_DOWN)**3) * (((Y_DOWN / 2.0) / ((Y_DOWN / 2.0) + XI_Z))**2)
    p_amp = 2.0 * abs(p_tr_up - 1.0) + 1.0 * abs(p_tr_down - 1.0)

    p_pressure_atom = p_amp * proton_core_shell
    e_pressure_atom = -p_amp * (vol_p / vol_e) * electron_1s_shell
    total_atomic_pressure = p_pressure_atom + e_pressure_atom

    if verbose:
        print("4. Atomaire vacuüm-relaxatie en massfield.py TR-modulatie uitvoeren...")
    z_grid.relax(external_pressure=total_atomic_pressure, iterations=RELAX_ITERATIONS)

    particles = [{"center": (0.0, 0.0, 0.0), "omega": M_p_raw * OMEGA_MACRO_SCALE, "r0": XI_Z}]
    TR_field, _ = get_macroscopic_tr_field(z_grid.X, z_grid.Y, z_grid.Z, particles, r0=XI_Z)

    prob_1s = (electron_1s_shell * cutoff_envelope) / vol_e
    tr_overlap_integral = np.sum(prob_1s * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * z_grid.dV

    delta_M_H_phys = 0.5 * (ALPHA**2) * red_mass_factor * M_e_phys * tr_overlap_integral
    M_H_phys = (M_p_phys + M_e_phys) - delta_M_H_phys

    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0
    M_H_MeV = M_H_phys * mass_conv_MeV
    E_bind_MeV = delta_M_H_phys * mass_conv_MeV
    E_bind_eV = E_bind_MeV * 1e6

    # CORRECTE BEREKENING MAGNETISCH MOMENT WATERSTOF (1s grondtoestand, elektron-gedomineerd & negatief)
    mu_z_H = - (abs(e_metrics["mu_z"]) * (r_bohr_grid / R_rms_e_model) + abs(p_metrics["mu_z"]))
    I_eff_H = e_metrics["I_eff"] + p_metrics["I_eff"]
    omega_0_H = J_SPIN / I_eff_H if I_eff_H > 1e-10 else 0.0

    if verbose:
        print("\n" + "=" * 70)
        print("GEÜNIFICEERDE ATOOMFYSICA ANALYSE: WATERSTOFATOOM (1H - atom_h.py)")
        print("=" * 70)
        print(f"Geïntegreerde Totale Lading (Q0):     {Q_total:.4f} (Verwacht: 0.0 Neutraal)")
        print(f"Berekende Bohr-straal (a0_model):     {R_bohr_model:.4e} [Model Units]")
        print(f"Fysieke Bohr-straal (a0):             {a_0_phys_m:.4e} m (Empirisch: 5.292e-11 m)")
        print(f"Fysieke Model Massa Proton (Mp):      {M_p_phys:.6f}")
        print(f"Fysieke Model Massa Elektron (Me):    {M_e_phys:.6f}")
        print(f"Fysieke Massa Waterstofatoom (MH):    {M_H_phys:.6f} ({M_H_MeV:.3f} MeV/c^2)")
        print("-" * 70)
        print(f"3D Z-Grid Overlap-integraal <a0/r>:   {tr_overlap_integral:.4f}")
        print(f"Atomair Massadefect (ΔM_H) [Model]:   {delta_M_H_phys:.4e}")
        print("-" * 70)
        if delta_M_H_phys > 0:
            print(f"BEREKENDE IONISATIE-ENERGIE (ΔE_1s):  {E_bind_eV:.4f} eV (Empirisch: 13.60 eV)")
        else:
            print(f"AFSTOTENDE ENERGIE (INSTABIEL):       {E_bind_eV:.4f} eV")
        print("=" * 70)

    return {
        "Q0": Q_total,
        "R_bohr_model": R_bohr_model,
        "a_0_phys_m": a_0_phys_m,
        "M_H_phys": M_H_phys,
        "M_H_MeV": M_H_MeV,
        "delta_M_phys": delta_M_H_phys,
        "E_bind_MeV": E_bind_MeV,
        "E_bind_eV": E_bind_eV,
        "I_eff": I_eff_H,
        "omega_0": omega_0_H,
        "mu_z": mu_z_H,
        "z_grid": z_grid,
        "torsion_density": total_atomic_pressure,
    }


if __name__ == "__main__":
    run_atom_h_model()