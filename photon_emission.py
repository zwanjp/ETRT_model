import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from massfield import get_macroscopic_tr_field
from parameters import *


def run_photon_emission_model(p_metrics=None, e_metrics=None, verbose=True, save_plot=True):
    """
    ETRT 3D Simulatie van Foton-Emissie uit het Waterstofatoom (1H: 2p_z -> 1s Lyman-alfa sprong).
    Koppelt Motor 1 (statische vacuüm-relaxatie van de 2p_z en 1s toestanden) rechtstreeks
    aan Motor 2 (dynamische 3D d'Alembert golfpropagatie van het losgelaten verschilveld delta_TR).
    """
    if p_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton en Elektron)...")
            print("-" * 65)
        p_metrics = _extract_model_metrics(run_proton_model, verbose=verbose)
        if verbose:
            print("-" * 65)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=verbose)
        if verbose:
            print("-" * 65)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    # 2. Atomaire schaalsprong (1 / ALPHA^2) en Bohr-straal op het rooster
    atomic_scale_factor = 1.0 / (ALPHA**2)
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    # We gebruiken GRID_SHAPE_PHOTON (101x101x101) met DX_ELECTRON (3.5) zodat zowel
    # de uitgestrekte 2p_z-wolk (n=2, straal ~4*r_bohr) als het wegreizende foton volledig passen
    if verbose:
        print("2. 3D Z-Grid initialiseren voor aangeslagen toestand (2p_z) en grondtoestand (1s)...")
    zg_1s = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    zg_2p = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)

    r_grid = np.sqrt(zg_1s.X**2 + zg_1s.Y**2 + zg_1s.Z**2)
    cutoff_envelope = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # --- PROTON KERN IN OORSPRONG (0,0,0) ---
    r_core_p = DX_ELECTRON
    proton_core_shell = get_proton_envelope(zg_1s.X, zg_1s.Y, zg_1s.Z, r_core_p)

    # --- ELEKTRON GOLFSCHILLEN: 1S GRONDTOESTAND EN 2P_Z AANGESLAGEN TOESTAND ---
    # 1s:   |psi_1s(r)|^2   ~ exp(-2 * r / r_bohr_grid)
    # 2p_z: |psi_2pz(r)|^2  ~ (z / r_bohr_grid)^2 * exp(-r / r_bohr_grid)
    electron_1s_shell = np.exp(-2.0 * r_grid / r_bohr_grid)
    electron_2p_shell = ((zg_1s.Z / r_bohr_grid)**2) * np.exp(-r_grid / r_bohr_grid)

    vol_p = np.sum(proton_core_shell * cutoff_envelope) * zg_1s.dV
    vol_1s = np.sum(electron_1s_shell * cutoff_envelope) * zg_1s.dV
    vol_2p = np.sum(electron_2p_shell * cutoff_envelope) * zg_1s.dV

    # --- TOPOLOGISCHE DRUK VOOR 1S EN 2P_Z ---
    p_tr_up = ((Y_VAC / Y_UP)**3) * (((Y_UP / 2.0) / ((Y_UP / 2.0) + XI_Z))**2)
    p_tr_down = ((Y_VAC / Y_DOWN)**3) * (((Y_DOWN / 2.0) / ((Y_DOWN / 2.0) + XI_Z))**2)
    p_amp = 2.0 * abs(p_tr_up - 1.0) + 1.0 * abs(p_tr_down - 1.0)

    pressure_1s = p_amp * proton_core_shell - p_amp * (vol_p / vol_1s) * electron_1s_shell
    pressure_2p = p_amp * proton_core_shell - p_amp * (vol_p / vol_2p) * electron_2p_shell

    # --- MOTOR 1: STATISCHE VACUÜM-RELAXATIE VOOR BEIDE TOESTANDEN ---
    if verbose:
        print("3. Motor 1 (relax): Evacuatie van vacuümspanning voor 2p_z en 1s berekenen...")
    zg_1s.relax(external_pressure=pressure_1s, iterations=RELAX_ITERATIONS)
    zg_2p.relax(external_pressure=pressure_2p, iterations=RELAX_ITERATIONS)

    # --- ENERGIEËN EN MASSADEFECT VAN DE KWANTUMSPRONG (2P_Z -> 1S) ---
    particles = [{"center": (0.0, 0.0, 0.0), "omega": M_p_raw * OMEGA_MACRO_SCALE, "r0": XI_Z}]
    TR_field, _ = get_macroscopic_tr_field(zg_1s.X, zg_1s.Y, zg_1s.Z, particles, r0=XI_Z)

    prob_1s = (electron_1s_shell * cutoff_envelope) / vol_1s
    prob_2p = (electron_2p_shell * cutoff_envelope) / vol_2p

    overlap_1s = np.sum(prob_1s * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * zg_1s.dV
    overlap_2p = np.sum(prob_2p * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * zg_2p.dV

    delta_M_1s_phys = 0.5 * (ALPHA**2) * red_mass_factor * M_e_phys * overlap_1s
    delta_M_2p_phys = 0.5 * (ALPHA**2) * red_mass_factor * M_e_phys * overlap_2p
    delta_M_photon_phys = delta_M_1s_phys - delta_M_2p_phys

    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0
    E_bind_1s_eV = delta_M_1s_phys * mass_conv_MeV * 1e6
    E_bind_2p_eV = delta_M_2p_phys * mass_conv_MeV * 1e6
    E_photon_eV = delta_M_photon_phys * mass_conv_MeV * 1e6

    # Golflengte van het uitgestraalde Lyman-alfa foton: lambda = h * c / E_photon
    h_planck = 2.0 * np.pi * HBAR
    lambda_photon_m = (h_planck * C) / (E_photon_eV * E_CHARGE) if E_photon_eV > 1e-12 else 0.0
    lambda_photon_nm = lambda_photon_m * 1e9

    # --- MOTOR 2: GEBOORTE EN DYNAMISCHE 3D GOLFPROPAGATIE VAN HET FOTON ---
    # Op t = 0 valt het elektron van 2p_z naar 1s. Het overtollige vacuüm-rekveld
    # delta_TR_0 = TR_2p - TR_1s wordt losgekoppeld en schiet als d'Alembert-golf weg!
    if verbose:
        print("4. Motor 2 (propagate_wave): Kwantumsprong 2p_z -> 1s ontkoppelt foton-golfpakket...")

    delta_TR_0 = zg_2p.TR - zg_1s.TR
    zg_wave = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    TR_oud = 1.0 + delta_TR_0
    TR_nu = TR_oud.copy()

    # Courant-Friedrichs-Lewy (CFL) tijdstap geschaald naar het atomaire rooster (dt/dx = 0.4)
    dt_atom = DX_ELECTRON * (DT_PHOTON / DX_PHOTON)
    mid = GRID_SHAPE_PHOTON[0] // 2
    n_steps = 80
    sample_steps = [35, 50, 65, 80]

    r_peaks = []
    t_peaks = []
    snapshots_1d = {}
    snapshot_2d_t0 = delta_TR_0[:, mid, :].copy()
    snapshot_2d_mid = None

    for step in range(1, n_steps + 1):
        TR_nieuw = zg_wave.propagate_wave(TR_oud, TR_nu, dt=dt_atom)
        TR_oud, TR_nu = TR_nu, TR_nieuw

        if step == 50:
            snapshot_2d_mid = (TR_nu[:, mid, :] - 1.0).copy()

        if step in sample_steps:
            t_cur = step * dt_atom
            prof_z = np.abs(TR_nu[mid, mid, mid+3:-5] - 1.0)
            z_coords = zg_wave.z[mid+3:-5]
            snapshots_1d[t_cur] = (z_coords.copy(), prof_z.copy())

            idx = int(np.argmax(prof_z))
            if 0 < idx < len(prof_z) - 1:
                y0, y1, y2 = prof_z[idx - 1], prof_z[idx], prof_z[idx + 1]
                denom = (y0 - 2.0 * y1 + y2)
                shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-15 else 0.0
                r_pk = z_coords[idx] + shift * DX_ELECTRON
            else:
                r_pk = z_coords[idx]

            r_peaks.append(r_pk)
            t_peaks.append(t_cur)

    # Lineaire fit dr/dt voor de voortplantingssnelheid v_photon (in eenheden van c)
    v_photon_model, _ = np.polyfit(t_peaks, r_peaks, 1)
    c_photon_phys = v_photon_model * C

    if verbose:
        print("\n" + "=" * 75)
        print("ETRT 3D FOTON-EMISSIE ANALYSE: WATERSTOF (2p_z -> 1s LYMAN-ALFA)")
        print("=" * 75)
        print(f"3D Overlap-integraal 1s grondtoestand:    {overlap_1s:.4f} (Theorie: 1.0000)")
        print(f"3D Overlap-integraal 2p_z aangeslagen:    {overlap_2p:.4f} (Theorie: 0.2500)")
        print(f"Bindingsenergie 1s grondtoestand (E_1s):  {E_bind_1s_eV:.4f} eV (Empirisch: 13.60 eV)")
        print(f"Bindingsenergie 2p_z toestand (E_2p):     {E_bind_2p_eV:.4f} eV (Empirisch: 3.40 eV)")
        print("-" * 75)
        print(f"UITGESTRAALDE FOTON-ENERGIE (E_gamma):    {E_photon_eV:.4f} eV (Empirisch: 10.20 eV)")
        print(f"FOTON GOLFLENGTE (Lyman-alfa):            {lambda_photon_nm:.2f} nm (Empirisch: 121.57 nm)")
        print("-" * 75)
        print(f"Gemeten 3D Golfsnelheid (v_foton):        {v_photon_model:.4f} c")
        print(f"Fysieke Lichtsnelheid uit Atoom-Emissie:  {c_photon_phys:.3e} m/s (Empirisch: {C:.3e} m/s)")
        print("=" * 75)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        extent = [zg_wave.z[0], zg_wave.z[-1], zg_wave.x[0], zg_wave.x[-1]]

        v_lim0 = np.percentile(np.abs(snapshot_2d_t0), 99.5)
        im0 = axes[0].imshow(
            snapshot_2d_t0, extent=extent, origin="lower", cmap="RdBu_r",
            vmin=-v_lim0, vmax=v_lim0
        )
        axes[0].set_title("t = 0: Losgelaten 2p_z -> 1s Veld (δTR_0)")
        axes[0].set_xlabel("z-as (Dipool-as) [Rooster-eenheden]")
        axes[0].set_ylabel("x-as [Rooster-eenheden]")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        v_lim1 = np.percentile(np.abs(snapshot_2d_mid), 99.5)
        im1 = axes[1].imshow(
            snapshot_2d_mid, extent=extent, origin="lower", cmap="RdBu_r",
            vmin=-v_lim1, vmax=v_lim1
        )
        axes[1].set_title(f"t = {50*dt_atom:.0f}: Wegreizend 3D Foton-Golfpakket")
        axes[1].set_xlabel("z-as (Dipool-as) [Rooster-eenheden]")
        axes[1].set_ylabel("x-as [Rooster-eenheden]")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        for t_val, (z_c, prof) in snapshots_1d.items():
            axes[2].plot(z_c, prof / np.max(prof), label=f"t = {t_val:.0f}")
        axes[2].set_title(f"Radiale Golfpropagatie (v = {v_photon_model:.4f} c)")
        axes[2].set_xlabel("Afstand r langs z-as [Rooster-eenheden]")
        axes[2].set_ylabel("Genormaliseerde Golfamplitude |δTR|")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("photon_emission_2p_1s.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'photon_emission_2p_1s.png'.")

    return {
        "Q0": 0.0,
        "M0_ph": 0.0,
        "E_bind_1s_eV": E_bind_1s_eV,
        "E_bind_2p_eV": E_bind_2p_eV,
        "E_photon_eV": E_photon_eV,
        "lambda_photon_nm": lambda_photon_nm,
        "delta_M_photon_phys": delta_M_photon_phys,
        "v_photon": v_photon_model,
        "c_photon_phys": c_photon_phys,
        "z_grid": zg_wave,
    }


def get_propagation_speed():
    """Hulpfunctie (compatibel met report.py) die de voortplantingssnelheid uit de 2p->1s emissie levert."""
    res = run_photon_emission_model(verbose=False, save_plot=False)
    return res["v_photon"]


if __name__ == "__main__":
    run_photon_emission_model()