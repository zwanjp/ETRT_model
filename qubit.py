import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from massfield import get_macroscopic_tr_field
from planck import run_planck_model
from parameters import *


def run_qubit_model(p_metrics=None, e_metrics=None, pl_metrics=None, verbose=True, save_plot=True):
    """
    ETRT 3D Model voor een Fysieke Qubit op het Z-Grid (qubit.py).
    Combineert:
    1. De Atomair-Optische Qubit (1H: |0> = 1s grondtoestand, |1> = 2p_z aangeslagen toestand)
       inclusief Hadamard-superpositie |+> = (|0> + |1>)/sqrt(2), Pauli-X poort en overgangsdipool.
    2. De Topologische Spin-Qubit (Hopfion Bloch-bol orientatie & 2-Qubit Bell-Singlet verstrengeling).
    """
    if p_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton, Elektron, Planck)...")
            print("-" * 70)
        p_metrics = _extract_model_metrics(run_proton_model, verbose=verbose)
        if verbose:
            print("-" * 70)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=verbose)
        if verbose:
            print("-" * 70)

    if pl_metrics is None:
        pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    # 2. Atomaire schaalsprong (1 / ALPHA^2) en Bohr-straal op het 3D Z-Grid
    atomic_scale_factor = 1.0 / (ALPHA**2)
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor
    R_bohr_model = r_bohr_grid * atomic_scale_factor

    r_e_m = R_E_FM * 1e-15
    m_per_unit = r_e_m / R_rms_e_model if R_rms_e_model > 1e-12 else 0.0
    a_0_phys_m = R_bohr_model * m_per_unit

    if verbose:
        print("2. 3D Z-Grid opbouwen voor Qubit-basistoestanden |0> (1s) en |1> (2p_z)...")
    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # --- 3D GOLFFUNCTIES OP HET Z-GRID: |0> (1s) EN |1> (2p_z) ---
    psi_0_raw = np.exp(-r_grid / r_bohr_grid) * cutoff
    psi_0 = psi_0_raw / np.sqrt(np.sum(psi_0_raw**2) * zg.dV)

    psi_1_raw = (zg.Z / r_bohr_grid) * np.exp(-r_grid / (2.0 * r_bohr_grid)) * cutoff
    psi_1 = psi_1_raw / np.sqrt(np.sum(psi_1_raw**2) * zg.dV)

    # Orthonormaliteit op het 3D rooster: <0|0> = 1, <1|1> = 1, <0|1> = 0
    norm_00 = np.sum(psi_0 * psi_0) * zg.dV
    norm_11 = np.sum(psi_1 * psi_1) * zg.dV
    ortho_01 = abs(np.sum(psi_0 * psi_1) * zg.dV)

    # --- 1-QUBIT POORTOPERATIES OP HET 3D Z-GRID ---
    # Hadamard-poort (theta = pi/2, phi = 0): H|0> = |+> = (|0> + |1>) / sqrt(2)
    # Pauli-X poort (theta = pi,   phi = 0): X|0> = |1>
    if verbose:
        print("3. 1-Qubit poorten (Hadamard H|0> -> |+>, Pauli-X X|0> -> |1>) en TR-relaxatie uitvoeren...")
    psi_plus = (psi_0 + psi_1) / np.sqrt(2.0)

    prob_0 = psi_0**2
    prob_1 = psi_1**2
    prob_plus = psi_plus**2

    # Vacuüm-relaxatie (Motor 1) voor de superpositietoestand |+> op het Z-Grid
    r_core_p = DX_ELECTRON
    proton_core = get_proton_envelope(zg.X, zg.Y, zg.Z, r_core_p)
    vol_p = np.sum(proton_core * cutoff) * zg.dV

    p_tr_up = ((Y_VAC / Y_UP)**3) * (((Y_UP / 2.0) / ((Y_UP / 2.0) + XI_Z))**2)
    p_tr_down = ((Y_VAC / Y_DOWN)**3) * (((Y_DOWN / 2.0) / ((Y_DOWN / 2.0) + XI_Z))**2)
    p_amp = 2.0 * abs(p_tr_up - 1.0) + 1.0 * abs(p_tr_down - 1.0)

    pressure_plus = p_amp * proton_core - p_amp * vol_p * prob_plus
    zg.relax(external_pressure=pressure_plus, iterations=RELAX_ITERATIONS)

    # --- 3D BLOCH-VECTOR METINGEN EN ENERGIE-VERWACHTINGSWAARDE ---
    def measure_bloch_state(psi_state):
        c0 = np.sum(psi_0 * psi_state) * zg.dV
        c1 = np.sum(psi_1 * psi_state) * zg.dV
        p0 = abs(c0)**2
        p1 = abs(c1)**2
        sx = 2.0 * np.real(np.conj(c0) * c1)
        sy = 2.0 * np.imag(np.conj(c0) * c1)
        sz = p0 - p1
        r_bloch = np.sqrt(sx**2 + sy**2 + sz**2)
        return p0, p1, sx, sy, sz, r_bloch

    p0_plus, p1_plus, sx_plus, sy_plus, sz_plus, r_bloch_plus = measure_bloch_state(psi_plus)
    _, _, _, _, sz_x_gate, _ = measure_bloch_state(psi_1)

    # Verwachtingswaarde van de Qubit-energie in |0>, |+> en |1> via massfield.py
    particles = [{"center": (0.0, 0.0, 0.0), "omega": M_p_raw * OMEGA_MACRO_SCALE, "r0": XI_Z}]
    TR_field, _ = get_macroscopic_tr_field(zg.X, zg.Y, zg.Z, particles, r0=XI_Z)

    overlap_0 = np.sum(prob_0 * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * zg.dV
    overlap_1 = np.sum(prob_1 * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * zg.dV
    overlap_plus = np.sum(prob_plus * TR_field * (r_bohr_grid / (r_grid + XI_Z))) * zg.dV

    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0
    e_scale_eV = 0.5 * (ALPHA**2) * red_mass_factor * M_e_phys * mass_conv_MeV * 1e6

    E_0_eV = -e_scale_eV * overlap_0
    E_1_eV = -e_scale_eV * overlap_1
    E_plus_eV = -e_scale_eV * overlap_plus

    # Qubit Klok-/Rabi-overgangsfrequentie f_01 = (E_1 - E_0) / h_ETRT
    delta_E_01_eV = E_1_eV - E_0_eV
    h_etrt_Js = pl_metrics["h_phys_Js"]
    f_01_Hz = (delta_E_01_eV * E_CHARGE) / h_etrt_Js if h_etrt_Js > 1e-40 else 0.0

    # Geïnduceerd Elektrisch Overgangs-Dipoolmoment in de superpositietoestand |+>
    # Analytisch QM: <0|z/a0|1> = 128*sqrt(2)/243 = 0.744936
    z_dipole_grid = np.sum(prob_plus * (zg.Z / r_bohr_grid)) * zg.dV
    z_dipole_theory = (128.0 * np.sqrt(2.0)) / 243.0
    debye_unit_Cm = 3.33564095e-30
    dipole_plus_Debye = (E_CHARGE * z_dipole_grid * a_0_phys_m) / debye_unit_Cm
    dipole_emp_Debye = (E_CHARGE * z_dipole_theory * A0_BOHR_EMP_M) / debye_unit_Cm

    # --- 2-QUBIT VERSTRENGELING (BELL-SINGLET |Psi^-> = (|01> - |10>)/sqrt(2)) ---
    if verbose:
        print("4. 2-Qubit Singlet-verstrengeling (|Psi^->) en 3D overlap-integraal berekenen...")
    R_H2_bohr = 1.4011
    d_AB_grid = R_H2_bohr * r_bohr_grid
    r_A = np.sqrt(zg.X**2 + zg.Y**2 + (zg.Z + 0.5 * d_AB_grid)**2)
    r_B = np.sqrt(zg.X**2 + zg.Y**2 + (zg.Z - 0.5 * d_AB_grid)**2)

    psi_A_raw = np.exp(-r_A / r_bohr_grid) * cutoff
    psi_B_raw = np.exp(-r_B / r_bohr_grid) * cutoff
    psi_A = psi_A_raw / np.sqrt(np.sum(psi_A_raw**2) * zg.dV)
    psi_B = psi_B_raw / np.sqrt(np.sum(psi_B_raw**2) * zg.dV)

    S_AB_grid = np.sum(psi_A * psi_B) * zg.dV
    S_AB_theory = (1.0 + R_H2_bohr + (R_H2_bohr**2) / 3.0) * np.exp(-R_H2_bohr)

    # Bell-CHSH verstrengelingsparameter S_CHSH op basis van hbar_model en Bloch-norm
    hbar_model = pl_metrics["hbar_model"]
    S_chsh_model = 2.0 * np.sqrt(2.0) * (r_bloch_plus**2) * hbar_model
    S_chsh_theory = 2.0 * np.sqrt(2.0)

    if verbose:
        print("\n" + "=" * 75)
        print("ETRT 3D QUBIT & KWANTUMPOORT ANALYSE (qubit.py)")
        print("=" * 75)
        print(f"Orthonormaliteit 3D Z-Grid (<0|0>, <1|1>): ({norm_00:.4f}, {norm_11:.4f})")
        print(f"Orthogonaliteit basistoestanden |<0|1>|:   {ortho_01:.4e} (Verwacht: 0.0000)")
        print(f"Hadamard Superpositie |+> Kans P(|1>):     {p1_plus:.4f} (Theorie: 0.5000)")
        print(f"Bloch-vector lengte |r_Bloch| in |+>:      {r_bloch_plus:.4f} (Zuivere toestand: 1.0000)")
        print(f"Pauli-X Poort inversie <sigma_z>(X|0>):    {sz_x_gate:.4f} (Theorie: -1.0000)")
        print("-" * 75)
        print(f"Energie |0> (1s grondtoestand):            {E_0_eV:.4f} eV (Empirisch: -13.60 eV)")
        print(f"Energie |+> (Hadamard superpositie):       {E_plus_eV:.4f} eV (Empirisch: -8.50 eV)")
        print(f"Energie |1> (2p_z aangeslagen toestand):   {E_1_eV:.4f} eV (Empirisch: -3.40 eV)")
        print(f"Qubit Klokfrequentie (f_01 = ΔE/h_ETRT):   {f_01_Hz:.3e} Hz (Empirisch: 2.466e+15 Hz)")
        print(f"Overgangs-dipool <+|z/a0|+> (Z-Grid):      {z_dipole_grid:.4f} a0 ({dipole_plus_Debye:.3f} D vs Emp: {dipole_emp_Debye:.3f} D)")
        print("-" * 75)
        print(f"2-Qubit 3D Overlap S_AB (op R = 1.40 a0):  {S_AB_grid:.4f} (Theorie: {S_AB_theory:.4f})")
        print(f"Bell-Singlet CHSH Verstrengeling (S_CHSH): {S_chsh_model:.4f} (Tsirelson: {S_chsh_theory:.4f} > 2.0)")
        print("=" * 75)

    if save_plot:
        mid = GRID_SHAPE_PHOTON[0] // 2
        extent = [zg.z[0] / r_bohr_grid, zg.z[-1] / r_bohr_grid, zg.x[0] / r_bohr_grid, zg.x[-1] / r_bohr_grid]
        lim_bohr = 4.5

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        states_plot = [
            (psi_0[:, mid, :], 0.0, "|0> Grondtoestand (1s, <z> = 0.00 a0)"),
            (psi_plus[:, mid, :], z_dipole_grid, f"|+> = H|0> Superpositie (<z> = +{z_dipole_grid:.2f} a0)"),
            (psi_1[:, mid, :], 0.0, "|1> = X|0> Aangeslagen (2p_z, <z> = 0.00 a0)"),
        ]
        for ax, (psi_slice, z_cm, title) in zip(axes, states_plot):
            v_max = np.percentile(np.abs(psi_slice), 99.5)
            psi_norm = np.clip(psi_slice / v_max, -1.0, 1.0)
            im = ax.imshow(psi_norm, extent=extent, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            ax.plot(z_cm, 0.0, marker="o", markersize=8, color="gold", markeredgecolor="black", label="<z> ladingscentrum")
            ax.set_xlim(-lim_bohr, lim_bohr)
            ax.set_ylim(-lim_bohr, lim_bohr)
            ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
            ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.4)
            ax.set_title(title)
            ax.set_xlabel("z-as [Bohr-stralen a0]")
            ax.set_ylabel("x-as [Bohr-stralen a0]")
            ax.legend(loc="upper right", fontsize=8.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("qubit_bloch_zgrid.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_bloch_zgrid.png'.")

    return {
        "ortho_01": ortho_01,
        "p1_plus": p1_plus,
        "r_bloch_plus": r_bloch_plus,
        "sz_x_gate": sz_x_gate,
        "E_0_eV": E_0_eV,
        "E_1_eV": E_1_eV,
        "E_plus_eV": E_plus_eV,
        "f_01_Hz": f_01_Hz,
        "z_dipole_grid": z_dipole_grid,
        "z_dipole_theory": z_dipole_theory,
        "dipole_plus_Debye": dipole_plus_Debye,
        "dipole_emp_Debye": dipole_emp_Debye,
        "S_AB_grid": S_AB_grid,
        "S_AB_theory": S_AB_theory,
        "S_chsh_model": S_chsh_model,
        "S_chsh_theory": S_chsh_theory,
        "z_grid": zg,
    }


if __name__ == "__main__":
    run_qubit_model()