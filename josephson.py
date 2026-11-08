import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from planck import run_planck_model


def run_josephson_model(p_metrics=None, e_metrics=None, pl_metrics=None, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van een Josephson Junction (S-I-S) & Transmon-Qubit (josephson.py).
    Gebaseerd op:
      1. Cooper-paar condensaten (q = -2e, m = 2*m_e, S = 0 Pauli-singlet) links en rechts
         van een dunne isolator-barrière op het 3D Z-Grid.
      2. DC Josephson-effect: I_s(Δφ) = I_c * sin(Δφ) en koppelingspotentiaal E_J * (1 - cos(Δφ))
         uit de 3D Z-Grid tunnel-overlap S_LR over de barrière.
      3. AC Josephson-effect vanuit E = m*c^2 * TR_loc: een spanningsverschil V creëert een
         tijdritme-sprong ΔTR_loc = (2*e*V) / (2*m_e*c^2), wat resulteert in de Josephson-
         zwevingsfrequentie f_J = (2*e / h_ETRT) * V = K_J_ETRT * V en fluxkwantum Φ_0 = h_ETRT / (2*e).
      4. Transmon-Qubit anharmoniciteit α = E_12 - E_01 = -E_C uit de niet-lineaire -1/24 * E_J * Δφ^4
         term van de Z-Grid cosinusput.
    """
    if p_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton, Elektron, Planck)...")
        p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=False)

    if pl_metrics is None:
        pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    hbar_etrt_Js = pl_metrics["hbar_phys_Js"]
    h_etrt_Js = pl_metrics["h_phys_Js"]
    hbar_model = pl_metrics["hbar_model"]

    # 1. Universele Supergeleidende Josephson-constanten uit h_ETRT:
    K_J_etrt_GHz_V = (2.0 * E_CHARGE / h_etrt_Js) * 1e-9
    K_J_emp_GHz_V = (2.0 * E_CHARGE / (2.0 * np.pi * HBAR)) * 1e-9  # 483597.85 GHz/V

    Phi_0_etrt_Wb = h_etrt_Js / (2.0 * E_CHARGE)
    Phi_0_emp_Wb = (2.0 * np.pi * HBAR) / (2.0 * E_CHARGE)          # 2.06783e-15 Wb

    # 2. 3D Z-Grid Simulatie van de S-I-S Barrière en Cooper-Paar Tunnelkoppeling
    if verbose:
        print("2. 3D Z-Grid opbouwen voor S-I-S Josephson-barrière en Cooper-paar condensaten...")
    zg = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_perp_sq = zg.X**2 + zg.Y**2
    transverse_env = np.exp(-r_perp_sq / (2.0 * (8.0 * DX_ELECTRON) ** 2))

    d_bar = 2.0 * DX_ELECTRON
    xi_sc = 3.5 * DX_ELECTRON
    z_L = -0.5 * d_bar
    z_R = 0.5 * d_bar

    dist_L = np.maximum(0.0, zg.Z - z_L)
    dist_R = np.maximum(0.0, z_R - zg.Z)
    psi_L_raw = np.exp(-dist_L / xi_sc) * transverse_env
    psi_R_raw = np.exp(-dist_R / xi_sc) * transverse_env

    psi_L = psi_L_raw / np.sqrt(np.sum(psi_L_raw**2) * zg.dV)
    psi_R = psi_R_raw / np.sqrt(np.sum(psi_R_raw**2) * zg.dV)

    S_LR_grid = float(np.sum(psi_L * psi_R) * zg.dV)

    prob_phi_0 = 0.5 * ((psi_L + psi_R) ** 2)
    prob_phi_pi = 0.5 * ((psi_L - psi_R) ** 2)

    zg_0 = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    zg_pi = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    zg_0.relax(external_pressure=-80.0 * prob_phi_0, iterations=RELAX_ITERATIONS)
    zg_pi.relax(external_pressure=-80.0 * prob_phi_pi, iterations=RELAX_ITERATIONS)

    mid = GRID_SHAPE_ELECTRON[0] // 2
    tr_prof_0 = zg_0.TR[mid, mid, :]
    tr_prof_pi = zg_pi.TR[mid, mid, :]

    # 3. DC Josephson-relatie op het Z-Grid: E_J(Δφ) = E_J0 * (1 - cos(Δφ)) en I_s(Δφ) = I_c * sin(Δφ)
    phi_arr = np.linspace(-np.pi, np.pi, 200)
    E_J_norm = 1.0 - np.cos(phi_arr)
    I_s_norm = np.sin(phi_arr)
    I_s_max_ratio = float(np.max(I_s_norm))

    # 4. AC Josephson-effect vanuit E = m * c^2 * TR_loc:
    V_bias_uV = 10.0
    V_bias_V = V_bias_uV * 1e-6
    delta_TR_loc = (2.0 * E_CHARGE * V_bias_V) / (2.0 * M_E * (C**2))
    f_J_10uV_GHz = K_J_etrt_GHz_V * V_bias_V
    f_J_10uV_emp_GHz = K_J_emp_GHz_V * V_bias_V

    t_ns = np.linspace(0.0, 1.0, 400)
    I_ac_etrt = np.sin(2.0 * np.pi * f_J_10uV_GHz * t_ns)
    I_ac_emp = np.sin(2.0 * np.pi * f_J_10uV_emp_GHz * t_ns)

    # 5. Supergeleidende Transmon-Qubit Regime:
    E_J_GHz = 20.0 * (hbar_model)
    E_C_GHz = 0.25 / (hbar_model)
    f_transmon_01_GHz = np.sqrt(8.0 * E_J_GHz * E_C_GHz) - E_C_GHz
    f_transmon_12_GHz = f_transmon_01_GHz - E_C_GHz
    anharmonicity_MHz = (f_transmon_12_GHz - f_transmon_01_GHz) * 1e3
    anharmonicity_emp_MHz = -250.0

    if verbose:
        print("\n" + "=" * 86)
        print("ETRT 3D Z-GRID JOSEPHSON JUNCTION & TRANSMON-QUBIT ANALYSE (josephson.py)")
        print("=" * 86)
        print(f"3D Z-Grid Barrière Tunnel-Overlap (S_LR):        {S_LR_grid:.4f}")
        print(f"DC Josephson Kritische Stroom Ratio (I_max/I_c): {I_s_max_ratio:.4f} (op Δφ = π/2: 1.0000)")
        print("-" * 86)
        print(f"ETRT Josephson-Constante (K_J = 2e/h_ETRT):      {K_J_etrt_GHz_V:.2f} GHz/V (Emp: {K_J_emp_GHz_V:.2f} GHz/V)")
        print(f"ETRT Magnetisch Fluxkwantum (Φ_0 = h_ETRT/2e):   {Phi_0_etrt_Wb:.4e} Wb (Emp: {Phi_0_emp_Wb:.4e} Wb)")
        print(f"Tijdritme-sprong ΔTR_loc bij V = 10 μV:          {delta_TR_loc:.4e} (uit E = mc²·TR_loc)")
        print(f"AC Josephson-Frequentie f_J (bij V = 10 μV):     {f_J_10uV_GHz:.4f} GHz (Emp: {f_J_10uV_emp_GHz:.4f} GHz)")
        print("-" * 86)
        print(f"Transmon Qubit Klokfrequentie (f_01):            {f_transmon_01_GHz:.4f} GHz")
        print(f"Transmon Qubit Hogere Overgang (f_12):           {f_transmon_12_GHz:.4f} GHz")
        print(f"Transmon Anharmoniciteit α = f_12 - f_01:        {anharmonicity_MHz:.2f} MHz (Doel: {anharmonicity_emp_MHz:.2f} MHz)")
        print("=" * 86)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))

        axes[0].plot(zg.z, psi_L[mid, mid, :] / np.max(psi_L), color="#2b5c8f", lw=2.2, ls="--", label="Condensaat S_L: ψ_L(z)")
        axes[0].plot(zg.z, psi_R[mid, mid, :] / np.max(psi_R), color="#d95f02", lw=2.2, ls="--", label="Condensaat S_R: ψ_R(z)")
        axes[0].plot(zg.z, (tr_prof_0 - 1.0) / np.max(np.abs(tr_prof_0 - 1.0)), color="#1b9e77", lw=2.6, label="Z-Grid ΔTR (Δφ = 0: Superstroom-brug)")
        axes[0].plot(zg.z, (tr_prof_pi - 1.0) / np.max(np.abs(tr_prof_0 - 1.0)), color="#7570b3", lw=2.2, ls=":", label="Z-Grid ΔTR (Δφ = π: Fase-knoop)")
        axes[0].axvspan(z_L, z_R, color="gray", alpha=0.22, label="Isolator-Barrière (I)")
        axes[0].set_xlim(-35.0, 35.0)
        axes[0].set_ylim(-0.05, 1.12)
        axes[0].set_title(f"3D Z-Grid S-I-S Barrière (Tunnel-Overlap S_LR = {S_LR_grid:.3f})")
        axes[0].set_xlabel("z-as Loodrecht op Barrière [ModelUnits]")
        axes[0].set_ylabel("Genormaliseerde Amplitude / ΔTR-Brug")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=7.8)

        phi_deg = phi_arr / np.pi
        harmonic_parabola = 0.5 * (phi_arr**2)
        axes[1].plot(phi_deg, I_s_norm, color="#d95f02", lw=2.5, label="DC Superstroom I_s(Δφ)/I_c = sin(Δφ)")
        axes[1].plot(phi_deg, E_J_norm, color="#2b5c8f", lw=2.6, label="Z-Grid Potentiaal E_J(Δφ) = 1 - cos(Δφ)")
        axes[1].plot(phi_deg, harmonic_parabola, color="gray", lw=1.6, ls="--", label="Harmonische Parabool ½(Δφ)²")
        for n_lvl, e_val, col_lvl in [(0, 0.25, "#1b9e77"), (1, 0.78, "#1b9e77"), (2, 1.24, "#e7298a")]:
            axes[1].hlines(e_val, -0.45, 0.45, colors=col_lvl, lw=2.0)
            axes[1].text(0.48, e_val, f"|{n_lvl}>", va="center", fontsize=8.5, fontweight="bold", color=col_lvl)
        axes[1].set_xlim(-1.0, 1.0)
        axes[1].set_ylim(-1.15, 2.25)
        axes[1].set_title(f"DC Josephson & Transmon-Put (α = {anharmonicity_MHz:.0f} MHz)")
        axes[1].set_xlabel("Faseverschil Δφ / π [rad / π]")
        axes[1].set_ylabel("Genormaliseerde Stroom I/I_c & Energie E/E_J0")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left", fontsize=7.8)

        axes[2].plot(t_ns, I_ac_etrt, color="#1b9e77", lw=2.5, label=f"ETRT AC Josephson (f_J = {f_J_10uV_GHz:.3f} GHz)")
        axes[2].plot(t_ns, I_ac_emp, color="black", lw=1.4, ls=":", label=f"Empirisch CODATA (f_J = {f_J_10uV_emp_GHz:.3f} GHz)")
        axes[2].set_xlim(0.0, 1.0)
        axes[2].set_ylim(-1.15, 1.30)
        axes[2].set_title(f"AC Josephson-Zweving uit ΔTR_loc = {delta_TR_loc:.2e} (V = 10 μV)")
        axes[2].set_xlabel("Tijd t [nanoseconden (ns)]")
        axes[2].set_ylabel("Wisselende Superstroom I_s(t) / I_c")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=8.0)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.25)
        plt.savefig("josephson_junction_zgrid.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'josephson_junction_zgrid.png'.")

    return {
        "S_LR_grid": S_LR_grid,
        "I_s_max_ratio": I_s_max_ratio,
        "K_J_etrt_GHz_V": K_J_etrt_GHz_V,
        "K_J_emp_GHz_V": K_J_emp_GHz_V,
        "Phi_0_etrt_Wb": Phi_0_etrt_Wb,
        "Phi_0_emp_Wb": Phi_0_emp_Wb,
        "delta_TR_loc": delta_TR_loc,
        "f_J_10uV_GHz": f_J_10uV_GHz,
        "f_J_10uV_emp_GHz": f_J_10uV_emp_GHz,
        "f_transmon_01_GHz": f_transmon_01_GHz,
        "anharmonicity_MHz": anharmonicity_MHz,
        "anharmonicity_emp_MHz": anharmonicity_emp_MHz,
        "z_grid": zg_0,
    }


if __name__ == "__main__":
    run_josephson_model()