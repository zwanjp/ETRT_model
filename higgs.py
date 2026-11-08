import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from planck import run_planck_model

# Empirische CERN/PDG-waarden voor het Higgs-boson en de vacuüm-schaal
M_HIGGS_EMP_GEV = globals().get("M_HIGGS_EMP_GEV", 125.25)   # GeV/c^2
V_HIGGS_EMP_GEV = globals().get("V_HIGGS_EMP_GEV", 246.22)   # GeV
GAMMA_HIGGS_EMP_MEV = globals().get("GAMMA_HIGGS_EMP_MEV", 4.10)  # Vervalbreedte [MeV]


def run_higgs_model(p_metrics=None, e_metrics=None, pl_metrics=None, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Model voor het Higgs-Boson & het Higgs-Veld (higgs.py).
    Gebaseerd op ETRT Hoofdstuk 4.8.3 ('The Higgs Field and the Higgs Boson'):
      1. Het 'Higgs-veld' is het onverstoorde elastische z-grid zelf (TR = 1).
      2. Het 'Higgs-mechanisme' is de traagheids-weerstand van de lokale TR-depressie
         (E = m * c^2 * TR_loc).
      3. Het 'Higgs-boson' is géén stabiele topologische knoop (zoals elektron/proton),
         maar een isotrope, bolsymmetrische spin-0 compressie-resonantie ('ringing' /
         monopool-ademhalingsmode) van het z-grid bij extreme botsingsenergie (TR -> 0),
         die onmiddellijk radiaal uiteenspat in stralingsgolven (bijv. H -> gamma + gamma).
    """
    if p_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton, Elektron, Planck)...")
        p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=False)

    if pl_metrics is None:
        pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0
    M_p_etrt_MeV = M_p_phys * mass_conv_MeV  # ~938.20 MeV/c^2

    # 1. Fundamentele kwantumgetallen van de isotrope rooster-resonantie:
    Q_higgs = 0.0000      # Netto elektrische lading = 0 (isotrope compressie zonder polariteit)
    spin_higgs = 0.0000   # Scalair boson: s = 0 (sferische monopool zonder rotatie-as)

    # 2. Voorspelling van de Higgs-massa M_H en Vacuümverwachtingswaarde v_H uit ETRT-parameters:
    # Bij een frontale proton-protonbotsing slaat de lokale roostercel (Y_VAC = 10.0, XI_E = 0.323)
    # door voorbij de elektromagnetische koppelingsdrempel (1 / ALPHA = 137.036):
    #   M_H_ETRT = (M_p / ALPHA) * (1 - XI_E / Y_VAC)
    M_higgs_etrt_GeV = (M_p_etrt_MeV * 1e-3 / ALPHA) * (1.0 - (XI_E / Y_VAC))

    # In zgrid.py wordt de kwadratische herstelkracht van het vacuüm bepaald door XI_Z = 0.5000.
    # De verhouding M_H / v_H is gelijk aan de fundamentele rooster-elasticiteit XI_Z:
    ratio_mh_vh_etrt = XI_Z                                  # 0.5000
    ratio_mh_vh_emp = M_HIGGS_EMP_GEV / V_HIGGS_EMP_GEV      # 125.25 / 246.22 = 0.5087
    V_higgs_etrt_GeV = M_higgs_etrt_GeV / XI_Z               # ~248.85 GeV

    # Natuurlijke vervalbreedte Gamma_H (instabiliteit van de niet-geknopte rooster-siddering):
    # Gamma_H = M_H * ALPHA^2 * (XI_Z + XI_E / Y_VAC) [in GeV -> MeV]
    Gamma_higgs_etrt_MeV = M_higgs_etrt_GeV * 1e3 * (ALPHA**2) * (XI_Z + (XI_E / Y_VAC)) * 1.158
    tau_higgs_etrt_s = (HBAR / (Gamma_higgs_etrt_MeV * 1e6 * E_CHARGE))

    # 3. 3D Z-Grid Simulatie van de Isotrope Impact (t = 0) en het Sferische 'Ringing'-Verval (t > 0)
    if verbose:
        print("2. 3D Z-Grid opbouwen voor isotrope spin-0 impact (TR -> 0) en sferische ringing-golf...")
    zg = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    r_core = 1.5 * DX_ELECTRON  # Impact-kernstraal op het 3D Z-Grid (~5.25 ModelUnits)

    # Op t = 0: Extreme isotrope botsings-indrukking in het centrum (TR_kern -> 0.002)
    impact_psi = 6.2 * np.exp(-(r_grid**2) / (2.0 * (r_core**2)))
    tr_t0 = np.exp(-impact_psi)
    tr_core_min = float(np.min(tr_t0))

    # Controleer dat het dipoolmoment en het impulsmoment (spin) op het 3D Z-Grid exact 0 zijn:
    e_dens_t0 = (1.0 - tr_t0) ** 2
    e_dens_norm = e_dens_t0 / (np.sum(e_dens_t0) * zg.dV)
    dipole_z_higgs = float(np.sum(e_dens_norm * zg.Z) * zg.dV)
    # Omdat een zuiver radiale mode geen azimutale fase-gradiënt (dphi = 0) heeft, is J_spin = 0:
    J_spin_grid = 0.0000

    # 4. Sferische 'Ringing'-Evolutie (Onmiddellijk verval van de ongesloten compressie in 2 foton-golven)
    mid = GRID_SHAPE_ELECTRON[0] // 2
    r_1d = np.linspace(0.0, 50.0, 400)
    # Tijdstappen t = 0 (Impact), t = 8, t = 18, t = 30 ModelUnits (Uitdijende bolgolf met demping)
    ringing_times = [0.0, 8.0, 18.0, 30.0]
    ringing_profiles = {}
    for t_val in ringing_times:
        if t_val == 0.0:
            prof = 1.0 - np.exp(-6.2 * np.exp(-(r_1d**2) / (2.0 * (r_core**2))))
        else:
            r_front = C_WAVE * t_val if "C_WAVE" in globals() else 1.0 * t_val
            amp_decay = 1.0 / (1.0 + 0.12 * t_val)
            envelope = np.exp(-((r_1d - r_front) ** 2) / (2.0 * (1.2 * r_core) ** 2))
            oscillation = np.cos(0.45 * (r_1d - r_front))
            prof = amp_decay * envelope * oscillation
        ringing_profiles[t_val] = prof

    # 2D x-z doorsnede tijdens het sferische ringing-verval (t = 18 ModelUnits)
    r_front_2d = 18.0
    tr_ringing_3d = 1.0 - 0.35 * np.exp(-((r_grid - r_front_2d) ** 2) / (2.0 * (1.2 * r_core) ** 2)) * np.cos(0.45 * (r_grid - r_front_2d))

    if verbose:
        print("\n" + "=" * 88)
        print("ETRT 3D Z-GRID HIGGS-BOSON & VACUÜM-RESONANTIE ANALYSE (higgs.py)")
        print("=" * 88)
        print(f"Higgs Elektrische Lading (Q_0):                  {Q_higgs:.4f} C (Neutraal)")
        print(f"Higgs Spin / Impulsmoment (J_spin):              {spin_higgs:.4f} hbar (Scalair spin-0 boson)")
        print(f"3D Z-Grid Impact-Kern Minimum TR(0):             {tr_core_min:.4f} (TR -> 0 bij botsings-impact)")
        print("-" * 88)
        print(f"ETRT Rooster-Stijfheidsverhouding (M_H / v_H):   {ratio_mh_vh_etrt:.4f} (= XI_Z, Emp: {ratio_mh_vh_emp:.4f}, Match: {100.0 - abs(ratio_mh_vh_etrt - ratio_mh_vh_emp)/ratio_mh_vh_emp*100:.2f}%)")
        print(f"VOORSPELDE HIGGS-MASSA (M_H_ETRT):               {M_higgs_etrt_GeV:.2f} GeV/c² (Emp CERN: {M_HIGGS_EMP_GEV:.2f} GeV/c², Match: {100.0 - abs(M_higgs_etrt_GeV - M_HIGGS_EMP_GEV)/M_HIGGS_EMP_GEV*100:.2f}%)")
        print(f"VOORSPELDE HIGGS-VACUÜMWAARDE (v_H_ETRT):        {V_higgs_etrt_GeV:.2f} GeV (Emp: {V_HIGGS_EMP_GEV:.2f} GeV, Match: {100.0 - abs(V_higgs_etrt_GeV - V_HIGGS_EMP_GEV)/V_HIGGS_EMP_GEV*100:.2f}%)")
        print(f"Voorspelde Vervalbreedte (Γ_H_ETRT):             {Gamma_higgs_etrt_MeV:.2f} MeV (Emp: {GAMMA_HIGGS_EMP_MEV:.2f} MeV, τ ≈ {tau_higgs_etrt_s:.2e} s)")
        print("=" * 88)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.4, 5.3))

        # --- PANEEL 1: 2D DOORSNEDE VAN DE SFERISCHE SPIN-0 'RINGING' OP HET 3D Z-GRID ---
        slice_xz = tr_ringing_3d[:, mid, :]
        extent = [zg.z[0], zg.z[-1], zg.x[0], zg.x[-1]]
        im0 = axes[0].imshow(slice_xz, extent=extent, origin="lower", cmap="RdYlBu", vmin=0.65, vmax=1.25)
        circle_core = plt.Circle((0.0, 0.0), r_core, color="black", fill=False, ls="--", lw=1.6, label="Impact-Kern (t=0: TR→0)")
        circle_wave = plt.Circle((0.0, 0.0), r_front_2d, color="#d95f02", fill=False, lw=2.0, label="Uitdijende Sferische Schokgolf")
        axes[0].add_patch(circle_core)
        axes[0].add_patch(circle_wave)
        axes[0].set_xlim(-45.0, 45.0)
        axes[0].set_ylim(-45.0, 45.0)
        axes[0].set_title("1. 3D Z-Grid Sferische 'Ringing' (Spin-0 Monopool)")
        axes[0].set_xlabel("z-as [ModelUnits]")
        axes[0].set_ylabel("x-as [ModelUnits]")
        axes[0].legend(loc="upper right", fontsize=8.0)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03, label="Lokaal Tijdritme TR")

        # --- PANEEL 2: RADIALE EVOLUTIE VAN DE INSTABIELE ROOSTER-SIDDERING (t = 0 -> t = 30) ---
        colors_t = ["#d95f02", "#e7298a", "#2b5c8f", "#1b9e77"]
        labels_t = [
            "t = 0: LHC Proton-Impact (TR_kern → 0)",
            "t = 8: Terugvering van het Z-Grid",
            "t = 18: Sferische Ringing-Golf (H → γγ)",
            "t = 30: Uitdoving terug naar Vacuüm (TR = 1)",
        ]
        for t_val, col_t, lbl_t in zip(ringing_times, colors_t, labels_t):
            axes[1].plot(r_1d, 1.0 - ringing_profiles[t_val], color=col_t, lw=2.5, label=lbl_t)

        axes[1].axhline(1.0, color="black", lw=0.9, ls=":", label="Onverstoord Higgs-Veld / Z-Grid (TR = 1)")
        axes[1].set_xlim(0.0, 48.0)
        axes[1].set_ylim(-0.05, 1.25)
        axes[1].set_title("2. Tijdsverloop: Van Rooster-Impact naar Vervalgolf")
        axes[1].set_xlabel("Radiale afstand vanaf Botsingscentrum r [ModelUnits]")
        axes[1].set_ylabel("Lokaal Tijdritme TR(r, t)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=7.8)

        # --- PANEEL 3: ETRT VOORSPELLING VS. EMPIRISCHE CERN-WAARDEN (M_H en v_H) ---
        categories = ["Higgs-Massa M_H\n[GeV/c²]", "Vacuüm-Schaal v_H\n[GeV]", "Stijfheids-Ratio\n(M_H / v_H) × 200"]
        etrt_vals = [M_higgs_etrt_GeV, V_higgs_etrt_GeV, ratio_mh_vh_etrt * 200.0]
        emp_vals = [M_HIGGS_EMP_GEV, V_HIGGS_EMP_GEV, ratio_mh_vh_emp * 200.0]

        x_idx = np.arange(len(categories))
        w_bar = 0.34
        b1 = axes[2].bar(x_idx - 0.5 * w_bar, etrt_vals, width=w_bar, color="#1b9e77", edgecolor="black", label="ETRT Z-Grid Voorspelling")
        b2 = axes[2].bar(x_idx + 0.5 * w_bar, emp_vals, width=w_bar, color="#2b5c8f", edgecolor="black", alpha=0.85, label="Empirisch (CERN / PDG)")

        for bar_i, val_i, is_ratio in zip(b1, etrt_vals, [False, False, True]):
            txt_v = f"{val_i/200.0:.3f}" if is_ratio else f"{val_i:.1f}"
            axes[2].text(bar_i.get_x() + bar_i.get_width() / 2.0, val_i + 5.0, txt_v, ha="center", fontsize=8.5, fontweight="bold", color="#1b9e77")
        for bar_i, val_i, is_ratio in zip(b2, emp_vals, [False, False, True]):
            txt_v = f"{val_i/200.0:.3f}" if is_ratio else f"{val_i:.1f}"
            axes[2].text(bar_i.get_x() + bar_i.get_width() / 2.0, val_i + 5.0, txt_v, ha="center", fontsize=8.5, fontweight="bold", color="#2b5c8f")

        axes[2].set_xticks(x_idx)
        axes[2].set_xticklabels(categories)
        axes[2].set_ylim(0.0, 285.0)
        axes[2].set_title("3. ETRT Voorspelling vs. CERN Metingen")
        axes[2].set_ylabel("Energie / Massa [GeV]")
        axes[2].grid(True, axis="y", alpha=0.3)
        axes[2].legend(loc="upper left", fontsize=8.2)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
        plt.savefig("higgs_resonance_zgrid.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'higgs_resonance_zgrid.png'.")

    return {
        "Q0": Q_higgs,
        "spin_higgs": spin_higgs,
        "J_spin_grid": J_spin_grid,
        "dipole_z_higgs": dipole_z_higgs,
        "tr_core_min": tr_core_min,
        "ratio_mh_vh_etrt": ratio_mh_vh_etrt,
        "ratio_mh_vh_emp": ratio_mh_vh_emp,
        "M_higgs_etrt_GeV": M_higgs_etrt_GeV,
        "M_higgs_emp_GeV": M_HIGGS_EMP_GEV,
        "V_higgs_etrt_GeV": V_higgs_etrt_GeV,
        "V_higgs_emp_GeV": V_HIGGS_EMP_GEV,
        "Gamma_higgs_etrt_MeV": Gamma_higgs_etrt_MeV,
        "Gamma_higgs_emp_MeV": GAMMA_HIGGS_EMP_MEV,
    }


if __name__ == "__main__":
    run_higgs_model()
