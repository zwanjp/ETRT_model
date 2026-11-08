import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from massfield import get_macroscopic_tr_field
from planck import run_planck_model


def run_qubit_shielding_test(g_tr_noise=0.0005, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van Qubit-Stabilisatie tegen TR-ruis (qubit_test_2.py).
    Vergelijkt voor een vaste omgevings-stoorgradiënt (g_TR = 0.0005 uit qubit_test_1.py)
    vier fysieke regimes op het 3D Z-Grid:
      1. Onbeschermde Qubit (Referentie uit qubit_test_1.py)
      2. Zware-Atomen Kooi (Octaëder van zware A=208 clusters die een diep TR-dal trekken via massfield.py)
      3. Negatief Statisch Coulomb-veld (Decomprimerende voorspanning P_ext < 0 via z_grid.relax)
      4. Gecombineerde ETRT-Stabilisatie (Diep TR-dal + Negatief Statisch Coulomb-veld)
    """
    if verbose:
        print("1. Ophalen van referentiemodellen en opbouwen van het 3D Z-Grid...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    # 2. 3D Z-Grid en Qubit Hadamard-superpositie |+> = (|0> + |1>) / sqrt(2)
    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    psi_0_raw = np.exp(-r_grid / r_bohr_grid) * cutoff
    psi_0 = psi_0_raw / np.sqrt(np.sum(psi_0_raw**2) * zg.dV)

    psi_1_raw = (zg.Z / r_bohr_grid) * np.exp(-r_grid / (2.0 * r_bohr_grid)) * cutoff
    psi_1 = psi_1_raw / np.sqrt(np.sum(psi_1_raw**2) * zg.dV)

    psi_plus = (psi_0 + psi_1) / np.sqrt(2.0)
    prob_plus = psi_plus**2

    # Ruimtelijke breedte sigma_z van de onbeschermde superpositie-wolk |+> op het 3D Z-Grid
    z_mean_plus = np.sum(prob_plus * zg.Z) * zg.dV
    z_var_plus = np.sum(prob_plus * ((zg.Z - z_mean_plus) ** 2)) * zg.dV
    sigma_z_unshielded = np.sqrt(z_var_plus)

    # 3. Schild A: Omringende Zware Atomen (6x A=208 Lood/Bismut/Tantaal wand-clusters in een octaëder-kooi)
    # Gepositioneerd op R_cage = 2.2 * r_bohr met effectieve schilstraal r0_heavy = 1.35 * r_bohr
    # (zodat R_cage < e * r0_heavy en de qubit midden in het collectieve TR-dal TR < 1 ligt)
    if verbose:
        print("2. Schild A doorrekenen: Octaëdrische kooi van zware atomen (A=208) via massfield.py...")
    R_cage = 2.20 * r_bohr_grid
    r0_heavy = 1.35 * r_bohr_grid
    A_heavy = 208.0
    omega_heavy = 0.14 * A_heavy * M_p_raw * OMEGA_MACRO_SCALE

    heavy_cage = [
        {"center": ( R_cage, 0.0, 0.0), "omega": omega_heavy, "r0": r0_heavy},
        {"center": (-R_cage, 0.0, 0.0), "omega": omega_heavy, "r0": r0_heavy},
        {"center": (0.0,  R_cage, 0.0), "omega": omega_heavy, "r0": r0_heavy},
        {"center": (0.0, -R_cage, 0.0), "omega": omega_heavy, "r0": r0_heavy},
        {"center": (0.0, 0.0,  R_cage), "omega": omega_heavy, "r0": r0_heavy},
        {"center": (0.0, 0.0, -R_cage), "omega": omega_heavy, "r0": r0_heavy},
    ]
    TR_heavy_field, psi_heavy_field = get_macroscopic_tr_field(
        zg.X, zg.Y, zg.Z, heavy_cage, r0=r0_heavy
    )

    # Gemiddeld verlaagd tijdritme <TR_heavy> en massfield-bufferstijfheid ter plaatse van de qubit
    tr_heavy_avg = float(np.sum(prob_plus * TR_heavy_field) * zg.dV)
    psi_heavy_avg = float(np.sum(prob_plus * psi_heavy_field) * zg.dV)
    # De zware TR-put vertraagt en dempt externe gradiënt-fluctuaties met factor TR_heavy / (1 + psi_heavy / XI_E)
    damping_heavy = tr_heavy_avg / (1.0 + (psi_heavy_avg / XI_E))

    # 4. Schild B: Sterk Negatief Statisch Coulomb-veld (Decomprimerende Kooi via z_grid.relax)
    if verbose:
        print("3. Schild B doorrekenen: Statisch negatief Coulomb-veld (P_ext < 0) via z_grid.relax()...")
    zg_coulomb = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    # Negatieve elektrostatische voorspanning (decomprimerende druk op het Z-Grid)
    p_coulomb_static = -0.18 * np.exp(-((r_grid / (2.5 * r_bohr_grid)) ** 2))
    zg_coulomb.relax(external_pressure=p_coulomb_static, iterations=RELAX_ITERATIONS)

    # Door de niet-lineaire herstelkracht dV/dTR = XI_Z * (1 - 1/TR^2) en elektrostatische opsluiting
    # wordt de kwetsbare 2p_z-buitenwolk compacter en stijgt de effectieve rooster-veerconstante k_eff:
    tr_coulomb_avg = float(np.sum(prob_plus * zg_coulomb.TR) * zg.dV)
    psi_plus_confined = psi_plus * np.exp(-0.25 * (r_grid / r_bohr_grid))
    prob_plus_confined = (psi_plus_confined**2) / (np.sum(psi_plus_confined**2) * zg.dV)
    z_mean_conf = np.sum(prob_plus_confined * zg.Z) * zg.dV
    sigma_z_confined = float(np.sqrt(np.sum(prob_plus_confined * ((zg.Z - z_mean_conf) ** 2)) * zg.dV))

    stiffness_boost = 1.0 + abs(tr_coulomb_avg - 1.0) / XI_E
    damping_coulomb = (sigma_z_confined / sigma_z_unshielded) / stiffness_boost

    # 5. Gecombineerd Schild (Zware Atomen TR-Dal + Negatief Statisch Coulomb-veld)
    damping_combined = damping_heavy * damping_coulomb

    # 6. Tijdsverloop van T2-coherentie |r(t)| en fase-spreiding sigma_phi(t) voor g_TR = 0.0005
    t_steps = np.linspace(0.0, 100.0, 100)
    regimes = [
        ("1. Onbeschermd (Referentie)", 1.0, "#d95f02", "--"),
        ("2. Zware Kernen TR-Dal (A=208)", damping_heavy, "#7570b3", "-."),
        ("3. Negatief Statisch Coulomb-veld", damping_coulomb, "#1b9e77", ":"),
        ("4. Gecombineerd Schild (TR-Dal + Coulomb)", damping_combined, "#2b5c8f", "-"),
    ]

    results = {}
    base_rate = g_tr_noise * r_bohr_grid * 20.0

    for label, damp_factor, color, ls in regimes:
        g_eff = g_tr_noise * damp_factor
        T2_eff = 1.0 / (base_rate * damp_factor) if damp_factor > 1e-12 else np.inf
        r_bloch_t = np.exp(-((t_steps / T2_eff) ** 2))
        phase_std_t = g_eff * t_steps * 10.0
        results[label] = {
            "damp_factor": damp_factor,
            "g_eff": g_eff,
            "T2": T2_eff,
            "T2_gain": 1.0 / damp_factor,
            "r_bloch_t": r_bloch_t,
            "phase_std_t": phase_std_t,
            "color": color,
            "ls": ls,
        }

    if verbose:
        print("\n" + "=" * 94)
        print(f"ETRT QUBIT-STABILISATIE ANALYSE BIJ STOORGRADIËNT g_TR = {g_tr_noise} (qubit_test_2.py)")
        print("=" * 94)
        print(f"Gemiddeld TR-niveau in Zware-Kernen Kooi (<TR_heavy>):  {tr_heavy_avg:.4f} (TR-reductie: {(1-tr_heavy_avg)*100:.2f}%)")
        print(f"Gemiddelde Z-Grid voorspanning Negatief Coulomb-veld:   {tr_coulomb_avg:.4f}")
        print(f"Ruimtelijke dipool-breedte σ_z (Onbeschermd -> Coulomb): {sigma_z_unshielded:.2f} -> {sigma_z_confined:.2f} [GridUnits]")
        print("-" * 94)
        print(f"{'Stabilisatie-Regime':<42} | {'g_eff':<10} | {'T2 [Units]':<11} | {'Winstfactor':<11} | {'|r(t=20)|'}")
        print("-" * 94)
        idx_t20 = int(np.argmin(np.abs(t_steps - 20.0)))
        for label, _, _, _ in regimes:
            r_info = results[label]
            print(
                f"{label:<42} | {r_info['g_eff']:<10.6f} | {r_info['T2']:<11.2f} | "
                f"{r_info['T2_gain']:<10.2f}x | {r_info['r_bloch_t'][idx_t20]:.4f}"
            )
        print("=" * 94)

    if save_plot:
        mid = GRID_SHAPE_PHOTON[0] // 2
        z_axis_bohr = zg.z / r_bohr_grid

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

        # Paneel 1: 1D Dwarsdoorsnede van het TR-Dal (Zware Kernen) en Decomprimerende Coulomb-kooi
        tr_slice_heavy = TR_heavy_field[mid, mid, :]
        tr_slice_coulomb = zg_coulomb.TR[mid, mid, :]
        prob_slice_norm = prob_plus[mid, mid, :] / np.max(prob_plus[mid, mid, :])

        axes[0].plot(z_axis_bohr, tr_slice_heavy, color="#7570b3", lw=2.2, label="TR(z) Zware-Kernen Dal (TR < 1)")
        axes[0].plot(z_axis_bohr, tr_slice_coulomb, color="#1b9e77", lw=2.2, ls="-.", label="TR(z) Negatief Coulomb-veld")
        axes[0].fill_between(
            z_axis_bohr, 0.65, 0.65 + 0.25 * prob_slice_norm, color="gold", alpha=0.35, label="Qubit |+> dichtheid (geschaald)"
        )
        axes[0].axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
        axes[0].set_xlim(-4.5, 4.5)
        axes[0].set_ylim(0.65, 1.18)
        axes[0].set_title("3D Z-Grid Profiel: TR-Dal & Coulomb-Buffer")
        axes[0].set_xlabel("z-as [Bohr-stralen a0]")
        axes[0].set_ylabel("Lokaal Tijdritme TR(z)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8.5)

        # Paneel 2: T2 Coherentiebehoud |r(t)| voor de 4 regimes
        for label, _, _, _ in regimes:
            r_info = results[label]
            axes[1].plot(
                t_steps, r_info["r_bloch_t"], color=r_info["color"], ls=r_info["ls"], lw=2.4,
                label=f"{label} ({r_info['T2_gain']:.1f}x)"
            )
        axes[1].set_title(f"T2 Coherentiebehoud |r(t)| bij g_TR = {g_tr_noise}")
        axes[1].set_xlabel("Tijd t [ModelUnits]")
        axes[1].set_ylabel("Bloch-norm |r(t)| (1.0 = Zuiver)")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8.5)

        # Paneel 3: Onderdrukking van de Fase-spreiding sigma_phi(t)
        for label, _, _, _ in regimes:
            r_info = results[label]
            axes[2].plot(
                t_steps, r_info["phase_std_t"], color=r_info["color"], ls=r_info["ls"], lw=2.4,
                label=label
            )
        axes[2].set_title("Onderdrukking van Fase-Ruis σ_phi(t)")
        axes[2].set_xlabel("Tijd t [ModelUnits]")
        axes[2].set_ylabel("Fase-spreiding σ_phi [rad]")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper left", fontsize=8.5)

        plt.tight_layout()
        plt.savefig("qubit_shielding_t2.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_shielding_t2.png'.")

    return results


if __name__ == "__main__":
    run_qubit_shielding_test()