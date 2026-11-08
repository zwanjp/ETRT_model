import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model
from neutron import run_neutron_model
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from planck import run_planck_model
from photon_emission import run_photon_emission_model
from casimir import run_casimir_model
from josephson import run_josephson_model
from qubit_test_4 import _compute_8node_cube_binding


def run_casimir_josephson_tube_test(g_tr_noise=0.0005, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van een Dubbelwandige Coaxiale Casimir-Buis met een
    Periodieke Keten van 4-Punts Josephson-Ringen (qubit_test_6.py).

    Onderzoekt het theoretische experiment:
      1. Een lange, flinterdunne dubbelwandige cilinder (stralen R_in en R_out) op het
         3D Z-Grid die zowel tussen de wanden als in de evanescente buiten-/binnenhuid
         een verlaagd tijdritme (TR < 1) en vlakke radiale gradiënt creëert.
      2. Periodiek langs de buis geplaatste 4-punts Josephson-plaatjes (ringen op z_k),
         waarbij elke twee opeenvolgende 4-punts ringen (k, k+1) samen één 8-punts kubus
         (T_+ U T_-, Majorana Octon) vormen en hun 4-punts tussenvlak delen.
      3. Coherentiewinst (T2) en End-to-End Kwantumtransport-Fidelity F(N_hops) langs een
         keten van N = 1..8 geschakelde kubussen onder omgevingsruis g_TR = 0.0005.
    """
    if verbose:
        print("1. Ophalen van referentiemodellen (Proton, Neutron, Elektron, Planck, Casimir, Josephson)...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    n_metrics = _extract_model_metrics(run_neutron_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)
    ph_metrics = run_photon_emission_model(p_metrics=p_metrics, e_metrics=e_metrics, verbose=False, save_plot=False)
    cas_metrics = run_casimir_model(
        p_metrics=p_metrics, e_metrics=e_metrics, pl_metrics=pl_metrics, ph_metrics=ph_metrics,
        verbose=False, save_plot=False
    )
    jos_metrics = run_josephson_model(
        p_metrics=p_metrics, e_metrics=e_metrics, pl_metrics=pl_metrics,
        verbose=False, save_plot=False
    )

    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    # 2. 3D Z-Grid Simulatie van de Dubbelwandige Coaxiale Casimir-Buis + 5 Josephson-Ringen (4 Kubussen)
    if verbose:
        print("2. 3D Z-Grid opbouwen voor de dubbelwandige Casimir-buis en 5 gekoppelde 4-punts Josephson-ringen...")
    zg_tube = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_cyl = np.sqrt(zg_tube.X**2 + zg_tube.Y**2)

    R_in = 2.0 * DX_ELECTRON     # Binnenwand straal: 7.0 ModelUnits
    R_out = 4.0 * DX_ELECTRON    # Buitenwand straal: 14.0 ModelUnits
    d_wall_gap = R_out - R_in    # Coaxiale Casimir-spouwbreedte: 7.0 ModelUnits

    in_annulus = (r_cyl >= R_in) & (r_cyl <= R_out)
    wall_in = np.exp(-((r_cyl - R_in) ** 2) / (2.0 * (0.6 * DX_ELECTRON) ** 2))
    wall_out = np.exp(-((r_cyl - R_out) ** 2) / (2.0 * (0.6 * DX_ELECTRON) ** 2))

    mode_r_sum = np.zeros_like(r_cyl)
    for n_m in range(1, 5):
        k_r = n_m * np.pi / d_wall_gap
        mode_r_sum += (1.0 / (n_m**2)) * (np.sin(k_r * (r_cyl - R_in)) ** 2)
    mode_r_sum /= np.max(mode_r_sum)

    casimir_tube_pressure = 0.08 * in_annulus * (1.0 - 0.35 * mode_r_sum) + 0.14 * (wall_in + wall_out)

    N_rings = 5
    N_cubes = N_rings - 1
    R_sq = 2.0 * DX_ELECTRON
    dz_ring = 4.0 * DX_ELECTRON
    z_rings = np.array([(k - 0.5 * (N_rings - 1)) * dz_ring for k in range(N_rings)])

    josephson_chain_density = np.zeros_like(zg_tube.X)
    sigma_node = 1.15 * DX_ELECTRON

    for k_idx, z_k in enumerate(z_rings):
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                x_node = sx * R_sq
                y_node = sy * R_sq
                r_node_sq = (zg_tube.X - x_node) ** 2 + (zg_tube.Y - y_node) ** 2 + (zg_tube.Z - z_k) ** 2
                josephson_chain_density += np.exp(-r_node_sq / (2.0 * sigma_node**2))

    josephson_chain_density /= np.sum(josephson_chain_density) * zg_tube.dV

    zg_tube.relax(
        external_pressure=(casimir_tube_pressure + 45.0 * josephson_chain_density),
        iterations=RELAX_ITERATIONS
    )

    mid = GRID_SHAPE_ELECTRON[0] // 2
    tr_radial_profile = zg_tube.TR[:, mid, mid]
    tr_core_center = float(tr_radial_profile[mid])
    tr_annulus_min = float(np.min(zg_tube.TR[in_annulus]))
    idx_skin_out = mid + int(round((R_out + DX_ELECTRON) / DX_ELECTRON))
    tr_outer_skin = float(tr_radial_profile[idx_skin_out])
    tr_effective_chain = float(np.sum(josephson_chain_density * zg_tube.TR) * zg_tube.dV)

    # 3. Controleer 3D Multipool-uitdoving van de 4-Kubus Keten
    dip_z_chain = float(np.sum(josephson_chain_density * zg_tube.Z) * zg_tube.dV) / dz_ring
    quad_xy_chain = float(np.sum(josephson_chain_density * (zg_tube.X**2 - zg_tube.Y**2)) * zg_tube.dV) / (R_sq**2)
    oct_xyz_chain = float(np.sum(josephson_chain_density * (zg_tube.X * zg_tube.Y * zg_tube.Z)) * zg_tube.dV) / (R_sq**3)

    # 4. Berekening van de Coherentiewinst (T2) van de Coaxiale Casimir-Josephson 8Q-Keten
    if verbose:
        print("3. Coherentiewinst (T2) en Kwantumtransport over N = 1..8 gekoppelde kubussen doorrekenen...")
    E_bind_2Q_MeV = 2.2245
    E_bind_8Q_MeV = _compute_8node_cube_binding(p_metrics, n_metrics, e_metrics)
    damping_2Q = 1.0 / (1.0 + E_bind_2Q_MeV / 1.50)
    damping_8Q_bare = damping_2Q / (0.5 * (E_bind_8Q_MeV / E_bind_2Q_MeV))

    S_LR_jos = jos_metrics["S_LR_grid"]
    shared_face_factor = (8.0 * N_cubes) / (4.0 * N_rings)
    josephson_lock_factor = 1.0 + (S_LR_jos / XI_E)
    casimir_coulomb_shield = (1.0 / tr_effective_chain) * 2.025

    total_tube_shield_gain = shared_face_factor * josephson_lock_factor * casimir_coulomb_shield
    damping_8Q_casimir_jos = damping_8Q_bare / total_tube_shield_gain

    base_rate = g_tr_noise * r_bohr_grid * 20.0
    T2_1Q = 1.0 / base_rate
    T2_4Q_tet = T2_1Q * 15.32
    T2_8Q_bare = 1.0 / (base_rate * damping_8Q_bare)
    T2_8Q_static_cage = T2_8Q_bare * 3.7885
    T2_8Q_casimir_tube = 1.0 / (base_rate * damping_8Q_casimir_jos)
    gain_casimir_tube = T2_8Q_casimir_tube / T2_1Q

    # 5. Simulatie van End-to-End Kwantumtransport langs N_hops = 1 .. 8 geschakelde kubussen
    S_edge_base = 0.6972
    Omega_hop = 0.25 * (S_edge_base + 0.5 * S_LR_jos)
    tau_hop = np.pi / Omega_hop

    hops_arr = np.arange(1, 9)

    bus_architectures = [
        ("1. Onbeschermde 1Q/2Q Keten", T2_1Q, "#d95f02", "--", "o"),
        ("2. 4-Qubit Tetraëder Keten (4Q)", T2_4Q_tet, "#7570b3", ":", "s"),
        ("3. 8-Qubit Kubus Keten (Losse 8Q)", T2_8Q_bare, "#e7298a", "-.", "^"),
        ("4. 8-Qubit Kubus + Statische Kooi", T2_8Q_static_cage, "#2b5c8f", "--", "D"),
        ("5. Casimir-Buis + 4P Josephson-Keten", T2_8Q_casimir_tube, "#1b9e77", "-", "*"),
    ]

    transport_results = {}
    for label, T2_val, color, ls, marker in bus_architectures:
        r_chain_N = np.exp(-hops_arr * ((tau_hop / T2_val) ** 2))
        F_chain_N = 0.5 * (1.0 + r_chain_N)
        err_chain_N = 1.0 - F_chain_N
        transport_results[label] = {
            "T2": T2_val,
            "T2_gain": T2_val / T2_1Q,
            "F_1hop": float(F_chain_N[0]),
            "F_4hops": float(F_chain_N[3]),
            "F_8hops": float(F_chain_N[7]),
            "err_4hops": float(err_chain_N[3]),
            "err_8hops": float(err_chain_N[7]),
            "F_chain_N": F_chain_N,
            "err_chain_N": err_chain_N,
            "color": color,
            "ls": ls,
            "marker": marker,
        }

    if verbose:
        print("\n" + "=" * 108)
        print(f"ETRT COAXIALE CASIMIR-BUIS & 4-PUNTS JOSEPHSON-KETEN ANALYSE BIJ g_TR = {g_tr_noise} (qubit_test_6.py)")
        print("=" * 108)
        print(
            f"Radiaal TR-Profiel Buis: Holle Kern TR(0) = {tr_core_center:.4f} | "
            f"Casimir-Spouw TR_min = {tr_annulus_min:.4f} | Buitenhuid TR_ext = {tr_outer_skin:.4f}"
        )
        print(
            f"Keten-Geometrie: {N_rings} Vierkante 4P-Ringen (20 knopen) = {N_cubes} Geschakelde 8P-Kubussen "
            f"(32-knoops equivalent, besparing 37.5%)"
        )
        print(
            f"3D Multipolen Keten: Dipool <z> = {dip_z_chain:+.4f}, "
            f"Quadrupool <x²-y²> = {quad_xy_chain:+.4f}, Octupool <xyz> = {oct_xyz_chain:+.4f}"
        )
        print(f"Axiale Kubus-naar-Kubus Staptijd (τ_hop): {tau_hop:.2f} ModelUnits")
        print("-" * 108)
        print(
            f"{'Kwantumbus / Keten-Architectuur':<38} | {'T2 [Units]':<10} | {'Winst':<8} | "
            f"{'F(1 Kubus)':<12} | {'F(4 Kubussen)':<14} | {'F(8 Kubussen)':<14} | {'Fout(4 Kub)'}"
        )
        print("-" * 108)
        for label, _, _, _, _ in bus_architectures:
            info = transport_results[label]
            print(
                f"{label:<38} | {info['T2']:<10.2f} | {info['T2_gain']:<7.1f}x | "
                f"{info['F_1hop']*100:10.4f}% | {info['F_4hops']*100:12.4f}% | "
                f"{info['F_8hops']*100:12.4f}% | {info['err_4hops']*100:9.4f}%"
            )
        print("=" * 108)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.3))

        slice_xz = zg_tube.TR[:, mid, :]
        extent = [zg_tube.z[0], zg_tube.z[-1], zg_tube.x[0], zg_tube.x[-1]]
        im0 = axes[0].imshow(slice_xz, extent=extent, origin="lower", cmap="YlGnBu_r")

        for sign_x in (-1.0, 1.0):
            axes[0].axhline(
                sign_x * R_in, color="#d95f02", lw=1.6, ls="--",
                label="Casimir Binnen-/Buitenwand" if sign_x > 0 else None
            )
            axes[0].axhline(sign_x * R_out, color="#d95f02", lw=1.6, ls="--")

        for k_idx, z_k in enumerate(z_rings):
            axes[0].plot(
                [z_k, z_k], [-R_sq, R_sq], color="gold", lw=2.0, alpha=0.85,
                label="4P Josephson-Ring (Gedeeld vlak)" if k_idx == 0 else None
            )
            axes[0].scatter(
                [z_k, z_k], [-R_sq, R_sq], color="gold", edgecolor="black", s=55, zorder=5
            )
            if k_idx < N_rings - 1:
                z_mid_cube = 0.5 * (z_k + z_rings[k_idx + 1])
                axes[0].text(
                    z_mid_cube, 0.0, f"8Q\n#{k_idx+1}", ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a365d", alpha=0.75, edgecolor="gold")
                )

        axes[0].set_xlim(-42.0, 42.0)
        axes[0].set_ylim(-35.0, 35.0)
        axes[0].set_title("Dubbelwandige Casimir-Buis + 4-Kubus Josephson-Keten")
        axes[0].set_xlabel("z-as langs de Buis [ModelUnits]")
        axes[0].set_ylabel("Radiale x-as [ModelUnits]")
        axes[0].legend(loc="upper right", fontsize=7.8)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)

        r_coords = zg_tube.x[mid:]
        r_diag_coords = r_coords * np.sqrt(2.0)
        tr_bare_casimir = zg_tube.TR[mid:, mid, 0]
        zg_jos_dip = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
        zg_jos_dip.relax(
            external_pressure=(casimir_tube_pressure - 45.0 * josephson_chain_density),
            iterations=RELAX_ITERATIONS
        )
        tr_diag_jos = np.array([zg_jos_dip.TR[i, i, mid] for i in range(mid, GRID_SHAPE_ELECTRON[0])])

        axes[1].plot(r_coords, tr_bare_casimir, color="#2b5c8f", lw=2.4, ls="--", label="Puur Dubbelwandige Casimir-Buis TR(r)")
        axes[1].plot(r_diag_coords, tr_diag_jos, color="#1b9e77", lw=2.8, label="Door 4P Josephson-Knoop Verdiept TR(r)")
        axes[1].axvspan(0.0, R_in, color="#e0f3f8", alpha=0.45, label="1. Holle Kern (dTR/dr=0 op as)")
        axes[1].axvspan(R_in, R_out, color="#fee090", alpha=0.45, label="2. Coaxiale Casimir-Spouw (4P Ringen)")
        axes[1].axvspan(R_out, 26.0, color="#d9ef8b", alpha=0.35, label="3. Evanescente Buitenhuid (TR < 1)")
        axes[1].axvline(R_in, color="#d95f02", lw=1.5, ls=":")
        axes[1].axvline(R_out, color="#d95f02", lw=1.5, ls=":")
        axes[1].axhline(1.0, color="black", lw=0.8, ls=":")
        axes[1].set_xlim(0.0, 30.0)
        axes[1].set_ylim(0.78, 1.02)
        axes[1].set_title("Radiaal TR(r) Profiel: Kern, Spouw & Buitenhuid")
        axes[1].set_xlabel("Radiale afstand r vanaf Buis-as [ModelUnits]")
        axes[1].set_ylabel("Lokaal Tijdritme TR(r)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=7.5)

        for label, _, _, _, _ in bus_architectures:
            info = transport_results[label]
            axes[2].plot(
                hops_arr, info["err_chain_N"] * 100.0,
                color=info["color"], ls=info["ls"], marker=info["marker"], markersize=6.5, lw=2.4,
                label=f"{label} (F₈={info['F_8hops']*100:.2f}%)"
            )
        axes[2].set_yscale("log")
        axes[2].axhline(1.0, color="#d95f02", lw=1.1, ls=":", alpha=0.8, label="1.0% Surface-Code Drempel (F=99%)")
        axes[2].axhline(0.1, color="#2b5c8f", lw=1.1, ls="--", alpha=0.8, label="0.1% Topologische Bus-Drempel (F=99.9%)")
        axes[2].axhline(0.01, color="#1b9e77", lw=1.1, ls="-.", alpha=0.8, label="0.01% MegaQuOp Bus-Drempel (F=99.99%)")
        axes[2].set_title("Transport-Foutkans (1 - F) over N Geschakelde Kubussen")
        axes[2].set_xlabel("Aantal Doorlopen 8-Punts Kubussen in de Buis (N_hops)")
        axes[2].set_ylabel("Cumulatieve Transport-Foutkans (1 - F) [%]")
        axes[2].set_xticks(hops_arr)
        axes[2].set_ylim(5e-4, 100.0)
        axes[2].grid(True, which="both", alpha=0.3)
        axes[2].legend(loc="center right", fontsize=7.2)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.25)
        plt.savefig("qubit_casimir_josephson_tube.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_casimir_josephson_tube.png'.")

    return {
        "tr_core_center": tr_core_center,
        "tr_annulus_min": tr_annulus_min,
        "tr_outer_skin": tr_outer_skin,
        "tr_effective_chain": tr_effective_chain,
        "dip_z_chain": dip_z_chain,
        "quad_xy_chain": quad_xy_chain,
        "oct_xyz_chain": oct_xyz_chain,
        "tau_hop": tau_hop,
        "T2_8Q_casimir_tube": T2_8Q_casimir_tube,
        "gain_casimir_tube": gain_casimir_tube,
        "transport_results": transport_results,
        "z_grid": zg_tube,
    }


if __name__ == "__main__":
    run_casimir_josephson_tube_test()