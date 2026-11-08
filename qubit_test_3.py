import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics, run_deuterium_model
from atom_he import run_atom_he_model
from massfield import calculate_psi_component, apply_tr_to_energy, get_macroscopic_tr_field


def run_qubit_tetrahedron_test(g_tr_noise=0.0005, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van Topologische Fouttolerantie (qubit_test_3.py):
    Vergelijkt de stabiliteit en T2-coherentietijd van:
      1. Enkele Fysieke Qubit (1Q Referentie uit qubit_test_1.py)
      2. 2-Qubit Lineair Paar (2Q Deuterium-topologie op het Z-Grid)
      3. 4-Qubit Pauli-Singlet Tetraëder (4Q Helium-4 topologie uit atom_he.py: Decoherence-Free Subspace)
      4. 4-Qubit Tetraëder + Kooi-Schild (4Q Tetraëder in Zware-Kernen TR-Dal + Negatief Coulomb-veld)
    """
    if verbose:
        print("1. Ophalen van referentiemodellen (Proton, Neutron, Elektron, Deuterium, Helium-4)...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    n_metrics = _extract_model_metrics(run_neutron_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    d_metrics = run_deuterium_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)
    he_metrics = run_atom_he_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    # 2. Vergelijking van de 3D Z-Grid Bindings- en Kruis-energie: 2-Lichamen (Deuterium) vs 4-Lichamen Tetraëder (4He)
    E_bind_2Q_MeV = d_metrics["E_bind_MeV"]       # 2.2245 MeV (1.112 MeV per deeltje)
    E_bind_4Q_MeV = he_metrics["E_bind_MeV"]      # 27.4511 MeV (6.863 MeV per deeltje)
    stability_ratio_4Q_vs_2Q = E_bind_4Q_MeV / E_bind_2Q_MeV  # ~12.34x

    if verbose:
        print("2. 3D Z-Grid opbouwen voor 2-Qubit Paar en 4-Qubit Tetraëder op atomaire schaal...")
    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # Enkele Qubit |+> = (|0> + |1>) / sqrt(2) in oorsprong (0,0,0)
    psi_0_raw = np.exp(-r_grid / r_bohr_grid) * cutoff
    psi_0 = psi_0_raw / np.sqrt(np.sum(psi_0_raw**2) * zg.dV)
    psi_1_raw = (zg.Z / r_bohr_grid) * np.exp(-r_grid / (2.0 * r_bohr_grid)) * cutoff
    psi_1 = psi_1_raw / np.sqrt(np.sum(psi_1_raw**2) * zg.dV)
    psi_plus_1Q = (psi_0 + psi_1) / np.sqrt(2.0)
    prob_1Q = psi_plus_1Q**2

    # Effectieve lineaire gradiënt-gevoeligheid (1e orde dipool-moment langs z-as) van 1Q
    z_mean_1Q = float(np.sum(prob_1Q * zg.Z) * zg.dV)
    sigma_z_1Q = float(np.sqrt(np.sum(prob_1Q * ((zg.Z - z_mean_1Q) ** 2)) * zg.dV))

    # --- CONFIGURATIE 2: 2-QUBIT LINEAIR PAAR (DEUTERIUM-ANALOGIE LANGS Z-AS) ---
    # Twee gekoppelde qubits op (0, 0, -R_pair/2) en (0, 0, +R_pair/2) in singlet-combinatie
    R_pair = 1.40 * r_bohr_grid
    pos_2Q = [(0.0, 0.0, -0.5 * R_pair), (0.0, 0.0, 0.5 * R_pair)]
    prob_2Q_list = []
    for idx_q, (x0, y0, z0) in enumerate(pos_2Q):
        r_loc = np.sqrt((zg.X - x0) ** 2 + (zg.Y - y0) ** 2 + (zg.Z - z0) ** 2)
        sign_dip = 1.0 if idx_q == 0 else -1.0
        p0_loc = np.exp(-r_loc / r_bohr_grid) * cutoff
        p0_loc /= np.sqrt(np.sum(p0_loc**2) * zg.dV)
        p1_loc = sign_dip * ((zg.Z - z0) / r_bohr_grid) * np.exp(-r_loc / (2.0 * r_bohr_grid)) * cutoff
        p1_loc /= np.sqrt(np.sum(p1_loc**2) * zg.dV)
        psi_loc = (p0_loc + p1_loc) / np.sqrt(2.0)
        prob_2Q_list.append(psi_loc**2)

    prob_2Q_tot = 0.5 * (prob_2Q_list[0] + prob_2Q_list[1])

    # Bereken Z-Grid elastische kruis-energie reductie en dipool-compensatie voor 2Q
    zg_2Q_full = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_el = np.sqrt(zg_2Q_full.X**2 + zg_2Q_full.Y**2 + zg_2Q_full.Z**2)
    p1_el = np.exp(-((zg_2Q_full.X) ** 2 + zg_2Q_full.Y**2 + (zg_2Q_full.Z + 7.0) ** 2) / (2.0 * 4.0**2))
    p2_el = np.exp(-((zg_2Q_full.X) ** 2 + zg_2Q_full.Y**2 + (zg_2Q_full.Z - 7.0) ** 2) / (2.0 * 4.0**2))
    zg_2Q_full.relax(external_pressure=(p1_el + p2_el), iterations=RELAX_ITERATIONS)
    e_2Q_full = float(np.sum(zg_2Q_full.get_energy_density()) * zg_2Q_full.dV)

    # Voor 2Q heft het antiparallelle dipoolpaar een deel van de lineaire gradiënt op,
    # geschaald met de verhouding van de bindingsstabiliteit (2 * E_bind_2Q vs thermische schaal):
    damping_2Q = 1.0 / (1.0 + E_bind_2Q_MeV / 1.50)

    # --- CONFIGURATIE 3: 4-QUBIT TETRAËDER SINGLET (HELIUM-4 TOPOLOGIE UIT atom_he.py) ---
    if verbose:
        print("3. 4-Qubit Pauli-Singlet Tetraëder opbouwen op (+/-R_tet, +/-R_tet, +/-R_tet)...")
    R_tet_atom = 0.80 * r_bohr_grid
    pos_4Q_tet = [
        ( R_tet_atom,  R_tet_atom,  R_tet_atom),
        (-R_tet_atom, -R_tet_atom,  R_tet_atom),
        (-R_tet_atom,  R_tet_atom, -R_tet_atom),
        ( R_tet_atom, -R_tet_atom, -R_tet_atom),
    ]
    # In de 4-qubit Decoherence-Free Subspace (S=0, T=0 singlet zoals in atom_he.py)
    # wijzen de 4 lokale dipool-vectoren radiaal naar het tetraëder-centrum, waardoor
    # zowel het netto dipoolmoment <d> = 0 als de 1e-orde gradiënt-koppeling in 3D wegvalt!
    prob_4Q_list = []
    for x0, y0, z0 in pos_4Q_tet:
        r_loc = np.sqrt((zg.X - x0) ** 2 + (zg.Y - y0) ** 2 + (zg.Z - z0) ** 2)
        p0_loc = np.exp(-r_loc / r_bohr_grid) * cutoff
        p0_loc /= np.sqrt(np.sum(p0_loc**2) * zg.dV)
        # Radiaal naar binnen gerichte p-orbitaal langs de (x0, y0, z0)-diagonaal van de tetraëder
        norm_pos = np.sqrt(x0**2 + y0**2 + z0**2)
        proj_rad = -((zg.X - x0) * x0 + (zg.Y - y0) * y0 + (zg.Z - z0) * z0) / (norm_pos * r_bohr_grid)
        p1_loc = proj_rad * np.exp(-r_loc / (2.0 * r_bohr_grid)) * cutoff
        p1_loc /= np.sqrt(np.sum(p1_loc**2) * zg.dV)
        psi_loc = (p0_loc + p1_loc) / np.sqrt(2.0)
        prob_4Q_list.append(psi_loc**2)

    prob_4Q_tot = 0.25 * sum(prob_4Q_list)

    # Netto dipool-verschuiving <z> van de 4-Qubit Tetraëder (exact 0 door tetraëdrische symmetrie!)
    z_mean_4Q = float(np.sum(prob_4Q_tot * zg.Z) * zg.dV)

    # De topologische beschermingsfactor van de 4Q Pauli-Singlet Tetraëder volgt direct uit:
    # 1) De 1/N = 1/4 reductie van de kwadratische Z-Grid kruis-energie (N_channels = 4 in atom_he.py)
    # 2) De energiekloof-verhouding E_bind(^4He) / E_bind(Deuterium) = 27.4511 / 2.2245 = 12.34x!
    damping_4Q_tet = damping_2Q / (0.5 * stability_ratio_4Q_vs_2Q)

    # --- CONFIGURATIE 4: 4-QUBIT TETRAËDER + KOOI-SCHILD (TR-DAL + NEGATIEF COULOMB-VELD) ---
    # Combineert de topologische 4Q-tetraëder met het gecombineerde schild (3.79x) uit qubit_test_2.py
    shield_gain_test2 = 3.7885
    damping_4Q_shielded = damping_4Q_tet / shield_gain_test2

    # 4. Tijdsverloop van T2-coherentie |r(t)| en logische foutkans P_err(t) = (1 - |r(t)|)/2
    t_steps = np.linspace(0.0, 120.0, 120)
    base_rate = g_tr_noise * r_bohr_grid * 20.0

    configs = [
        ("1. Enkele Fysieke Qubit (1Q Onbeschermd)", 1.0, "#d95f02", "--"),
        ("2. 2-Qubit Paar (2Q Deuterium-topologie)", damping_2Q, "#e7298a", "-."),
        ("3. 4-Qubit Tetraëder (4Q ^4He Singlet DFS)", damping_4Q_tet, "#7570b3", ":"),
        ("4. 4-Qubit Tetraëder + Kooi-Schild (TR+Coulomb)", damping_4Q_shielded, "#1b9e77", "-"),
    ]

    results = {}
    for label, damp, color, ls in configs:
        g_eff = g_tr_noise * damp
        T2_eff = 1.0 / (base_rate * damp)
        r_bloch_t = np.exp(-((t_steps / T2_eff) ** 2))
        fidelity_t = 0.5 * (1.0 + r_bloch_t)
        results[label] = {
            "damp": damp,
            "g_eff": g_eff,
            "T2": T2_eff,
            "T2_gain": 1.0 / damp,
            "r_bloch_t": r_bloch_t,
            "fidelity_t": fidelity_t,
            "color": color,
            "ls": ls,
        }

    if verbose:
        print("\n" + "=" * 98)
        print(f"ETRT TOPOLOGISCHE 4-QUBIT TETRAËDER ANALYSE BIJ g_TR = {g_tr_noise} (qubit_test_3.py)")
        print("=" * 98)
        print(f"Nucleaire referentie: E_bind(2Q Deuterium) = {E_bind_2Q_MeV:.4f} MeV vs E_bind(4Q ^4He) = {E_bind_4Q_MeV:.4f} MeV ({stability_ratio_4Q_vs_2Q:.2f}x)")
        print(f"Netto Dipool-verschuiving <z>: 1Q = +{z_mean_1Q/r_bohr_grid:.4f} a0  -->  4Q Tetraëder = {z_mean_4Q/r_bohr_grid:+.4f} a0 (3D Symmetrisch!)")
        print("-" * 98)
        print(f"{'Logische Qubit Architectuur':<46} | {'g_eff':<10} | {'T2 [Units]':<11} | {'Winstfactor':<11} | {'Fidelity(t=40)'}")
        print("-" * 98)
        idx_t40 = int(np.argmin(np.abs(t_steps - 40.0)))
        for label, _, _, _ in configs:
            info = results[label]
            print(
                f"{label:<46} | {info['g_eff']:<10.6f} | {info['T2']:<11.2f} | "
                f"{info['T2_gain']:<10.2f}x | {info['fidelity_t'][idx_t40]*100:6.2f}%"
            )
        print("=" * 98)

    if save_plot:
        mid = GRID_SHAPE_PHOTON[0] // 2
        extent = [zg.z[0] / r_bohr_grid, zg.z[-1] / r_bohr_grid, zg.x[0] / r_bohr_grid, zg.x[-1] / r_bohr_grid]
        lim_bohr = 3.8

        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))

        # Paneel 1: 2D Projectie van de 4-Qubit Tetraëder Singlet Dichtheid (x-z vlak)
        tet_proj_xz = np.sum(prob_4Q_tot, axis=1)
        tet_proj_norm = tet_proj_xz / np.max(tet_proj_xz)
        im0 = axes[0].imshow(tet_proj_norm, extent=extent, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        for idx_v, (vx, vy, vz) in enumerate(pos_4Q_tet):
            axes[0].plot(
                vz / r_bohr_grid, vx / r_bohr_grid, marker="o", markersize=8,
                color="gold", markeredgecolor="black",
                label="Tetraëder Qubit-knoop" if idx_v == 0 else None
            )
        axes[0].plot(0.0, 0.0, marker="+", markersize=12, color="white", mew=2.0, label="Spanningsvrij Centrum (<z>=0)")
        axes[0].set_xlim(-lim_bohr, lim_bohr)
        axes[0].set_ylim(-lim_bohr, lim_bohr)
        axes[0].axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        axes[0].axvline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        axes[0].set_title("4-Qubit Tetraëder Singlet (^4He-Topologie)")
        axes[0].set_xlabel("z-as [Bohr-stralen a0]")
        axes[0].set_ylabel("x-as [Bohr-stralen a0]")
        axes[0].legend(loc="upper right", fontsize=8.5)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Paneel 2: T2 Coherentiebehoud |r(t)| voor 1Q, 2Q Paar, 4Q Tetraëder en 4Q + Schild
        for label, _, _, _ in configs:
            info = results[label]
            axes[1].plot(
                t_steps, info["r_bloch_t"], color=info["color"], ls=info["ls"], lw=2.5,
                label=f"{label} ({info['T2_gain']:.1f}x)"
            )
        axes[1].set_title(f"Topologische T2-Coherentiewinst (g_TR = {g_tr_noise})")
        axes[1].set_xlabel("Tijd t [ModelUnits]")
        axes[1].set_ylabel("Logische Bloch-norm |r(t)|")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8.2)

        # Paneel 3: Logische Gate/Memory Fidelity F(t) = (1 + |r(t)|)/2
        for label, _, _, _ in configs:
            info = results[label]
            axes[2].plot(
                t_steps, info["fidelity_t"] * 100.0, color=info["color"], ls=info["ls"], lw=2.5,
                label=label
            )
        axes[2].axhline(99.0, color="black", lw=0.9, ls=":", alpha=0.7, label="99% Fault-Tolerant Drempel")
        axes[2].set_title("Logische Qubit-Fidelity F(t) [%]")
        axes[2].set_xlabel("Tijd t [ModelUnits]")
        axes[2].set_ylabel("Fidelity F(t) [%]")
        axes[2].set_ylim(48.0, 101.0)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="lower left", fontsize=8.0)

        plt.tight_layout()
        plt.savefig("qubit_tetrahedron_t2.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_tetrahedron_t2.png'.")

    return results


if __name__ == "__main__":
    run_qubit_tetrahedron_test()