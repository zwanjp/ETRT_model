import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics, run_deuterium_model
from atom_he import run_atom_he_model
from planck import run_planck_model
from qubit_test_4 import _compute_8node_cube_binding


def run_qubit_topological_gate_test(g_tr_noise=0.0005, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van een Topologische Kwantumpoort in de 8-Punts Kubus (qubit_test_5.py):
    Modelleert hoe de twee in elkaar grijpende 4-punts tetraëders binnen de 8-punts kubus:
      - Tetraëder T_+ (x*y*z > 0): Logische Control-Qubit |Q_C> (Tetron A)
      - Tetraëder T_- (x*y*z < 0): Logische Target-Qubit  |Q_T> (Tetron B)
    via de 12 kubusribben (d_edge = 2*R_tet) een topologische Pariteits-Wissel (X_L),
    Hadamard (H_L) en CNOT-poort uitvoeren, terwijl de multipoolmomenten (l=1, 2, 3)
    gedurende de hele poortoperatie exact nul blijven.
    """
    if verbose:
        print("1. Ophalen van referentiemodellen en initialiseren van het 3D Z-Grid...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    n_metrics = _extract_model_metrics(run_neutron_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)
    d_metrics = run_deuterium_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)
    he_metrics = run_atom_he_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)

    E_bind_2Q_MeV = d_metrics["E_bind_MeV"]
    E_bind_4Q_MeV = he_metrics["E_bind_MeV"]
    E_bind_8Q_MeV = _compute_8node_cube_binding(p_metrics, n_metrics, e_metrics)

    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor
    hbar_model = pl_metrics["hbar_model"]

    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # 2. Geometrie van de 8-punts Kubus: T_+ (Control-Tetron) en T_- (Target-Tetron)
    R_tet = 0.80 * r_bohr_grid
    d_edge_grid = 2.0 * R_tet                  # Ribbe-afstand tussen T_+ en T_- (1.60 a0)
    d_diag_tet = 2.0 * np.sqrt(2.0) * R_tet    # Interne ribbe binnen één tetraëder (2.26 a0)

    pos_T_plus = [
        ( R_tet,  R_tet,  R_tet),
        (-R_tet, -R_tet,  R_tet),
        (-R_tet,  R_tet, -R_tet),
        ( R_tet, -R_tet, -R_tet),
    ]
    pos_T_minus = [
        (-R_tet, -R_tet, -R_tet),
        ( R_tet,  R_tet, -R_tet),
        ( R_tet, -R_tet,  R_tet),
        (-R_tet,  R_tet,  R_tet),
    ]

    if verbose:
        print("2. 3D Knoop-golffuncties en kubusribbe-tunnelkoppeling S_edge tussen T_+ en T_- berekenen...")

    def get_node_orbital(pos):
        x0, y0, z0 = pos
        r_loc = np.sqrt((zg.X - x0) ** 2 + (zg.Y - y0) ** 2 + (zg.Z - z0) ** 2)
        p0_loc = np.exp(-r_loc / r_bohr_grid) * cutoff
        p0_loc /= np.sqrt(np.sum(p0_loc**2) * zg.dV)
        norm_pos = np.sqrt(x0**2 + y0**2 + z0**2)
        proj_rad = -((zg.X - x0) * x0 + (zg.Y - y0) * y0 + (zg.Z - z0) * z0) / (norm_pos * r_bohr_grid)
        p1_loc = proj_rad * np.exp(-r_loc / (2.0 * r_bohr_grid)) * cutoff
        p1_loc /= np.sqrt(np.sum(p1_loc**2) * zg.dV)
        return p0_loc, p1_loc

    orbitals_plus = [get_node_orbital(p) for p in pos_T_plus]
    orbitals_minus = [get_node_orbital(p) for p in pos_T_minus]

    # 3D Overlap-integraal S_edge langs een kubusribbe (tussen (R,R,R) in T_+ en (R,R,-R) in T_-)
    # en S_tet langs een vlakdiagonaal binnen dezelfde tetraëder (tussen (R,R,R) en (-R,-R,R))
    psi_node_p0 = orbitals_plus[0][0]
    psi_edge_neighbor = orbitals_minus[1][0]   # (R, R, -R), afstand = 2 * R_tet = 1.60 a0
    psi_tet_neighbor = orbitals_plus[1][0]     # (-R, -R, R), afstand = 2*sqrt(2)*R_tet = 2.26 a0

    S_edge_3d = float(np.sum(psi_node_p0 * psi_edge_neighbor) * zg.dV)
    S_tet_3d = float(np.sum(psi_node_p0 * psi_tet_neighbor) * zg.dV)

    # Elastische kruis-koppeling op het Z-Grid tussen T_+ en T_-
    # De inter-tetraëdrische bindingswinst is E_inter = E_bind(8Q) - 2 * E_bind(4Q)
    E_inter_tet_MeV = E_bind_8Q_MeV - 2.0 * E_bind_4Q_MeV
    # Adiabatische topologische poort-koppelingssterkte J_gate (over 12 kubusribben):
    J_gate_model = (S_edge_3d / hbar_model) * (E_inter_tet_MeV / E_bind_8Q_MeV) / 3.0
    tau_cnot = np.pi / (2.0 * J_gate_model)   # Poortduur voor een volledige CNOT-inversie (~10.5 ModelUnits)
    tau_hadamard = 0.5 * tau_cnot             # Poortduur voor een sqrt(NOT) / Hadamard pariteits-split

    # 3. Opbouw van de Logische Pariteitstoestanden |0_L> en |1_L> voor T_+ en T_-
    # In de A1-singlet ademhalingsmodus van elke tetraëder wijzen alle 4 radiale lobben
    # synchroon naar binnen (|0_L>, pariteit +) of naar buiten (|1_L>, pariteit -),
    # waardoor de tetraëdrische symmetrie tijdens de hele rotatie theta_bloch behouden blijft!
    def get_tetron_density(orbitals, theta_bloch):
        c0 = np.cos(0.5 * theta_bloch)
        c1 = np.sin(0.5 * theta_bloch)
        dens_sum = np.zeros_like(zg.X)
        for p0, p1 in orbitals:
            psi_node = (c0 * p0 + c1 * p1)
            dens_sum += psi_node**2
        return 0.25 * dens_sum

    def measure_multipoles(prob_density):
        z_n = zg.Z / r_bohr_grid
        x_n = zg.X / r_bohr_grid
        y_n = zg.Y / r_bohr_grid
        r_sq_n = x_n**2 + y_n**2 + z_n**2
        dip_z = float(np.sum(prob_density * z_n) * zg.dV)
        quad_zz = float(np.sum(prob_density * (3.0 * (z_n**2) - r_sq_n)) * zg.dV)
        oct_xyz = float(np.sum(prob_density * (x_n * y_n * z_n)) * zg.dV)
        return dip_z, quad_zz, oct_xyz

    # 4. Tijdsafhankelijke Simulatie van de Topologische CNOT-Poort (en Bell-State creatie):
    if verbose:
        print("3. Tijdsverloop van de Topologische CNOT-poort (T_+ -> T_-) en multipool-invariantie simuleren...")

    n_steps = 100
    t_gate = np.linspace(0.0, tau_cnot, n_steps)

    # Decoherentie-schaalfactor uit qubit_test_4.py:
    base_rate = g_tr_noise * r_bohr_grid * 20.0
    damping_2Q = 1.0 / (1.0 + E_bind_2Q_MeV / 1.50)
    damping_8Q = damping_2Q / (0.5 * (E_bind_8Q_MeV / E_bind_2Q_MeV))
    shield_gain = 3.7885
    damping_8Q_shielded = damping_8Q / shield_gain

    T2_1Q = 1.0 / base_rate
    T2_8Q = 1.0 / (base_rate * damping_8Q)
    T2_8Q_shielded = 1.0 / (base_rate * damping_8Q_shielded)

    # Rabi-rotatiehoek van Target T_- wanneer Control T_+ = |1_L>: theta_T(t) = pi * (t / tau_cnot)
    theta_target_t = np.pi * (t_gate / tau_cnot)

    # Coherentie-enveloppen tijdens de poortoperatie
    env_8Q_shield = np.exp(-((t_gate / T2_8Q_shielded) ** 2))
    env_8Q_unshield = np.exp(-((t_gate / T2_8Q) ** 2))
    env_1Q_unshield = np.exp(-((t_gate / T2_1Q) ** 2))

    P1_target_ctrl0 = np.zeros_like(t_gate)  # Control |0_L>: Ribbe-brug inactief -> Target blijft exact |0_L>
    P1_target_ctrl1_8Q_shield = 0.5 * (1.0 - env_8Q_shield * np.cos(theta_target_t))
    P1_target_ctrl1_8Q = 0.5 * (1.0 - env_8Q_unshield * np.cos(theta_target_t))
    P1_target_ctrl1_1Q = 0.5 * (1.0 - env_1Q_unshield * np.cos(theta_target_t))

    # Opbouw van verstrengeling wanneer Control T_+ = |+_L> = (|0_L> + |1_L>)/sqrt(2):
    # De Concurrence C(t) = sin(theta_T(t) / 2) * env(t) groeit van 0 op t=0 naar 1.0 op t=tau_CNOT!
    concurrence_8Q_shield = np.sin(0.5 * theta_target_t) * env_8Q_shield
    concurrence_8Q = np.sin(0.5 * theta_target_t) * env_8Q_unshield
    concurrence_1Q = np.sin(0.5 * theta_target_t) * env_1Q_unshield

    S_chsh_8Q_shield = 2.0 * np.sqrt(1.0 + concurrence_8Q_shield**2) * hbar_model
    S_chsh_8Q = 2.0 * np.sqrt(1.0 + concurrence_8Q**2) * hbar_model
    S_chsh_1Q = 2.0 * np.sqrt(1.0 + concurrence_1Q**2) * hbar_model

    # Controleer multipool-invariantie van de symmetrisch gekoppelde 8-punts kubus op t = 0, tau/2 en tau
    prob_cube_start = 0.5 * (get_tetron_density(orbitals_plus, 0.0) + get_tetron_density(orbitals_minus, 0.0))
    prob_cube_mid = 0.5 * (get_tetron_density(orbitals_plus, np.pi / 2.0) + get_tetron_density(orbitals_minus, np.pi / 2.0))
    prob_cube_end = 0.5 * (get_tetron_density(orbitals_plus, np.pi) + get_tetron_density(orbitals_minus, np.pi))

    mp_start = measure_multipoles(prob_cube_start)
    mp_mid = measure_multipoles(prob_cube_mid)
    mp_end = measure_multipoles(prob_cube_end)

    # CNOT Gate Fidelity F_CNOT op t = tau_cnot
    F_cnot_1Q = float(0.5 * (1.0 + env_1Q_unshield[-1]))
    F_cnot_8Q = float(0.5 * (1.0 + env_8Q_unshield[-1]))
    F_cnot_8Q_shield = float(0.5 * (1.0 + env_8Q_shield[-1]))

    # Waarheidstabel van de Topologische CNOT-poort op t = tau_cnot (met 8Q + Kooi-Schild)
    f_gate = F_cnot_8Q_shield
    truth_table = [
        ("|0_L, 0_L>", "|0_L, 0_L>", f_gate, 1.0 - f_gate, 0.0, 0.0),
        ("|0_L, 1_L>", "|0_L, 1_L>", 1.0 - f_gate, f_gate, 0.0, 0.0),
        ("|1_L, 0_L>", "|1_L, 1_L>", 0.0, 0.0, 1.0 - f_gate, f_gate),
        ("|1_L, 1_L>", "|1_L, 0_L>", 0.0, 0.0, f_gate, 1.0 - f_gate),
    ]

    if verbose:
        print("\n" + "=" * 106)
        print("ETRT TOPOLOGISCHE KWANTUMPOORT IN DE 8-PUNTS KUBUS: CNOT & PARITEIT-WISSEL (qubit_test_5.py)")
        print("=" * 106)
        print(f"Kubusribbe-afstand T_+ <-> T_- (d_edge):     {d_edge_grid/r_bohr_grid:.2f} a0 vs Interne Tetraëder-ribbe: {d_diag_tet/r_bohr_grid:.2f} a0")
        print(f"3D Z-Grid Tunnel-Overlap (S_edge vs S_tet):  S_edge = {S_edge_3d:.4f} (12 kubusribben) vs S_tet = {S_tet_3d:.4f}")
        print(f"Inter-Tetraëdrische Koppelingsenergie:       ΔE_inter = {E_inter_tet_MeV:.2f} MeV (van {E_bind_8Q_MeV:.2f} MeV totaal)")
        print(f"Topologische CNOT-Poortduur (tau_CNOT):      {tau_cnot:.2f} [ModelUnits] (Hadamard tau_H = {tau_hadamard:.2f} [ModelUnits])")
        print("-" * 106)
        print("MULTIPOOL-INVARIANTIE VAN DE 8-PUNTS KUBUS TIJDENS DE POORTOPERATIE (Dipool, Quadrupool, Octupool):")
        print(f"  t = 0.00 * tau_CNOT (Start |0_L, 0_L>):    Dipool <z> = {mp_start[0]:+.4f} | Quad Q_zz = {mp_start[1]:+.4f} | Oct <xyz> = {mp_start[2]:+.4f}")
        print(f"  t = 0.50 * tau_CNOT (Halverwege Braiding): Dipool <z> = {mp_mid[0]:+.4f} | Quad Q_zz = {mp_mid[1]:+.4f} | Oct <xyz> = {mp_mid[2]:+.4f}")
        print(f"  t = 1.00 * tau_CNOT (Einde |Phi^+_Cube>):  Dipool <z> = {mp_end[0]:+.4f} | Quad Q_zz = {mp_end[1]:+.4f} | Oct <xyz> = {mp_end[2]:+.4f}")
        print("-" * 106)
        print(f"{'Architectuur tijdens CNOT-Poort (g_TR = 0.0005)':<48} | {'P(|1_T>) eind':<14} | {'Bell S_CHSH':<13} | {'CNOT Fidelity'}")
        print("-" * 106)
        print(f"{'1. Conventionele 1Q/2Q Poort (Onbeschermd)':<48} | {P1_target_ctrl1_1Q[-1]:<14.4f} | {S_chsh_1Q[-1]:<13.4f} | {F_cnot_1Q*100:6.2f}%")
        print(f"{'2. 8-Qubit Kubus CNOT (T_+ -> T_-, Onbeschermd)':<48} | {P1_target_ctrl1_8Q[-1]:<14.4f} | {S_chsh_8Q[-1]:<13.4f} | {F_cnot_8Q*100:6.3f}%")
        print(f"{'3. 8-Qubit Kubus CNOT + Kooi-Schild (TR+Coulomb)':<48} | {P1_target_ctrl1_8Q_shield[-1]:<14.4f} | {S_chsh_8Q_shield[-1]:<13.4f} | {F_cnot_8Q_shield*100:6.4f}%")
        print("-" * 106)
        print("WAARHEIDSTABEL TOPOLOGISCHE CNOT-POORT (T_+ Control -> T_- Target op het 3D Z-Grid):")
        for in_st, out_st, p00, p01, p10, p11 in truth_table:
            print(f"  In: {in_st}  -->  Uit: {out_st}   [ P(00)={p00:.5f}, P(01)={p01:.5f}, P(10)={p10:.5f}, P(11)={p11:.5f} ]")
        print("=" * 106)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))

        # Paneel 1: Topologische CNOT Rabi-Inversie van Target T_- (Control |1_L> vs Control |0_L>)
        axes[0].plot(
            t_gate, P1_target_ctrl1_8Q_shield, color="#1b9e77", lw=2.6,
            label=f"Control T+ = |1_L> (8Q+Schild, F={F_cnot_8Q_shield*100:.3f}%)"
        )
        axes[0].plot(
            t_gate, P1_target_ctrl1_8Q, color="#2b5c8f", lw=2.2, ls="--",
            label=f"Control T+ = |1_L> (8Q Kubus, F={F_cnot_8Q*100:.2f}%)"
        )
        axes[0].plot(
            t_gate, P1_target_ctrl1_1Q, color="#d95f02", lw=2.2, ls=":",
            label=f"Control = |1> (Conventioneel, F={F_cnot_1Q*100:.1f}%)"
        )
        axes[0].plot(
            t_gate, P1_target_ctrl0, color="black", lw=1.8, ls="-.",
            label="Control T+ = |0_L> (Target blijft |0_L>)"
        )
        axes[0].axvline(tau_hadamard, color="gray", lw=0.9, ls="--", alpha=0.6, label="tau_H (sqrt(NOT) / Hadamard)")
        axes[0].set_title("Topologische CNOT-Schakeling (T+ Control -> T- Target)")
        axes[0].set_xlabel("Poorttijd t [ModelUnits]")
        axes[0].set_ylabel("Target T- Flip-Kans P(|1_L>)")
        axes[0].set_ylim(-0.03, 1.05)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="center right", fontsize=7.8)

        # Paneel 2: Opbouw van Inter-Tetraëdrische Verstrengeling (Bell-CHSH parameter S_CHSH(t))
        axes[1].plot(
            t_gate, S_chsh_8Q_shield, color="#1b9e77", lw=2.6,
            label=f"8Q Kubus + Kooi-Schild (Eind S = {S_chsh_8Q_shield[-1]:.4f})"
        )
        axes[1].plot(
            t_gate, S_chsh_8Q, color="#2b5c8f", lw=2.2, ls="--",
            label=f"8Q Kubus Onbeschermd (Eind S = {S_chsh_8Q[-1]:.4f})"
        )
        axes[1].plot(
            t_gate, S_chsh_1Q, color="#d95f02", lw=2.2, ls=":",
            label=f"Conventionele Poort (Eind S = {S_chsh_1Q[-1]:.4f})"
        )
        axes[1].axhline(2.0, color="red", lw=1.1, ls="--", alpha=0.7, label="Klassieke Bell-grens (S = 2.0)")
        axes[1].axhline(2.0 * np.sqrt(2.0), color="black", lw=0.9, ls=":", alpha=0.7, label="Tsirelson-maximum (2√2 = 2.828)")
        axes[1].set_title("Opbouw van Verstrengelde 8-Punts Bell-Toestand |Phi+_Cube>")
        axes[1].set_xlabel("Poorttijd t [ModelUnits]")
        axes[1].set_ylabel("Bell-CHSH Parameter S_CHSH(t)")
        axes[1].set_ylim(1.90, 2.95)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="lower right", fontsize=8.0)

        # Paneel 3: 4x4 Waarheidstabel-Matrix P(Uit | In) van de Topologische CNOT-Poort
        cnot_matrix = np.array([
            [f_gate, 1.0 - f_gate, 0.0, 0.0],
            [1.0 - f_gate, f_gate, 0.0, 0.0],
            [0.0, 0.0, 1.0 - f_gate, f_gate],
            [0.0, 0.0, f_gate, 1.0 - f_gate],
        ])
        im2 = axes[2].imshow(cnot_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        basis_labels = ["|00>_L", "|01>_L", "|10>_L", "|11>_L"]
        axes[2].set_xticks(range(4))
        axes[2].set_yticks(range(4))
        axes[2].set_xticklabels(basis_labels)
        axes[2].set_yticklabels(basis_labels)
        axes[2].set_xlabel("Uitgangstoestand (T+ Control, T- Target)")
        axes[2].set_ylabel("Ingangstoestand (T+ Control, T- Target)")
        axes[2].set_title(f"Topologische CNOT-Overgangsmatrix (F = {f_gate*100:.3f}%)")
        for i in range(4):
            for j in range(4):
                val = cnot_matrix[i, j]
                txt_color = "white" if val > 0.5 else "black"
                axes[2].text(j, i, f"{val:.4f}", ha="center", va="center", color=txt_color, fontsize=9.5, fontweight="bold")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("qubit_cube_gate_cnot.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_cube_gate_cnot.png'.")

    return {
        "S_edge_3d": S_edge_3d,
        "S_tet_3d": S_tet_3d,
        "E_inter_tet_MeV": E_inter_tet_MeV,
        "tau_cnot": tau_cnot,
        "tau_hadamard": tau_hadamard,
        "F_cnot_1Q": F_cnot_1Q,
        "F_cnot_8Q": F_cnot_8Q,
        "F_cnot_8Q_shield": F_cnot_8Q_shield,
        "S_chsh_end": float(S_chsh_8Q_shield[-1]),
    }


if __name__ == "__main__":
    run_qubit_topological_gate_test()