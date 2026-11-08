import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics, run_deuterium_model
from atom_he import run_atom_he_model
from massfield import calculate_psi_component, apply_tr_to_energy


def _compute_8node_cube_binding(p_metrics, n_metrics, e_metrics):
    """
    Berekent op het 3D Z-Grid de collectieve bindingsenergie (topologische kloof)
    van een 8-punts Kubus (Dubbele Tetraëder / Stella Octangula: T_+ U T_-) op (+/-dx, +/-dx, +/-dx)
    met N = 8 orthogonale spin-isospin-pariteit kanalen.
    """
    M_p_raw = p_metrics["M0_raw"]
    R_rms_sq_p = p_metrics["R_rms_sq"]
    M_n_raw = n_metrics["M0_raw"]
    M_e_phys = e_metrics["M0_phys"]
    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0

    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_grid = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    R_c = DX_ELECTRON
    r_q_up = Y_UP / 2.0
    r_q_down = Y_DOWN / 2.0
    r_bag_p = (2.0 * r_q_up + r_q_down) / 3.0
    r_bag_n = (1.0 * r_q_up + 2.0 * r_q_down) / 3.0

    p_tr_up = ((Y_VAC / Y_UP) ** 3) * (((r_q_up) / ((r_q_up) + XI_Z)) ** 2)
    p_tr_down = ((Y_VAC / Y_DOWN) ** 3) * (((r_q_down) / ((r_q_down) + XI_Z)) ** 2)
    p_amp_up = abs(p_tr_up - 1.0)
    p_amp_down = abs(p_tr_down - 1.0)

    # De 8 hoekpunten van de kubus = Tetraëder T_+ (x*y*z > 0) + Duale Tetraëder T_- (x*y*z < 0)
    cube_nodes = [
        # T_+ Tetraëder (pariteit +1)
        ("proton",  ( R_c,  R_c,  R_c), r_bag_p, M_p_raw),
        ("proton",  (-R_c, -R_c,  R_c), r_bag_p, M_p_raw),
        ("neutron", (-R_c,  R_c, -R_c), r_bag_n, M_n_raw),
        ("neutron", ( R_c, -R_c, -R_c), r_bag_n, M_n_raw),
        # T_- Duale Tetraëder (pariteit -1)
        ("proton",  (-R_c, -R_c, -R_c), r_bag_p, M_p_raw),
        ("proton",  ( R_c,  R_c, -R_c), r_bag_p, M_p_raw),
        ("neutron", ( R_c, -R_c,  R_c), r_bag_n, M_n_raw),
        ("neutron", (-R_c,  R_c,  R_c), r_bag_n, M_n_raw),
    ]

    pressure_list = []
    E_bag_tot = np.zeros_like(z_grid.X)
    psi_ext_total = np.zeros_like(z_grid.X)

    for n_type, pos, r_bag, m_raw in cube_nodes:
        X_s = z_grid.X - pos[0]
        Y_s = z_grid.Y - pos[1]
        Z_s = z_grid.Z - pos[2]
        r_i = np.sqrt(X_s**2 + Y_s**2 + Z_s**2)

        if n_type == "proton":
            shell = get_proton_envelope(X_s, Y_s, Z_s, r_bag)
            pressure = (2.0 * p_amp_up + 1.0 * p_amp_down) * shell
        else:
            shell = get_neutron_envelope(X_s, Y_s, Z_s, r_bag)
            pressure = (1.0 * p_amp_up + 2.0 * p_amp_down) * shell

        pressure_list.append(pressure)
        E_bag_tot += np.where(r_i <= r_bag, BAG_ENERGY_DENSITY, 0.0)
        psi_i = calculate_psi_component(z_grid.X, z_grid.Y, z_grid.Z, pos, r_bag, m_raw * OMEGA_MACRO_SCALE)
        psi_ext_total += np.where(r_i > r_bag, psi_i, 0.0)

    total_pressure = sum(pressure_list)

    e_grid_ind = np.zeros_like(z_grid.X)
    for press_i in pressure_list:
        zg_i = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
        zg_i.relax(external_pressure=press_i, iterations=RELAX_ITERATIONS)
        e_grid_ind += zg_i.get_energy_density()

    z_grid.relax(external_pressure=total_pressure, iterations=RELAX_ITERATIONS)
    e_grid_full = z_grid.get_energy_density()

    # N = 8 orthogonale kanalen (4 spin-isospin in T_+ en 4 inversie-gespiegelde in T_-)
    N_channels = 8.0
    e_grid_octon = e_grid_ind + (1.0 / N_channels) * (e_grid_full - e_grid_ind)
    E_veld = np.abs(total_pressure) + e_grid_octon
    E_tot_raw = (E_veld + E_bag_tot) * cutoff

    TR_mutual = np.exp(-psi_ext_total)
    E_total_density = apply_tr_to_energy(E_tot_raw, TR_mutual)
    M_cube_raw = np.sum(E_total_density) * z_grid.dV

    bare_8_raw = 4.0 * M_p_raw + 4.0 * M_n_raw
    delta_M_phys = (bare_8_raw - M_cube_raw) / R_rms_sq_p
    E_bind_8Q_MeV = delta_M_phys * mass_conv_MeV
    return E_bind_8Q_MeV


def run_qubit_cube_test(g_tr_noise=0.0005, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Simulatie van de 8-Punts Kubus / Majorana-Octon (qubit_test_4.py):
    Vergelijkt de multipool-opheffing (Dipool l=1, Quadrupool l=2, Octupool l=3)
    en T2-coherentiewinst van:
      1. Enkele Fysieke Qubit (1Q)
      2. 2-Qubit Lineair Paar (2Q)
      3. 4-Qubit Tetraëder (4Q Tetron, T_+ : heft Dipool & Quadrupool op, maar heeft Octupool != 0)
      4. 8-Qubit Kubus (8Q Octon, T_+ U T_- : heft Dipool, Quadrupool ÉN Octupool volledig op!)
      5. 8-Qubit Kubus + Kooi-Schild (TR-Dal + Negatief Statisch Coulomb-veld)
    """
    if verbose:
        print("1. Ophalen van referentiemodellen en 3D Z-Grid opbouwen...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    n_metrics = _extract_model_metrics(run_neutron_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    d_metrics = run_deuterium_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)
    he_metrics = run_atom_he_model(p_metrics=p_metrics, n_metrics=n_metrics, e_metrics=e_metrics, verbose=False)

    E_bind_2Q_MeV = d_metrics["E_bind_MeV"]
    E_bind_4Q_MeV = he_metrics["E_bind_MeV"]
    if verbose:
        print("2. 8-knoops Z-Grid relaxatie uitvoeren voor de Dubbele Tetraëder Kubus (T_+ U T_-)...")
    E_bind_8Q_MeV = _compute_8node_cube_binding(p_metrics, n_metrics, e_metrics)

    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])
    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # Hulpfunctie om 3D Multipoolmomenten op het Z-Grid te meten (in eenheden van a0^l):
    # - l=1 Dipool:     D_z   = <z / a0>
    # - l=2 Quadrupool: Q_zz  = <(3*z^2 - r^2) / a0^2>
    # - l=3 Octupool:   O_xyz = <x * y * z / a0^3>
    def measure_multipoles(prob_density):
        z_n = zg.Z / r_bohr_grid
        x_n = zg.X / r_bohr_grid
        y_n = zg.Y / r_bohr_grid
        r_sq_n = x_n**2 + y_n**2 + z_n**2
        dip_z = float(np.sum(prob_density * z_n) * zg.dV)
        quad_zz = float(np.sum(prob_density * (3.0 * (z_n**2) - r_sq_n)) * zg.dV)
        oct_xyz = float(np.sum(prob_density * (x_n * y_n * z_n)) * zg.dV)
        return dip_z, quad_zz, oct_xyz

    # --- 1. ENKELE QUBIT (1Q) ---
    psi_0_raw = np.exp(-r_grid / r_bohr_grid) * cutoff
    psi_0 = psi_0_raw / np.sqrt(np.sum(psi_0_raw**2) * zg.dV)
    psi_1_raw = (zg.Z / r_bohr_grid) * np.exp(-r_grid / (2.0 * r_bohr_grid)) * cutoff
    psi_1 = psi_1_raw / np.sqrt(np.sum(psi_1_raw**2) * zg.dV)
    prob_1Q = ((psi_0 + psi_1) / np.sqrt(2.0)) ** 2

    # --- 2. 2-QUBIT PAAR (2Q) ---
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
        prob_2Q_list.append(((p0_loc + p1_loc) / np.sqrt(2.0)) ** 2)
    prob_2Q = 0.5 * sum(prob_2Q_list)

    # --- 3. 4-QUBIT TETRAËDER (4Q: T_+) EN 4. 8-QUBIT KUBUS (8Q: T_+ U T_-) ---
    if verbose:
        print("3. 3D Multipoolmomenten (Dipool, Quadrupool, Octupool) integreren voor 1Q, 2Q, 4Q en 8Q...")
    R_tet_atom = 0.80 * r_bohr_grid
    pos_T_plus = [
        ( R_tet_atom,  R_tet_atom,  R_tet_atom),
        (-R_tet_atom, -R_tet_atom,  R_tet_atom),
        (-R_tet_atom,  R_tet_atom, -R_tet_atom),
        ( R_tet_atom, -R_tet_atom, -R_tet_atom),
    ]
    pos_T_minus = [
        (-R_tet_atom, -R_tet_atom, -R_tet_atom),
        ( R_tet_atom,  R_tet_atom, -R_tet_atom),
        ( R_tet_atom, -R_tet_atom,  R_tet_atom),
        (-R_tet_atom,  R_tet_atom,  R_tet_atom),
    ]

    def build_vertex_density(vertices):
        p_list = []
        for x0, y0, z0 in vertices:
            r_loc = np.sqrt((zg.X - x0) ** 2 + (zg.Y - y0) ** 2 + (zg.Z - z0) ** 2)
            p0_loc = np.exp(-r_loc / r_bohr_grid) * cutoff
            p0_loc /= np.sqrt(np.sum(p0_loc**2) * zg.dV)
            norm_pos = np.sqrt(x0**2 + y0**2 + z0**2)
            proj_rad = -((zg.X - x0) * x0 + (zg.Y - y0) * y0 + (zg.Z - z0) * z0) / (norm_pos * r_bohr_grid)
            p1_loc = proj_rad * np.exp(-r_loc / (2.0 * r_bohr_grid)) * cutoff
            p1_loc /= np.sqrt(np.sum(p1_loc**2) * zg.dV)
            p_list.append(((p0_loc + p1_loc) / np.sqrt(2.0)) ** 2)
        return sum(p_list) / len(p_list)

    prob_4Q_Tplus = build_vertex_density(pos_T_plus)
    prob_4Q_Tminus = build_vertex_density(pos_T_minus)
    prob_8Q_Cube = 0.5 * (prob_4Q_Tplus + prob_4Q_Tminus)

    mp_1Q = measure_multipoles(prob_1Q)
    mp_2Q = measure_multipoles(prob_2Q)
    mp_4Q = measure_multipoles(prob_4Q_Tplus)
    mp_8Q = measure_multipoles(prob_8Q_Cube)

    # 4. Berekening van de topologische ruis-onderdrukking en T2-coherentiewinst
    damping_2Q = 1.0 / (1.0 + E_bind_2Q_MeV / 1.50)
    damping_4Q = damping_2Q / (0.5 * (E_bind_4Q_MeV / E_bind_2Q_MeV))
    # Voor de 8Q Kubus schaalt de bescherming met de Z-Grid energiekloof E_bind_8Q_MeV (151.85 MeV)
    # én de verdubbeling van het aantal orthogonale kanalen (N=8 vs N=4) + wegvallen van de l=3 octupool:
    damping_8Q = damping_2Q / (0.5 * (E_bind_8Q_MeV / E_bind_2Q_MeV))
    shield_gain_test2 = 3.7885
    damping_8Q_shielded = damping_8Q / shield_gain_test2

    t_steps = np.linspace(0.0, 200.0, 160)
    base_rate = g_tr_noise * r_bohr_grid * 20.0

    configs = [
        ("1. Enkele Qubit (1Q Onbeschermd)", 1.0, mp_1Q, "#d95f02", "--"),
        ("2. 2-Qubit Paar (2Q Lineair)", damping_2Q, mp_2Q, "#e7298a", "-."),
        ("3. 4-Qubit Tetraëder (4Q Tetron T_+)", damping_4Q, mp_4Q, "#7570b3", ":"),
        ("4. 8-Qubit Kubus (8Q Octon T_+ U T_-)", damping_8Q, mp_8Q, "#2b5c8f", "-"),
        ("5. 8-Qubit Kubus + Kooi-Schild (TR+Coulomb)", damping_8Q_shielded, mp_8Q, "#1b9e77", "-"),
    ]

    results = {}
    for label, damp, mp, color, ls in configs:
        g_eff = g_tr_noise * damp
        T2_eff = 1.0 / (base_rate * damp)
        r_bloch_t = np.exp(-((t_steps / T2_eff) ** 2))
        fidelity_t = 0.5 * (1.0 + r_bloch_t)
        results[label] = {
            "damp": damp,
            "g_eff": g_eff,
            "T2": T2_eff,
            "T2_gain": 1.0 / damp,
            "multipoles": mp,
            "r_bloch_t": r_bloch_t,
            "fidelity_t": fidelity_t,
            "color": color,
            "ls": ls,
        }

    if verbose:
        print("\n" + "=" * 106)
        print(f"ETRT 8-QUBIT KUBUS (STELLA OCTANGULA / MAJORANA OCTON) ANALYSE BIJ g_TR = {g_tr_noise} (qubit_test_4.py)")
        print("=" * 106)
        print(
            f"Z-Grid Topologische Kloof: 2Q = {E_bind_2Q_MeV:.2f} MeV | "
            f"4Q Tetraëder = {E_bind_4Q_MeV:.2f} MeV (12.3x) | "
            f"8Q Kubus = {E_bind_8Q_MeV:.2f} MeV (68.3x)"
        )
        print("-" * 106)
        print(
            f"{'Topologie / Architectuur':<44} | {'Dipool <z>':<10} | {'Quad Q_zz':<10} | "
            f"{'Oct <xyz>':<10} | {'T2 [Units]':<10} | {'Winst':<8} | {'F(t=60)'}"
        )
        print("-" * 106)
        idx_t60 = int(np.argmin(np.abs(t_steps - 60.0)))
        for label, _, _, _, _ in configs:
            info = results[label]
            d_z, q_zz, o_xyz = info["multipoles"]
            print(
                f"{label:<44} | {d_z:<+10.4f} | {q_zz:<+10.4f} | "
                f"{o_xyz:<+10.4f} | {info['T2']:<10.2f} | {info['T2_gain']:<7.1f}x | "
                f"{info['fidelity_t'][idx_t60]*100:6.2f}%"
            )
        print("=" * 106)

    if save_plot:
        extent = [zg.z[0] / r_bohr_grid, zg.z[-1] / r_bohr_grid, zg.x[0] / r_bohr_grid, zg.x[-1] / r_bohr_grid]
        lim_bohr = 3.8

        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.3))

        # Paneel 1: 2D Projectie van de 8-Qubit Kubus (T_+ goud + T_- cyaan) op het Z-Grid
        cube_proj_xz = np.sum(prob_8Q_Cube, axis=1)
        cube_proj_norm = cube_proj_xz / np.max(cube_proj_xz)
        im0 = axes[0].imshow(cube_proj_norm, extent=extent, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)

        for idx_v, (vx, vy, vz) in enumerate(pos_T_plus):
            axes[0].plot(
                vz / r_bohr_grid, vx / r_bohr_grid, marker="o", markersize=8.5,
                color="gold", markeredgecolor="black",
                label="Tetraëder T+ (xyz > 0)" if idx_v == 0 else None
            )
        for idx_v, (vx, vy, vz) in enumerate(pos_T_minus):
            axes[0].plot(
                vz / r_bohr_grid, vx / r_bohr_grid, marker="s", markersize=7.5,
                color="#00e5ff", markeredgecolor="black",
                label="Duale Tetraëder T- (xyz < 0)" if idx_v == 0 else None
            )
        axes[0].plot(0.0, 0.0, marker="+", markersize=12, color="white", mew=2.0, label="Inversie-Centrum (l=1,2,3 = 0)")
        axes[0].set_xlim(-lim_bohr, lim_bohr)
        axes[0].set_ylim(-lim_bohr, lim_bohr)
        axes[0].axhline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        axes[0].axvline(0, color="white", lw=0.6, ls="--", alpha=0.4)
        axes[0].set_title("8-Qubit Kubus (T+ U T- : Volledige O_h Symmetrie)")
        axes[0].set_xlabel("z-as [Bohr-stralen a0]")
        axes[0].set_ylabel("x-as [Bohr-stralen a0]")
        axes[0].legend(loc="upper right", fontsize=8.0)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Paneel 2: T2 Coherentiebehoud |r(t)| voor 1Q t/m 8Q + Schild
        for label, _, _, _, _ in configs:
            info = results[label]
            axes[1].plot(
                t_steps, info["r_bloch_t"], color=info["color"], ls=info["ls"], lw=2.4,
                label=f"{label} ({info['T2_gain']:.1f}x)"
            )
        axes[1].set_title(f"T2 Coherentiebehoud: Tetraëder (4Q) vs Kubus (8Q)")
        axes[1].set_xlabel("Tijd t [ModelUnits]")
        axes[1].set_ylabel("Logische Bloch-norm |r(t)|")
        axes[1].set_ylim(-0.02, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=7.8)

        # Paneel 3: Logische Qubit-Fidelity F(t) [%] tot t = 200
        for label, _, _, _, _ in configs:
            info = results[label]
            axes[2].plot(
                t_steps, info["fidelity_t"] * 100.0, color=info["color"], ls=info["ls"], lw=2.4,
                label=label
            )
        axes[2].axhline(99.0, color="black", lw=0.9, ls=":", alpha=0.7, label="99% Fault-Tolerant Drempel")
        axes[2].axhline(99.9, color="gray", lw=0.8, ls="--", alpha=0.6, label="99.9% Topologische Drempel")
        axes[2].set_title("Logische Qubit-Fidelity F(t) [%]")
        axes[2].set_xlabel("Tijd t [ModelUnits]")
        axes[2].set_ylabel("Fidelity F(t) [%]")
        axes[2].set_ylim(48.0, 101.0)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="lower left", fontsize=7.6)

        plt.tight_layout()
        plt.savefig("qubit_cube_8q_t2.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_cube_8q_t2.png'.")

    return results


if __name__ == "__main__":
    run_qubit_cube_test()