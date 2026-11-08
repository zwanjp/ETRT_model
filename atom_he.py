import io
import re
import contextlib
import numpy as np

try:
    from zgrid import ZGrid
    from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
    from neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
    from electron import run_unified_model as run_electron_model
    from deuterium import _extract_model_metrics
    from massfield import calculate_psi_component, apply_tr_to_energy
    from parameters import *
    from visualize import plot_1d_slice
except ImportError:
    from .zgrid import ZGrid
    from .proton import run_proton_model, get_spherical_envelope as get_proton_envelope
    from .neutron import run_neutron_model, get_spherical_envelope as get_neutron_envelope
    from .electron import run_unified_model as run_electron_model
    from .deuterium import _extract_model_metrics
    from .massfield import calculate_psi_component, apply_tr_to_energy
    from .parameters import *
    from .visualize import plot_1d_slice


def run_atom_he_model(p_metrics=None, n_metrics=None, e_metrics=None, verbose=True):
    """
    ETRT 3D Model voor Helium-4 (^4He: 2 Protonen + 2 Neutronen in een rooster-uitgelijnde
    tetraëder op het nucleaire Z-Grid + 1s^2 Elektronschil op atomaire schaal).
    """
    if p_metrics is None or n_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton, Neutron, Elektron)...")
            print("-" * 60)
        p_metrics = _extract_model_metrics(run_proton_model, verbose=verbose)
        if verbose:
            print("-" * 60)
        n_metrics = _extract_model_metrics(run_neutron_model, verbose=verbose)
        if verbose:
            print("-" * 60)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=verbose)
        if verbose:
            print("-" * 60)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    R_rms_sq_p = p_metrics["R_rms_sq"]

    M_n_raw = n_metrics["M0_raw"]
    M_n_phys = n_metrics["M0_phys"]

    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0

    # 2. Atomaire 1s^2 elektronschil schaling (kernlading Z = 2)
    atomic_scale_factor = 1.0 / (ALPHA**2)
    Z_charge = 2.0
    M_alpha_approx = 2.0 * M_p_phys + 2.0 * M_n_phys
    red_mass_factor_he = M_alpha_approx / (M_alpha_approx + M_e_phys)

    r_bohr_grid_he = (R_rms_e_model / Z_charge) / red_mass_factor_he
    R_bohr_he_model = r_bohr_grid_he * atomic_scale_factor

    r_e_m = R_E_FM * 1e-15
    m_per_unit = r_e_m / R_rms_e_model if R_rms_e_model > 1e-12 else 0.0
    a_0_he_phys_m = R_bohr_he_model * m_per_unit

    if verbose:
        print("2. Z-Grid initialiseren voor de Helium-4 kern (2p + 2n)...")
    z_grid = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
    r_grid = np.sqrt(z_grid.X**2 + z_grid.Y**2 + z_grid.Z**2)
    cutoff_envelope = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # 3. Rooster-uitgelijnde regelmatige tetraëder
    R_tet = DX_ELECTRON
    d_pair = 2.0 * np.sqrt(2.0) * R_tet
    pos_p1 = ( R_tet,  R_tet,  R_tet)
    pos_p2 = (-R_tet, -R_tet,  R_tet)
    pos_n1 = (-R_tet,  R_tet, -R_tet)
    pos_n2 = ( R_tet, -R_tet, -R_tet)

    r_q_up = Y_UP / 2.0
    r_q_down = Y_DOWN / 2.0
    r_bag_p = (2.0 * r_q_up + r_q_down) / 3.0
    r_bag_n = (1.0 * r_q_up + 2.0 * r_q_down) / 3.0

    p_tr_up = ((Y_VAC / Y_UP)**3) * (((r_q_up) / ((r_q_up) + XI_Z))**2)
    p_tr_down = ((Y_VAC / Y_DOWN)**3) * (((r_q_down) / ((r_q_down) + XI_Z))**2)
    p_amp_up = abs(p_tr_up - 1.0)
    p_amp_down = abs(p_tr_down - 1.0)

    nucleons = [
        ("proton",  pos_p1, r_bag_p, M_p_raw),
        ("proton",  pos_p2, r_bag_p, M_p_raw),
        ("neutron", pos_n1, r_bag_n, M_n_raw),
        ("neutron", pos_n2, r_bag_n, M_n_raw),
    ]

    pressure_list = []
    total_proton_shell = np.zeros_like(z_grid.X)
    E_bag_tot = np.zeros_like(z_grid.X)
    psi_ext_total = np.zeros_like(z_grid.X)

    for n_type, pos, r_bag, m_raw in nucleons:
        X_s = z_grid.X - pos[0]
        Y_s = z_grid.Y - pos[1]
        Z_s = z_grid.Z - pos[2]
        r_i = np.sqrt(X_s**2 + Y_s**2 + Z_s**2)

        if n_type == "proton":
            shell = get_proton_envelope(X_s, Y_s, Z_s, r_bag)
            pressure = (2.0 * p_amp_up + 1.0 * p_amp_down) * shell
            total_proton_shell += shell
        else:
            shell = get_neutron_envelope(X_s, Y_s, Z_s, r_bag)
            pressure = (1.0 * p_amp_up + 2.0 * p_amp_down) * shell

        pressure_list.append(pressure)
        E_bag_tot += np.where(r_i <= r_bag, BAG_ENERGY_DENSITY, 0.0)
        psi_i = calculate_psi_component(z_grid.X, z_grid.Y, z_grid.Z, pos, r_bag, m_raw * OMEGA_MACRO_SCALE)
        psi_ext_total += np.where(r_i > r_bag, psi_i, 0.0)

    total_pressure = sum(pressure_list)

    # 4. Atomaire neutraliteit
    electron_1s2_shell = 2.0 * np.exp(-2.0 * Z_charge * r_grid / r_bohr_grid_he)
    vol_p = np.sum(total_proton_shell * cutoff_envelope) * z_grid.dV
    vol_e = np.sum(electron_1s2_shell * cutoff_envelope) * z_grid.dV
    rho_p = (total_proton_shell * cutoff_envelope) * (2.0 / vol_p) if vol_p > 1e-10 else 0.0
    rho_e = (electron_1s2_shell * cutoff_envelope) * (-2.0 / vol_e) if vol_e > 1e-10 else 0.0
    Q_total = np.sum(rho_p + rho_e) * z_grid.dV

    e_grid_individual_sum = np.zeros_like(z_grid.X)
    for press_i in pressure_list:
        zg_i = ZGrid(shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON)
        zg_i.relax(external_pressure=press_i, iterations=RELAX_ITERATIONS)
        e_grid_individual_sum += zg_i.get_energy_density()

    z_grid.relax(external_pressure=total_pressure, iterations=RELAX_ITERATIONS)
    e_grid_full = z_grid.get_energy_density()

    N_channels = 4.0
    e_grid_singlet = e_grid_individual_sum + (1.0 / N_channels) * (e_grid_full - e_grid_individual_sum)

    E_veld = np.abs(total_pressure) + e_grid_singlet
    E_tot_raw = (E_veld + E_bag_tot) * cutoff_envelope

    TR_mutual = np.exp(-psi_ext_total)
    E_total_density = apply_tr_to_energy(E_tot_raw, TR_mutual)
    M_he_nuc_raw = np.sum(E_total_density) * z_grid.dV

    # 5. Massadefect en Bindingsenergie
    bare_4_raw = 2.0 * M_p_raw + 2.0 * M_n_raw
    delta_M_raw = bare_4_raw - M_he_nuc_raw
    delta_M_phys = delta_M_raw / R_rms_sq_p

    bare_atom_phys = 2.0 * M_p_phys + 2.0 * M_n_phys + 2.0 * M_e_phys
    M_he_phys = bare_atom_phys - delta_M_phys

    M_He_MeV = M_he_phys * mass_conv_MeV
    E_bind_MeV = delta_M_phys * mass_conv_MeV

    if verbose:
        print("\n" + "=" * 70)
        print("GEÜNIFICEERDE ATOOMFYSICA ANALYSE: HELIUM-4 (^4He - atom_he.py)")
        print("=" * 70)
        print(f"Geïntegreerde Totale Lading (Q0):     {Q_total:.4f} (Verwacht: 0.0 Neutraal)")
        print(f"Berekende Bohr-straal (^4He):         {R_bohr_he_model:.4e} [Model Units]")
        print(f"Fysieke Bohr-straal (^4He):           {a_0_he_phys_m:.4e} m")
        print(f"Ruwe Massa 2p + 2n (los):             {bare_4_raw:.4f}")
        print(f"Ruwe Massa ^4He kern (Z-Grid + TR):   {M_he_nuc_raw:.4f}")
        print(f"Fysieke Massa Helium-4 (M_He):        {M_he_phys:.6f} ({M_He_MeV:.3f} MeV/c^2)")
        print("-" * 70)
        print(f"Massadefect (ΔM_phys) [ModelUnit]:    {delta_M_phys:.6f}")
        print(f"VOORSPELDE ALPHA-BINDINGSENERGIE:     {E_bind_MeV:.4f} MeV (Empirisch: 28.30 MeV)")
        print("=" * 70)

    return {
        "Q0": Q_total,
        "R_bohr_model": R_bohr_he_model,
        "a_0_phys_m": a_0_he_phys_m,
        "M_He_phys": M_he_phys,
        "M_He_MeV": M_He_MeV,
        "delta_M_raw": delta_M_raw,
        "delta_M_phys": delta_M_phys,
        "E_bind_MeV": E_bind_MeV,
        "z_grid": z_grid,
        "torsion_density": total_pressure,
    }


if __name__ == "__main__":
    run_atom_he_model()