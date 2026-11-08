import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from sun_earth_moon_tr import G_NEWTON, C_LIGHT, M_SUN

# Kosmologische en galactische schaalconstanten (SI)
KPC_M = 3.08567758149e19                        # 1 kiloparsec [m]
MPC_M = 1.0e3 * KPC_M                           # 1 Megaparsec [m]
H0_KMS_MPC = 67.4                               # Hubble-constante (Planck 2018) [km/s/Mpc]
H0_SI = (H0_KMS_MPC * 1e3) / MPC_M              # Hubble-constante [1/s] (~2.184e-18 s^-1)

# Fundamentele ETRT Galactische Rooster-Versnelling a_0 (uit c, H_0 en de Euler-schaal e van de Heuvelrug):
#   a_0_ETRT = (c * H_0) / (2 * e) = 1.2045e-10 m/s^2  (Empirisch SPARC/Milgrom: 1.20e-10 m/s^2, 99.6% match!)
A0_ETRT_MS2 = (C_LIGHT * H0_SI) / (2.0 * np.e)
A0_EMP_MS2 = 1.20e-10

# Kosmische Energiebalans uit de fundamentele ETRT elasticiteitsparameter XI_E = 0.3230:
# - Geknopte / geclusterde Materie-fractie: Omega_m_ETRT = XI_E = 0.3230 (32.30%, Planck 2018: 31.53%)
# - Ongeknopte Rooster-Expansie / 'Donkere Energie': Omega_Lambda_ETRT = 1 - XI_E = 0.6770 (67.70%, Planck 2018: 68.47%)
OMEGA_M_ETRT = XI_E
OMEGA_M_EMP = 0.3153
OMEGA_LAMBDA_ETRT = 1.0 - XI_E
OMEGA_LAMBDA_EMP = 0.6847


def enclosed_baryonic_mass_mw(r_kpc, M_bulge_sun=1.1e10, r_bulge_kpc=0.7, M_disk_sun=5.0e10, r_disk_kpc=2.8):
    """
    Berekent de ingesloten zichtbare (baryonische) massa M_bar(r) [in kg] van de Melkweg
    op galactocentrische afstand r_kpc (Hernquist-bulge + exponentiële schijf, zonder Dark Matter).
    Totale baryonische massa: M_bar_tot = 6.10e10 M_sun.
    """
    r_arr = np.maximum(np.asarray(r_kpc, dtype=float), 0.05)
    m_bulge = M_bulge_sun * ((r_arr / (r_arr + r_bulge_kpc)) ** 2)
    u_d = r_arr / r_disk_kpc
    m_disk = M_disk_sun * (1.0 - (1.0 + u_d) * np.exp(-u_d))
    return (m_bulge + m_disk) * M_SUN


def compute_galactic_rotation_curve(r_kpc, M_bar_enclosed_kg):
    """
    Berekent de omloopsnelheid v(r) [km/s] in een sterrenstelsel vanuit ETRT Hoofdstuk 3.4:
      1. Klassieke Newton-bijdrage van alleen de zichtbare materie:
         g_N(r) = G * M_bar(r) / r^2   ==>   v_Newton(r) = sqrt(r * g_N(r))
      2. ETRT Logaritmische Heuvelrug & Spiegel-Domein Spanning:
         g_hill(r) = sqrt(g_N(r) * A0_ETRT_MS2) * [1 - exp(-(r / r_trans)^2)]
    """
    r_m = np.maximum(np.asarray(r_kpc, dtype=float), 0.05) * KPC_M
    g_newton = (G_NEWTON * M_bar_enclosed_kg) / (r_m**2)

    r_trans_kpc = 4.2
    hill_envelope = 1.0 - np.exp(-((np.asarray(r_kpc, dtype=float) / r_trans_kpc) ** 2))
    g_hill = np.sqrt(g_newton * A0_ETRT_MS2) * hill_envelope

    g_etrt_tot = g_newton + g_hill

    v_newton_kms = np.sqrt(r_m * g_newton) / 1e3
    v_hill_kms = np.sqrt(r_m * g_hill) / 1e3
    v_etrt_kms = np.sqrt(r_m * g_etrt_tot) / 1e3

    return v_etrt_kms, v_newton_kms, v_hill_kms


def run_dark_matter_energy_model(verbose=True, save_plot=True):
    """
    ETRT Simulatie van Donkere Materie (Vlakke Galactische Rotatiecurven & Tully-Fisher) en
    Donkere Energie (Omslag naar Kosmische Afstoting op e^2*R_halo & Omega_m / Omega_Lambda)
    (dark_matter_energy.py). Gebaseerd op ETRT Hoofdstuk 3.4 ('The Nature of Dark Matter and Dark Energy').
    """
    r_kpc_grid = np.linspace(0.5, 35.0, 400)
    M_bar_mw_grid = enclosed_baryonic_mass_mw(r_kpc_grid)
    v_etrt_mw, v_newt_mw, v_hill_mw = compute_galactic_rotation_curve(r_kpc_grid, M_bar_mw_grid)

    mw_obs_data = [
        (2.5,  208.0, 7.0),
        (5.0,  226.0, 6.0),
        (8.2,  225.0, 5.0),
        (12.0, 221.0, 6.0),
        (15.0, 217.0, 7.0),
        (20.0, 212.0, 8.0),
        (25.0, 206.0, 10.0),
        (30.0, 202.0, 12.0),
    ]

    v_sun_etrt, v_sun_newt, _ = compute_galactic_rotation_curve(8.2, enclosed_baryonic_mass_mw(8.2))
    v_25_etrt, v_25_newt, _ = compute_galactic_rotation_curve(25.0, enclosed_baryonic_mass_mw(25.0))
    v_sun_etrt, v_sun_newt = float(v_sun_etrt), float(v_sun_newt)
    v_25_etrt, v_25_newt = float(v_25_etrt), float(v_25_newt)

    sparc_galaxies = [
        ("1. DDO 154 (Dwergstelsel)",  3.50e8,   8.0,  47.0, "o", "#d95f02"),
        ("2. NGC 2403 (Spiraal)",      1.10e10, 14.5, 134.0, "s", "#7570b3"),
        ("3. NGC 3198 (Referentie)",   2.10e10, 21.0, 150.0, "^", "#e7298a"),
        ("4. Melkweg (SBbc Spiraal)",  6.10e10, 22.0, 212.0, "D", "#2b5c8f"),
        ("5. Andromeda (M31)",         1.20e11, 31.0, 250.0, "*", "#1b9e77"),
    ]

    galaxy_results = {}
    for g_name, m_bar_sun, r_meas_kpc, v_obs_kms, mk, col in sparc_galaxies:
        m_bar_kg = m_bar_sun * M_SUN
        r_meas_m = r_meas_kpc * KPC_M
        g_n_meas = (G_NEWTON * m_bar_kg) / (r_meas_m**2)
        g_tot_meas = g_n_meas + np.sqrt(g_n_meas * A0_ETRT_MS2)
        v_etrt_gal_kms = float(np.sqrt(r_meas_m * g_tot_meas) / 1e3)
        v_asymp_kms = float(((G_NEWTON * m_bar_kg * A0_ETRT_MS2) ** 0.25) / 1e3)
        v_newt_gal_kms = float(np.sqrt(r_meas_m * g_n_meas) / 1e3)

        galaxy_results[g_name] = {
            "M_bar_sun": m_bar_sun,
            "r_meas_kpc": r_meas_kpc,
            "v_etrt_kms": v_etrt_gal_kms,
            "v_asymp_kms": v_asymp_kms,
            "v_newt_kms": v_newt_gal_kms,
            "v_obs_kms": v_obs_kms,
            "marker": mk,
            "color": col,
        }

    R_halo_Mpc = 0.184
    R_zero_gravity_etrt_Mpc = (np.e**2) * R_halo_Mpc
    R_zero_gravity_emp_Mpc = 1.380

    if verbose:
        print("\n" + "=" * 118)
        print("ETRT DONKERE MATERIE (ROTATIECURVEN) & DONKERE ENERGIE (KOSMISCHE AFSTOTING) ANALYSE (dark_matter_energy.py)")
        print("=" * 118)
        print(f"ETRT Galactische Rooster-Versnelling (a_0):   {A0_ETRT_MS2:.4e} m/s² (= c·H0/(2e), Empirisch: {A0_EMP_MS2:.2e} m/s²)")
        print(f"Melkweg Snelheid op Zonnebaan (r = 8.2 kpc):  v_ETRT = {v_sun_etrt:.1f} km/s (Newton: {v_sun_newt:.1f} km/s, Gaia DR3: 225.0 km/s)")
        print(f"Melkweg Buitenrand (r = 25.0 kpc):            v_ETRT = {v_25_etrt:.1f} km/s (Newton: {v_25_newt:.1f} km/s, Gaia DR3: 206.0 km/s)")
        print(f"Lokale Groep Omslagpunt (Φ_G = 0, e²·R_halo): r_omslag = {R_zero_gravity_etrt_Mpc:.2f} Mpc (Waargenomen R_0: {R_zero_gravity_emp_Mpc:.2f} ± 0.10 Mpc)")
        print(f"Kosmische Materie-Fractie (Ω_m = XI_E):       {OMEGA_M_ETRT*100:.2f}% (Planck 2018: {OMEGA_M_EMP*100:.2f}%)")
        print(f"Donkere Energie-Fractie (Ω_Λ = 1 - XI_E):     {OMEGA_LAMBDA_ETRT*100:.2f}% (Planck 2018: {OMEGA_LAMBDA_EMP*100:.2f}%)")
        print("=" * 118)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

        ax0 = axes[0]
        ax0.plot(r_kpc_grid, v_newt_mw, color="#d95f02", lw=2.2, ls="--", label="Klassiek Newton (Alleen zichtbare materie)")
        ax0.plot(r_kpc_grid, v_hill_mw, color="#7570b3", lw=2.0, ls=":", label="ETRT Log-Heuvelrug / Spiegel-Domein Term")
        ax0.plot(r_kpc_grid, v_etrt_mw, color="#1b9e77", lw=3.0, label="Totale ETRT Rotatiecurve v_ETRT(r)")
        ax0.fill_between(
            r_kpc_grid, v_newt_mw, v_etrt_mw,
            color="#1b9e77", alpha=0.14, label="Toegeschreven aan 'Donkere Materie'"
        )

        obs_r = [pt[0] for pt in mw_obs_data]
        obs_v = [pt[1] for pt in mw_obs_data]
        obs_err = [pt[2] for pt in mw_obs_data]
        ax0.errorbar(
            obs_r, obs_v, yerr=obs_err, fmt="o", color="#2b5c8f", ecolor="#2b5c8f",
            elinewidth=1.4, capsize=3.5, markersize=6.0, zorder=6, label="Waargenomen (Gaia DR3 / Eilers et al.)"
        )
        ax0.axvline(8.2, color="gold", lw=1.5, ls="-.", label="Zonnebaan (r = 8.2 kpc: v ≈ 225 km/s)")

        ax0.set_xlim(0.0, 35.0)
        ax0.set_ylim(60.0, 265.0)
        ax0.set_title("1. Melkweg Rotatiecurve Zonder 'Dark Matter'")
        ax0.set_xlabel("Galactocentrische afstand r [kiloparsec (kpc)]")
        ax0.set_ylabel("Omloopsnelheid v(r) [km/s]")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower left", fontsize=7.4)

        ax1 = axes[1]
        v_line_kms = np.logspace(np.log10(35.0), np.log10(320.0), 200)
        v_line_ms = v_line_kms * 1e3
        M_btfr_etrt_sun = ((v_line_ms**4) / (G_NEWTON * A0_ETRT_MS2 * 1.48)) / M_SUN
        M_btfr_asymp_sun = ((v_line_ms**4) / (G_NEWTON * A0_ETRT_MS2)) / M_SUN

        ax1.plot(v_line_kms, M_btfr_etrt_sun, color="#1b9e77", lw=2.6, label="ETRT Wet op r_meet: M_bar ∝ v⁴ (helling n = 4.00)")
        ax1.plot(v_line_kms, M_btfr_asymp_sun, color="gray", lw=1.4, ls="--", label="Asymptotische limiet r → ∞: v_∞ = (G·M_bar·a0)^¼")

        for g_name, _, _, _, _, _ in sparc_galaxies:
            gi = galaxy_results[g_name]
            short_g = g_name.split(" (")[0].replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "")
            ax1.scatter(
                [gi["v_obs_kms"]], [gi["M_bar_sun"]],
                color=gi["color"], marker=gi["marker"], s=85, edgecolor="black", zorder=6,
                label=f"{short_g}: v_ETRT={gi['v_etrt_kms']:.0f} (obs={gi['v_obs_kms']:.0f} km/s)"
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(35.0, 320.0)
        ax1.set_ylim(1.8e8, 2.5e11)
        ax1.set_title("2. Baryonische Tully-Fisher Relatie (M_bar ∝ v⁴)")
        ax1.set_xlabel("Vlakke Buitensnelheid v_flat [km/s] (log)")
        ax1.set_ylabel("Zichtbare Baryonische Massa M_bar [M_☉] (log)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc="upper left", fontsize=7.4)

        ax2 = axes[2]
        r_Mpc_arr = np.linspace(0.20, 4.50, 400)
        phi_cosmo_arr = 2.0 - np.log(r_Mpc_arr / R_halo_Mpc)

        ax2.plot(r_Mpc_arr, phi_cosmo_arr, color="#2b5c8f", lw=2.8, label="ETRT Gradiëntfactor Φ_G(r) = 2 - ln(r/R_halo)")
        ax2.axhline(0.0, color="black", lw=1.0, ls=":")
        ax2.axvline(
            R_zero_gravity_etrt_Mpc, color="#d95f02", lw=1.8, ls="--",
            label=f"Omslagpunt e²·R_halo = {R_zero_gravity_etrt_Mpc:.2f} Mpc (Waargenomen R0≈1.38 Mpc)"
        )
        ax2.fill_between(
            r_Mpc_arr, 0.0, phi_cosmo_arr, where=(phi_cosmo_arr >= 0),
            color="#1b9e77", alpha=0.18, label="Galactische Aantrekking (Φ_G > 0: Lokale Groep gebonden)"
        )
        ax2.fill_between(
            r_Mpc_arr, 0.0, phi_cosmo_arr, where=(phi_cosmo_arr < 0),
            color="#e7298a", alpha=0.18, label="Kosmische Afstoting / 'Donkere Energie' (Φ_G < 0: Hubble-uitdijing)"
        )

        ax2.text(
            1.62, 1.45,
            f"Oersplitsings-Balans (+1 / -1 Rooster):\n"
            f"• Materie Ω_m = XI_E = {OMEGA_M_ETRT*100:.1f}% (Planck: {OMEGA_M_EMP*100:.1f}%)\n"
            f"• Donkere Energie Ω_Λ = 1 - XI_E = {OMEGA_LAMBDA_ETRT*100:.1f}% (Planck: {OMEGA_LAMBDA_EMP*100:.1f}%)",
            fontsize=7.8, fontweight="bold", color="#1a365d",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7fafc", edgecolor="#2b5c8f", alpha=0.92)
        )

        ax2.set_xlim(0.20, 4.50)
        ax2.set_ylim(-1.30, 2.25)
        ax2.set_title("3. Donkere Energie: Omslag naar Kosmische Afstoting")
        ax2.set_xlabel("Intergalactische afstand r [Megaparsec (Mpc)]")
        ax2.set_ylabel("ETRT Gradiëntfactor Φ_G(r) [+: Aantrekking, -: Afstoting]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower left", fontsize=7.3)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
        plt.savefig("dark_matter_energy_etrt.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'dark_matter_energy_etrt.png'.")

    return {
        "a0_etrt_ms2": A0_ETRT_MS2,
        "a0_emp_ms2": A0_EMP_MS2,
        "v_sun_etrt_kms": v_sun_etrt,
        "v_sun_emp_kms": 225.0,
        "v_mw_25kpc_etrt_kms": v_25_etrt,
        "v_mw_25kpc_emp_kms": 206.0,
        "R_zero_gravity_etrt_Mpc": R_zero_gravity_etrt_Mpc,
        "R_zero_gravity_emp_Mpc": R_zero_gravity_emp_Mpc,
        "Omega_m_etrt": OMEGA_M_ETRT,
        "Omega_m_emp": OMEGA_M_EMP,
        "Omega_Lambda_etrt": OMEGA_LAMBDA_ETRT,
        "Omega_Lambda_emp": OMEGA_LAMBDA_EMP,
        "galaxies": galaxy_results,
    }


if __name__ == "__main__":
    run_dark_matter_energy_model()