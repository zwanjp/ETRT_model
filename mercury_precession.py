import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from sun_earth_moon_tr import G_NEWTON, C_LIGHT, M_SUN, R_SUN, OMEGA_SUN, SECONDS_PER_DAY

# Astronomische constanten voor baanprecessie
DAYS_PER_CENTURY = 36525.0                      # Juliaanse eeuw [dagen]
RAD_TO_ARCSEC = (180.0 / np.pi) * 3600.0        # Boogseconden per radiaal (~206 264.81)


def integrate_etrt_orbit_rk4(a_m, ecc, omega_sun_m=OMEGA_SUN, n_steps=16000):
    """
    Integreert numeriek (4e-orde Runge-Kutta) de Binet-baanvergelijking van een planeet
    in het niet-lineaire ETRT TR-veld van de Zon over één volledige omloop (phi = 0 -> 2*pi):

      u''(phi) + u(phi) = 1/p + 3 * Omega_Zon * u(phi)^2

    waarbij u(phi) = 1 / r(phi) en p = a * (1 - e^2) de semi-latus rectum is.
    De extra term +3 * Omega_Zon * u^2 volgt in ETRT uit:
      (1) De statische kromming van het tijdritme TR_loc(r) = exp(-Omega_Zon / r) in E = m*c^2*TR_loc (+1 * Omega_Zon * u^2)
      (2) De dynamische aanpassing van de bewegende TR-put van de planeet (met v^2/c^2 ~ 2*Omega_Zon/r)
          aan de variërende z-grid stijfheid (+2 * Omega_Zon * u^2).
    """
    p_m = a_m * (1.0 - ecc**2)
    phi_arr = np.linspace(0.0, 2.0 * np.pi, n_steps + 1)
    dphi = phi_arr[1] - phi_arr[0]

    du = 0.0
    ddu = 0.0
    du_history = np.zeros(n_steps + 1)

    def rhs(ph, du_v, ddu_v):
        u_tot = (1.0 + ecc * np.cos(ph)) / p_m + du_v
        return ddu_v, -du_v + 3.0 * omega_sun_m * (u_tot**2)

    for i in range(n_steps):
        ph = phi_arr[i]
        k1_u, k1_v = rhs(ph, du, ddu)
        k2_u, k2_v = rhs(ph + 0.5 * dphi, du + 0.5 * dphi * k1_u, ddu + 0.5 * dphi * k1_v)
        k3_u, k3_v = rhs(ph + 0.5 * dphi, du + 0.5 * dphi * k2_u, ddu + 0.5 * dphi * k2_v)
        k4_u, k4_v = rhs(ph + dphi, du + dphi * k3_u, ddu + dphi * k3_v)
        du += (dphi / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        ddu += (dphi / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        du_history[i + 1] = du

    # Perihelium-verschuiving per omloop [rad] uit de afgeleide bij phi = 2*pi:
    dphi_orbit_rad = (p_m / ecc) * ddu
    return dphi_orbit_rad, phi_arr, du_history


def run_mercury_precession_model(verbose=True, save_plot=True):
    """
    ETRT Simulatie van de Periheliumprecessie van Mercurius (en de overige binnenplaneten + Icarus)
    vanuit de niet-lineaire Z-Grid stijfheid en E = m * c^2 * TR_loc (mercury_precession.py).
    Gebaseerd op ETRT Hoofdstuk 3.3 ('The Perihelion Precession of Mercury').
    """
    # Planetaire baangegevens: (Naam, a [10^9 m = miljoen km], excentriciteit e, omlooptijd T [dagen], Empirisch [''/eeuw])
    solar_bodies = [
        ("1. Mercurius", 57.90905, 0.205630,  87.9691, 42.98),
        ("2. Venus",    108.20893, 0.006772, 224.7010,  8.62),
        ("3. Aarde",    149.59802, 0.016709, 365.2564,  3.84),
        ("4. Mars",     227.93920, 0.093400, 686.9800,  1.35),
        ("5. Icarus",   161.26600, 0.826800, 408.7800, 10.05),
    ]

    results = {}
    phi_merc_arr = None
    du_merc_arr = None

    for name, a_Gm, ecc, T_days, emp_arcsec in solar_bodies:
        a_m = a_Gm * 1e9
        p_m = a_m * (1.0 - ecc**2)
        r_peri_m = a_m * (1.0 - ecc)
        r_ap_m = a_m * (1.0 + ecc)

        # Zonne-tijdritme TR(r) in perihelium en aphelium
        psi_peri = OMEGA_SUN / r_peri_m
        psi_ap = OMEGA_SUN / r_ap_m
        dtr_swing = psi_peri - psi_ap  # TR-stijfheidsverschil tussen perihelium en aphelium
        stiffness_ratio = (r_ap_m / r_peri_m)  # Verhouding Psi_peri / Psi_ap

        # Analytische ETRT precessie per omloop en per eeuw
        orbits_per_century = DAYS_PER_CENTURY / T_days
        dphi_ana_orb_rad = (6.0 * np.pi * OMEGA_SUN) / p_m
        dphi_ana_cent_arcsec = dphi_ana_orb_rad * orbits_per_century * RAD_TO_ARCSEC

        # Numerieke RK4 baanintegratie van de ETRT Binet-vergelijking
        dphi_rk4_orb_rad, phi_grid, du_hist = integrate_etrt_orbit_rk4(a_m, ecc)
        dphi_rk4_cent_arcsec = dphi_rk4_orb_rad * orbits_per_century * RAD_TO_ARCSEC

        if "Mercurius" in name:
            phi_merc_arr = phi_grid
            du_merc_arr = du_hist

        results[name] = {
            "a_Gm": a_Gm,
            "ecc": ecc,
            "T_days": T_days,
            "orbits_per_century": orbits_per_century,
            "r_peri_Gm": r_peri_m / 1e9,
            "r_ap_Gm": r_ap_m / 1e9,
            "psi_peri": psi_peri,
            "psi_ap": psi_ap,
            "dtr_swing": dtr_swing,
            "clock_swing_us_day": dtr_swing * SECONDS_PER_DAY * 1e6,
            "stiffness_ratio": stiffness_ratio,
            "dphi_orb_arcsec": dphi_ana_orb_rad * RAD_TO_ARCSEC,
            "dphi_ana_arcsec_cent": dphi_ana_cent_arcsec,
            "dphi_rk4_arcsec_cent": dphi_rk4_cent_arcsec,
            "dphi_emp_arcsec_cent": emp_arcsec,
        }

    merc_info = results["1. Mercurius"]

    if verbose:
        print("\n" + "=" * 118)
        print("ETRT PERIHELIUMPRECESSIE VAN MERCURIUS & BINNENPLANETEN DOOR NIET-LINEAIRE Z-GRID STIJFHEID (mercury_precession.py)")
        print("=" * 118)
        print("1. Fysisch Mechanisme:   Mercurius beweegt op een sterk excentrische baan (e = 0.2056, r = 46.0 tot 69.8 M km)")
        print("                         diep door de steilste regio van de Zonne-TR-put. De niet-lineaire rooster-elasticiteit")
        print("                         en de continue aanpassing van E = m·c²·TR_loc leveren een extra 1/r³ herstelkracht.")
        print("2. ETRT Binet-Vgl.:      d²u/dφ² + u = 1/p + 3·Ω_Zon·u²   ==>   Δφ_omloop = 6·π·Ω_Zon / [a·(1 - e²)]")
        print("-" * 118)
        print(f"Gravitatie-spankracht Zon (Ω_Zon = GM_☉/c²):  {OMEGA_SUN:,.2f} m")
        print(f"Mercurius Perihelium / Aphelium:              r_peri = {merc_info['r_peri_Gm']:.2f} M km | r_ap = {merc_info['r_ap_Gm']:.2f} M km")
        print(f"Zonne-TR Put op Mercuriusbaan:                ΔTR_peri = -{merc_info['psi_peri']:.4e} | ΔTR_ap = -{merc_info['psi_ap']:.4e}")
        print(f"Omloop-Oscillatie in Z-Grid Spanning:         ΔTR_swing = {merc_info['dtr_swing']:.4e} ({merc_info['clock_swing_us_day']:.1f} μs/dag, +{(merc_info['stiffness_ratio']-1)*100:.1f}% in perihelium!)")
        print(f"Mercurius Precessie per Omloop (87.97 d):     {merc_info['dphi_orb_arcsec']:.4f} boogseconden/omloop (× {merc_info['orbits_per_century']:.2f} omlopen/eeuw)")
        print(f"VOORSPELDE MERCURIUS PRECESSIE (ETRT):        {merc_info['dphi_rk4_arcsec_cent']:.2f} ''/eeuw (Analytisch: {merc_info['dphi_ana_arcsec_cent']:.2f}, Empirisch: {merc_info['dphi_emp_arcsec_cent']:.2f} ''/eeuw)")
        print("-" * 118)
        print(
            f"{'Hemellichaam':<16} | {'a [M km]':<9} | {'Excentr. e':<10} | {'ΔTR_swing':<12} | "
            f"{'Omlopen/eeuw':<12} | {'ETRT RK4 [''/eeuw]':<18} | {'Empirisch [''/eeuw]':<19} | {'Match'}"
        )
        print("-" * 118)
        for name, _, _, _, _ in solar_bodies:
            inf = results[name]
            match_pct = 100.0 - abs(inf["dphi_rk4_arcsec_cent"] - inf["dphi_emp_arcsec_cent"]) / inf["dphi_emp_arcsec_cent"] * 100.0
            print(
                f"{name:<16} | {inf['a_Gm']:<9.2f} | {inf['ecc']:<10.4f} | {inf['dtr_swing']:<12.3e} | "
                f"{inf['orbits_per_century']:<12.2f} | {inf['dphi_rk4_arcsec_cent']:<+18.2f} | "
                f"{inf['dphi_emp_arcsec_cent']:<+19.2f} | {match_pct:.2f}%"
            )
        print("=" * 118)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

        # --- PANEEL 1: DE EXCENTRISCHE BAAN VAN MERCURIUS DIEP IN DE ZONNE-TR-PUT (ZOALS IN MERCURY.PY) ---
        ax0 = axes[0]
        r_Gm = np.linspace(15.0, 260.0, 1000)
        r_m = r_Gm * 1e9

        # Werkelijke fysieke TR-reductie (in eenheden van 10^-8) en ETRT profiel met subtiele logaritmische spanning
        dtr_newton_1e8 = -(OMEGA_SUN / r_m) * 1e8
        r_eff_Gm = 8.0
        dtr_etrt_hill_1e8 = dtr_newton_1e8 * (1.0 - 0.12 * np.log(r_Gm / r_eff_Gm))

        ax0.plot(r_Gm, dtr_newton_1e8, color="#d95f02", lw=2.0, ls="--", label="Klassiek 1/r Profiel (Newton)")
        ax0.plot(r_Gm, dtr_etrt_hill_1e8, color="#1b9e77", lw=2.8, label="ETRT Niet-Lineair Z-Grid Profiel (Zon)")

        # Markeer de excentrische baanzone van Mercurius (46.0 - 69.8 M km) en de Aardbaan (149.6 M km)
        r_p_m = merc_info["r_peri_Gm"]
        r_a_m = merc_info["r_ap_Gm"]
        ax0.axvspan(r_p_m, r_a_m, color="#2b5c8f", alpha=0.20, label=f"Mercurius Baan ({r_p_m:.1f} – {r_a_m:.1f} M km)")
        ax0.axvline(r_p_m, color="#2b5c8f", lw=1.3, ls=":")
        ax0.axvline(r_a_m, color="#2b5c8f", lw=1.3, ls=":")
        ax0.axvline(149.6, color="black", lw=1.4, ls="-.", alpha=0.75, label="Aardbaan (1 AU = 149.6 M km)")

        ax0.annotate(
            f"Perihelium (46.0 M km)\n+51.8% sterkere TR-spanning\n(ΔTR = -3.21×10⁻⁸)",
            xy=(r_p_m, -3.21), xytext=(78.0, -5.8),
            fontsize=7.8, fontweight="bold", color="#1a365d",
            arrowprops=dict(arrowstyle="->", color="#1a365d", lw=1.1)
        )
        ax0.annotate(
            "Aarde (149.6 M km)\nBijna cirkelvormig (e=0.017)\nVlakke TR-gradiënt",
            xy=(149.6, -0.987), xytext=(155.0, -3.2),
            fontsize=7.6, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9)
        )

        ax0.set_xlim(15.0, 250.0)
        ax0.set_ylim(-10.0, 0.2)
        ax0.set_title("1. Mercuriusbaan in de Steile Zonne-TR-Put")
        ax0.set_xlabel("Afstand tot de Zon r [Miljoen km]")
        ax0.set_ylabel("Lokaal Tijdritme-verschil (TR(r) - 1) [× 10⁻⁸]")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower right", fontsize=7.5)

        # --- PANEEL 2: ROZET-PRECESSIE VAN DE MERCURIUSBAAN & RK4 RADIALE AFWIJKING ---
        ax1 = axes[1]
        # Visualiseer de rozet-rotatie van de ellips (met vergrote precessiehoek voor visuele duidelijkheid)
        theta_orb = np.linspace(0.0, 2.0 * np.pi, 500)
        a_vis = merc_info["a_Gm"]
        e_vis = merc_info["ecc"]
        p_vis = a_vis * (1.0 - e_vis**2)
        r_ellips = p_vis / (1.0 + e_vis * np.cos(theta_orb))

        for k_orb, (ang_deg, col_orb, al_orb, lbl_orb) in enumerate([
            (0.0, "#2b5c8f", 0.35, "Initiële Baan (Eeuw 0)"),
            (18.0, "#7570b3", 0.55, "Verschoven Baan (Tussenstap)"),
            (36.0, "#1b9e77", 0.95, "Geavanceerd Perihelium (+42.98''/eeuw)"),
        ]):
            ang_rad = np.radians(ang_deg)
            x_orb = r_ellips * np.cos(theta_orb + ang_rad)
            y_orb = r_ellips * np.sin(theta_orb + ang_rad)
            ax1.plot(x_orb, y_orb, color=col_orb, lw=2.2 if k_orb == 2 else 1.6, alpha=al_orb, label=lbl_orb)
            # Markeer het perihelium op theta = 0
            r_p_vis = p_vis / (1.0 + e_vis)
            ax1.plot(
                r_p_vis * np.cos(ang_rad), r_p_vis * np.sin(ang_rad),
                marker="o", markersize=6.5, color=col_orb, markeredgecolor="black", zorder=6
            )

        # Teken de Zon in het brandpunt (0, 0)
        ax1.scatter([0.0], [0.0], color="gold", edgecolor="#d95f02", s=160, zorder=7, label="Zon (Brandpunt TR-Put)")
        ax1.annotate(
            "Perihelium-precessie\nΔφ = +42.98'' / eeuw\n(0.1035'' per omloop)",
            xy=(46.0 * np.cos(np.radians(36.0)), 46.0 * np.sin(np.radians(36.0))),
            xytext=(10.0, 52.0),
            fontsize=8.0, fontweight="bold", color="#1b9e77",
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.2)
        )

        ax1.set_xlim(-80.0, 70.0)
        ax1.set_ylim(-70.0, 75.0)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("2. Precessie-Rozet door Extra Z-Grid Stijfheid")
        ax1.set_xlabel("x-positie in Baanvlak [Miljoen km]")
        ax1.set_ylabel("y-positie in Baanvlak [Miljoen km]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower left", fontsize=7.4)

        # --- PANEEL 3: ETRT VOORSPELLING VS. EMPIRISCHE METINGEN (BINNENPLANETEN + ICARUS) ---
        ax2 = axes[2]
        short_labels = ["Mercurius\n(e=0.206)", "Icarus\n(e=0.827)", "Venus\n(e=0.007)", "Aarde\n(e=0.017)", "Mars\n(e=0.093)"]
        ordered_keys = ["1. Mercurius", "5. Icarus", "2. Venus", "3. Aarde", "4. Mars"]
        etrt_vals = [results[k]["dphi_rk4_arcsec_cent"] for k in ordered_keys]
        emp_vals = [results[k]["dphi_emp_arcsec_cent"] for k in ordered_keys]

        x_idx = np.arange(len(ordered_keys))
        w_b = 0.36
        b_etrt = ax2.bar(x_idx - 0.5 * w_b, etrt_vals, width=w_b, color="#1b9e77", edgecolor="black", label="ETRT Z-Grid RK4 Voorspelling")
        b_emp = ax2.bar(x_idx + 0.5 * w_b, emp_vals, width=w_b, color="#2b5c8f", edgecolor="black", alpha=0.88, label="Empirisch (Astronomische Meting)")

        for bar_i, val_i in zip(b_etrt, etrt_vals):
            ax2.text(bar_i.get_x() + bar_i.get_width() / 2.0 - 0.04, val_i + 0.9, f"{val_i:.2f}''", ha="center", fontsize=7.5, fontweight="bold", color="#1b9e77")
        for bar_i, val_i in zip(b_emp, emp_vals):
            ax2.text(bar_i.get_x() + bar_i.get_width() / 2.0 + 0.04, val_i + 2.6, f"{val_i:.2f}''", ha="center", fontsize=7.5, fontweight="bold", color="#2b5c8f")

        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(short_labels, fontsize=8.0)
        ax2.set_ylim(0.0, 49.5)
        ax2.set_title("3. Periheliumprecessie Binnenplaneten & Icarus [''/eeuw]")
        ax2.set_ylabel("Periheliumprecessie Δφ [boogseconden / eeuw]")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=8.0)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
        plt.savefig("mercury_precession_etrt.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'mercury_precession_etrt.png'.")

    return {
        "Omega_sun_m": OMEGA_SUN,
        "mercury_dtr_swing": merc_info["dtr_swing"],
        "mercury_stiffness_ratio": merc_info["stiffness_ratio"],
        "mercury_dphi_orb_arcsec": merc_info["dphi_orb_arcsec"],
        "mercury_etrt_arcsec_cent": merc_info["dphi_rk4_arcsec_cent"],
        "mercury_emp_arcsec_cent": merc_info["dphi_emp_arcsec_cent"],
        "venus_etrt_arcsec_cent": results["2. Venus"]["dphi_rk4_arcsec_cent"],
        "venus_emp_arcsec_cent": results["2. Venus"]["dphi_emp_arcsec_cent"],
        "earth_etrt_arcsec_cent": results["3. Aarde"]["dphi_rk4_arcsec_cent"],
        "earth_emp_arcsec_cent": results["3. Aarde"]["dphi_emp_arcsec_cent"],
        "mars_etrt_arcsec_cent": results["4. Mars"]["dphi_rk4_arcsec_cent"],
        "mars_emp_arcsec_cent": results["4. Mars"]["dphi_emp_arcsec_cent"],
        "icarus_etrt_arcsec_cent": results["5. Icarus"]["dphi_rk4_arcsec_cent"],
        "icarus_emp_arcsec_cent": results["5. Icarus"]["dphi_emp_arcsec_cent"],
        "bodies": results,
    }


if __name__ == "__main__":
    run_mercury_precession_model()
