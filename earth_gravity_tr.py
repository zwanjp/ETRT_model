import numpy as np
import matplotlib.pyplot as plt

# Fysieke constanten voor de Aarde en het vacuüm (SI)
G_NEWTON = 6.67430e-11          # Gravitatieconstante [m^3 kg^-1 s^-2]
C_LIGHT = 299792458.0           # Lichtsnelheid [m/s]
M_EARTH = 5.97219e24            # Massa van de Aarde [kg]
R_EARTH = 6.3710e6              # Gemiddelde straal van de Aarde r_v = R_earth [m] (6371 km)
SECONDS_PER_DAY = 86400.0       # Seconden per etmaal [s/dag]

# Karakteristieke ETRT gravitatie-spankracht van de Aarde:
# Omega_earth = G * M_earth / c^2 (Gravitatiestraal van de Aarde, ~4.435 mm)
OMEGA_EARTH_M = (G_NEWTON * M_EARTH) / (C_LIGHT**2)
PSI_SURFACE = OMEGA_EARTH_M / R_EARTH  # Dimensieloze oppervlakte-reductie (~6.961e-10)


def compute_earth_tr_profile(r_m, r0_m=R_EARTH, omega_m=OMEGA_EARTH_M, eta=1.0, tr_l=1.0):
    """
    Berekent het Time-Rhythm profiel TR(r), de relatieve afwijking delta_TR(r) = TR(r) - TR_L,
    en de ruimtelijke TR-gradiënt dTR/dr vanaf het middelpunt van de Aarde (r = 0)
    tot in de wijde kosmische omgeving (r >> R_earth).

    1. Binnen de Aarde (0 <= r <= R_earth, volgens 'TR_within_objects.pdf' Vgl. 43-45 & massfield.py):
       Psi_in(r) = (Omega / R_earth) * [2 - exp(-(R_earth - r) / R_earth)]
       - Op r = 0 (Middelpunt): Psi_in(0) = (Omega / R_earth) * (2 - 1/e) ≈ 1.6321 * Psi(R_earth)
       - Op r = R_earth (Oppervlak): Psi_in(R_earth) = Omega / R_earth (naadloze C1-aansluiting)

    2. Buiten de Aarde (r >= R_earth, volgens Hoofdstuk 2.4, massfield.py & newton_coulomb.py):
       Psi_ext(r) = (Omega / r) * [1 - eta * ln(r / R_earth)]
       - eta = 1.0: Volledig ETRT-profiel inclusief de logaritmische Tijd-Behouds Heuvelrug
                    (TR = 1 op r = e * R_earth, heuveltop op r = e^2 * R_earth)
       - eta = 0.0: Klassiek Newton/Schwarzschild profiel zonder heuvelrug (puur 1/r)
    """
    r_arr = np.asarray(r_m, dtype=float)
    r_safe_ext = np.maximum(r_arr, r0_m)
    r_safe_in = np.minimum(r_arr, r0_m)

    # Inwendige rooster-stress (0 <= r <= r0_m) met platte bodem in r = 0 (Hoofdstuk 15):
    psi_in = (omega_m / r0_m) * (2.0 - np.exp(-((r0_m**2 - r_safe_in**2) / (2.0 * (r0_m**2)))))

    # Uitwendige rooster-stress (r >= r0_m)
    psi_ext = (omega_m / r_safe_ext) * (1.0 - eta * np.log(r_safe_ext / r0_m))

    psi_tot = np.where(r_arr <= r0_m, psi_in, psi_ext)
    tr_tot = tr_l * np.exp(-psi_tot)
    delta_tr = tr_tot - tr_l

    # Analytische ruimtelijke gradiënt dTR/dr = -TR(r) * (dPsi/dr) [1/m] (exact 0 in r = 0!)
    dpsi_dr_in = -(omega_m / (r0_m**2)) * (r_safe_in / r0_m) * np.exp(-((r0_m**2 - r_safe_in**2) / (2.0 * (r0_m**2))))
    phi_ext = 1.0 + eta * (1.0 - np.log(r_safe_ext / r0_m))
    dpsi_dr_ext = -(omega_m / (r_safe_ext**2)) * phi_ext

    dpsi_dr = np.where(r_arr <= r0_m, dpsi_dr_in, dpsi_dr_ext)
    dtr_dr = -tr_tot * dpsi_dr

    return tr_tot, delta_tr, dtr_dr, psi_tot


def run_earth_gravity_tr_model(verbose=True, save_plot=True):
    """
    Genereert de numerieke tabel en de 3-paneels grafiek van het Time-Rhythm (TR)
    en de TR-gradiënt in en rond de Aarde (van r = 0 tot r = 12 * R_earth = 76 452 km).
    """
    r_max_mult = 12.0
    r_grid_m = np.linspace(0.0, r_max_mult * R_EARTH, 2400)
    u_grid = r_grid_m / R_EARTH  # Afstand in eenheden van aardstralen (r / R_earth)

    # 1. Volledig ETRT-profiel (eta = 1.0, met Tijd-Behouds Heuvelrug)
    tr_etrt, dtr_etrt, dtr_dr_etrt, psi_etrt = compute_earth_tr_profile(r_grid_m, eta=1.0)

    # 2. Klassiek monotoon profiel (eta = 0.0, zonder Tijd-Behouds Heuvelrug, en harmonische binnenput)
    _, dtr_no_hill, dtr_dr_no_hill, _ = compute_earth_tr_profile(r_grid_m, eta=0.0)
    # Klassieke homogene bol binnenin: delta_TR_class_in(r) = -Psi_surf * (1.5 - 0.5 * (r / R_earth)^2)
    dtr_classic_in = -PSI_SURFACE * (1.5 - 0.5 * (np.minimum(u_grid, 1.0) ** 2))
    dtr_classic_ext = -OMEGA_EARTH_M / np.maximum(r_grid_m, R_EARTH)
    dtr_classic = np.where(r_grid_m <= R_EARTH, dtr_classic_in, dtr_classic_ext)

    dtr_dr_classic_in = PSI_SURFACE * (np.minimum(r_grid_m, R_EARTH) / (R_EARTH**2))
    dtr_dr_classic_ext = OMEGA_EARTH_M / (np.maximum(r_grid_m, R_EARTH) ** 2)
    dtr_dr_classic = np.where(r_grid_m <= R_EARTH, dtr_dr_classic_in, dtr_dr_classic_ext)

    # Karakteristieke locaties in en rond de Aarde
    landmarks = [
        ("1. Middelpunt Aarde (r = 0)", 0.0),
        ("2. Binnenkern-grens (r = 1221 km)", 1221.0e3 / R_EARTH),
        ("3. Buitenkern-grens (r = 3480 km)", 3480.0e3 / R_EARTH),
        ("4. Aardoppervlak (r = R_earth)", 1.0),
        ("5. ISS / Lage Baan (h = 400 km)", (R_EARTH + 400.0e3) / R_EARTH),
        ("6. Euler-Omslagpunt (r = e·R_earth)", np.e),
        ("7. GPS-Satellietbaan (r = 26 560 km)", 26560.0e3 / R_EARTH),
        ("8. Geostationaire Baan (r = 42 164 km)", 42164.0e3 / R_EARTH),
        ("9. Top Heuvelrug (r = e²·R_earth)", np.e**2),
        ("10. Wijde Omgeving (r = 12·R_earth)", 12.0),
        ("11. Baan van de Maan (r = 60.3·R_earth)", 384400.0e3 / R_EARTH),
    ]

    if verbose:
        print("\n" + "=" * 112)
        print("ETRT ZWAARTEKRACHT IN EN ROND DE AARDE: TIJDRITME (TR) VAN MIDDELPUNT TOT WIJDE OMGEVING (earth_gravity_tr.py)")
        print("=" * 112)
        print(f"Aardstraal (R_earth = r_v):                 {R_EARTH/1e3:,.1f} km ({R_EARTH:.4e} m)")
        print(f"Gravitatie-spankracht (Omega_M = GM/c²):    {OMEGA_EARTH_M*1e3:.4f} mm ({OMEGA_EARTH_M:.4e} m)")
        print(f"Oppervlakte TR-reductie (Psi_oppervlak):    {PSI_SURFACE:.4e} (-{PSI_SURFACE*SECONDS_PER_DAY*1e6:.2f} μs/dag)")
        print(f"Euler Nulpunt (r_e = e·R_earth):            {np.e*R_EARTH/1e3:,.1f} km (hoogte h = {(np.e-1)*R_EARTH/1e3:,.1f} km -> ΔTR = 0)")
        print(f"Heuvelrug-Top (e²·R_earth):                 {(np.e**2)*R_EARTH/1e3:,.1f} km (hoogte h = {(np.e**2-1)*R_EARTH/1e3:,.1f} km -> dTR/dr = 0)")
        print("-" * 112)
        print(
            f"{'Locatie / Zone':<38} | {'r / R_earth':<11} | {'r [km]':<10} | "
            f"{'ΔTR ETRT (η=1)':<15} | {'Klok [μs/dag]':<14} | {'ΔTR Klassiek'}"
        )
        print("-" * 112)
        for name, u_val in landmarks:
            r_pt = u_val * R_EARTH
            _, dtr_pt, _, _ = compute_earth_tr_profile(r_pt, eta=1.0)
            if u_val <= 1.0:
                dtr_cl_pt = -PSI_SURFACE * (1.5 - 0.5 * (u_val**2))
            else:
                dtr_cl_pt = -PSI_SURFACE / u_val
            clock_us_day = float(dtr_pt) * SECONDS_PER_DAY * 1e6
            print(
                f"{name:<38} | {u_val:<11.4f} | {r_pt/1e3:<10.0f} | "
                f"{float(dtr_pt):<+15.4e} | {clock_us_day:<+14.2f} | {float(dtr_cl_pt):<+14.4e}"
            )
        print("=" * 112)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.5))

        # Schaalafwijking in eenheden van 10^-10 (zodat het oppervlak op -6.96 ligt en het centrum op -11.36)
        scale_1e10 = 1e10

        # --- PANEEL 1: VOLLEDIGE TR-GRAFIEK VAN MIDDELPUNT (r=0) TOT WIJDE OMGEVING (r=12 R_earth) ---
        ax0 = axes[0]
        ax0.axvspan(0.0, 1.0, color="#fee090", alpha=0.38, label="Binnen de Aarde (0 ≤ r ≤ R_⊕)")
        ax0.axvspan(1.0, np.e, color="#e0f3f8", alpha=0.40, label="Buiten-Put (R_⊕ < r < e·R_⊕: TR < 1)")
        ax0.axvspan(np.e, r_max_mult, color="#d9ef8b", alpha=0.28, label="Tijd-Behouds Heuvelrug (r > e·R_⊕: TR > 1)")

        ax0.plot(u_grid, dtr_etrt * scale_1e10, color="#1b9e77", lw=2.8, label="ETRT TR(r) met Heuvelrug (η = 1)")
        ax0.plot(u_grid, dtr_no_hill * scale_1e10, color="#2b5c8f", lw=2.0, ls="-.", label="ETRT Zonder Heuvelrug (η = 0)")
        ax0.plot(u_grid, dtr_classic * scale_1e10, color="#d95f02", lw=1.8, ls="--", label="Klassiek Newton/GR Profiel")

        ax0.axhline(0.0, color="black", lw=1.0, ls=":")
        ax0.axvline(1.0, color="#d95f02", lw=1.4, ls=":")
        ax0.axvline(np.e, color="#1b9e77", lw=1.2, ls="--", alpha=0.75)

        # Markeer sleutelpunten
        key_pts = [
            (0.0, float(dtr_etrt[0]) * scale_1e10, "Centrum r=0 (Platte bodem)\n(-9.70)", (0.45, -10.8)),
            (1.0, -PSI_SURFACE * scale_1e10, "Oppervlak R_⊕\n(-6.96)", (1.35, -7.8)),
            (np.e, 0.0, "r = e·R_⊕\n(TR = 1)", (np.e + 0.3, -1.6)),
            (np.e**2, float(compute_earth_tr_profile((np.e**2)*R_EARTH)[1]) * scale_1e10, "Top e²·R_⊕\n(+0.94)", (np.e**2 - 1.2, 1.35)),
        ]
        for x_p, y_p, txt_p, (xt, yt) in key_pts:
            ax0.plot(x_p, y_p, marker="o", markersize=7, color="gold", markeredgecolor="black", zorder=6)
            ax0.annotate(
                txt_p, xy=(x_p, y_p), xytext=(xt, yt),
                fontsize=7.8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
            )

        ax0.set_xlim(0.0, r_max_mult)
        ax0.set_ylim(-12.2, 2.4)
        ax0.set_title("1. TR-Profiel van Middelpunt (r=0) tot Wijde Omgeving")
        ax0.set_xlabel("Afstand vanaf Aardcentrum r / R_⊕ [Aardstralen]")
        ax0.set_ylabel("Tijdritme-afwijking (TR(r) - 1) × 10¹⁰")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower right", fontsize=7.6)

        # --- PANEEL 2: INGEZOOMD OP DE TIJD-BEHOUDS HEUVELRUG & SATELLIETBANEN (r = 1.5 .. 12 R_earth) ---
        ax1 = axes[1]
        mask_zoom = u_grid >= 1.5
        ax1.plot(u_grid[mask_zoom], dtr_etrt[mask_zoom] * scale_1e10, color="#1b9e77", lw=2.8, label="ETRT Tijd-Behouds Heuvelrug (TR > 1)")
        ax1.plot(u_grid[mask_zoom], dtr_classic[mask_zoom] * scale_1e10, color="#d95f02", lw=2.0, ls="--", label="Klassiek 1/r Profiel (blijft overal < 1)")
        ax1.fill_between(
            u_grid[mask_zoom], 0.0, dtr_etrt[mask_zoom] * scale_1e10,
            where=(dtr_etrt[mask_zoom] > 0), color="#1b9e77", alpha=0.18,
            label="Gecompenseerd Tijd-Volume (TR > 1)"
        )
        ax1.axhline(0.0, color="black", lw=1.0, ls=":")

        # Satelliet- en kritische banen
        orbits = [
            (np.e, "Euler-Nulpunt (e·R_⊕ = 17 318 km)", "#2b5c8f"),
            (26560.0e3 / R_EARTH, "GPS-Baan (4.17 R_⊕ = 26 560 km)", "#7570b3"),
            (42164.0e3 / R_EARTH, "Geostationair (6.62 R_⊕ = 42 164 km)", "#e7298a"),
            (np.e**2, "Heuvelrug-Top (e²·R_⊕ = 47 075 km)", "#d95f02"),
        ]
        for u_orb, lbl_orb, col_orb in orbits:
            y_orb = float(compute_earth_tr_profile(u_orb * R_EARTH, eta=1.0)[1]) * scale_1e10
            ax1.axvline(u_orb, color=col_orb, lw=1.3, ls=":", alpha=0.85)
            ax1.plot(u_orb, y_orb, marker="D", markersize=6.5, color=col_orb, markeredgecolor="black", zorder=5, label=lbl_orb)

        ax1.set_xlim(1.5, r_max_mult)
        ax1.set_ylim(-4.8, 1.35)
        ax1.set_title("2. Detail Buitenveld: De Tijd-Behouds Heuvelrug (TR > 1)")
        ax1.set_xlabel("Afstand vanaf Aardcentrum r / R_⊕ [Aardstralen]")
        ax1.set_ylabel("Tijdritme-afwijking (TR(r) - 1) × 10¹⁰")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=7.5)

        # --- PANEEL 3: DE TR-GRADIËNT dTR/dr & EFFECTIEVE ZWAARTEKRACHTSVERSNELLING g(r) ---
        ax2 = axes[2]
        # Effectieve gravitatie-versnelling g(r) = c^2 * (dTR/dr) [m/s^2] (positief = naar de Aarde gericht)
        g_etrt_ms2 = (C_LIGHT**2) * dtr_dr_etrt
        g_no_hill_ms2 = (C_LIGHT**2) * dtr_dr_no_hill
        g_classic_ms2 = (C_LIGHT**2) * dtr_dr_classic

        ax2.axvspan(0.0, 1.0, color="#fee090", alpha=0.35, label="Binnen Aarde (0 ≤ r ≤ R_⊕)")
        ax2.plot(u_grid, g_etrt_ms2, color="#1b9e77", lw=2.6, label="ETRT Gradiënt c²·dTR/dr (η = 1)")
        ax2.plot(u_grid, g_no_hill_ms2, color="#2b5c8f", lw=2.0, ls="-.", label="ETRT Zonder Heuvelrug (η = 0)")
        ax2.plot(u_grid, g_classic_ms2, color="#d95f02", lw=1.8, ls="--", label="Klassieke Gravitatie g(r) (Homogene bol)")

        ax2.axvline(1.0, color="#d95f02", lw=1.4, ls=":", label="Aardoppervlak R_⊕ (Piek g₀ = 9.82 m/s²)")
        ax2.axvline(np.e**2, color="black", lw=1.1, ls="--", alpha=0.7, label="r = e²·R_⊕ (Top Heuvelrug: dTR/dr = 0)")
        ax2.axhline(0.0, color="black", lw=0.9, ls=":")

        ax2.set_xlim(0.0, 9.0)
        ax2.set_ylim(-0.8, 20.5)
        ax2.set_title("3. Ruimtelijke TR-Gradiënt (c²·dTR/dr) van Kern tot Ruimte")
        ax2.set_xlabel("Afstand vanaf Aardcentrum r / R_⊕ [Aardstralen]")
        ax2.set_ylabel("Gradiënt-versnelling c² · dTR/dr [m/s²]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=7.6)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.24)
        plt.savefig("earth_tr_profile.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'earth_tr_profile.png'.")

    return {
        "R_earth_m": R_EARTH,
        "Omega_earth_m": OMEGA_EARTH_M,
        "psi_surface": PSI_SURFACE,
        "r_euler_km": (np.e * R_EARTH) / 1e3,
        "r_hill_top_km": ((np.e**2) * R_EARTH) / 1e3,
    }


if __name__ == "__main__":
    run_earth_gravity_tr_model()