import numpy as np
import matplotlib.pyplot as plt

# Natuurconstanten en astronomische parameters (SI)
G_NEWTON = 6.67430e-11          # Gravitatieconstante [m^3 kg^-1 s^-2]
C_LIGHT = 299792458.0           # Lichtsnelheid [m/s]
SECONDS_PER_DAY = 86400.0       # Seconden per dag [s]

# Zon, Aarde en Maan: Massa's [kg] en Stralen r_v [m]
M_SUN, R_SUN = 1.98847e30, 6.9634e8          # Straal Zon: 696 340 km
M_EARTH, R_EARTH = 5.97219e24, 6.3710e6      # Straal Aarde: 6 371 km
M_MOON, R_MOON = 7.342e22, 1.7374e6          # Straal Maan: 1 737.4 km

# Gemiddelde baanafstanden [m]
AU_M = 1.495978707e11                        # 1 AU (Zon - Aarde): 149 597 870.7 km
D_EARTH_MOON = 3.84400e8                     # Aarde - Maan afstand: 384 400 km

# ETRT Gravitatie-spankrachten Omega_i = G * M_i / c^2 [m]
OMEGA_SUN = (G_NEWTON * M_SUN) / (C_LIGHT**2)      # ~1476.67 m
OMEGA_EARTH = (G_NEWTON * M_EARTH) / (C_LIGHT**2)  # ~4.4350e-3 m (4.435 mm)
OMEGA_MOON = (G_NEWTON * M_MOON) / (C_LIGHT**2)    # ~5.4523e-5 m (0.0545 mm)


def body_psi_profile(r_dist_m, r0_m, omega_m, eta=0.0):
    """
    Berekent de ETRT rooster-stress Psi(r) binnen én buiten een bolvormig hemellichaam:
    - Binnen (r <= r0): Psi_in(r) = (Omega / r0) * [2 - exp(-(r0 - r) / r0)]
                        (uit 'TR_within_objects.pdf' Vgl. 43-45)
    - Buiten (r > r0):  Psi_ext(r) = (Omega / r) * [1 - eta * ln(r / r0)]
                        (uit 'massfield.py' & 'Technisch Addendum Macro-schaal Gravitatie')
    """
    r_arr = np.asarray(r_dist_m, dtype=float)
    r_in = np.minimum(r_arr, r0_m)
    r_ext = np.maximum(r_arr, r0_m)

    psi_in = (omega_m / r0_m) * (2.0 - np.exp(-((r0_m**2 - r_in**2) / (2.0 * (r0_m**2)))))
    psi_ext = (omega_m / r_ext) * (1.0 - eta * np.log(r_ext / r0_m))
    return np.where(r_arr <= r0_m, psi_in, psi_ext)


def superposed_tr_field_1d(x_from_earth_m, moon_sign=+1.0, eta=0.0, tr_l=1.0):
    """
    Berekent langs de verbindingslijn Zon - Aarde - Maan de totale rooster-stress
    Psi_tot(x) = Psi_Zon(x) + Psi_Aarde(x) + Psi_Maan(x)
    en het resulterende Tijdritme TR(x) = TR_L * exp(-Psi_tot(x)) (volgens massfield.py).

    Coördinatenstelsel:
      - x_from_earth_m = 0:         Middelpunt van de Aarde (op afstand AU_M van de Zon)
      - x_from_earth_m < 0:         Richting de Zon (Zon staat op x = -AU_M)
      - moon_sign = -1.0:           Nieuwe Maan (Maan tussen Zon en Aarde op x = -D_EARTH_MOON)
      - moon_sign = +1.0:           Volle Maan (Maan achter de Aarde op x = +D_EARTH_MOON)
    """
    x_arr = np.asarray(x_from_earth_m, dtype=float)
    r_sun = np.abs(AU_M + x_arr)
    r_earth = np.abs(x_arr)
    x_moon = moon_sign * D_EARTH_MOON
    r_moon = np.abs(x_arr - x_moon)

    psi_sun = body_psi_profile(r_sun, R_SUN, OMEGA_SUN, eta=eta)
    psi_earth = body_psi_profile(r_earth, R_EARTH, OMEGA_EARTH, eta=eta)
    psi_moon = body_psi_profile(r_moon, R_MOON, OMEGA_MOON, eta=eta)

    psi_tot = psi_sun + psi_earth + psi_moon
    tr_tot = tr_l * np.exp(-psi_tot)
    return tr_tot, psi_tot, psi_sun, psi_earth, psi_moon


def run_sun_earth_moon_superposition(verbose=True, save_plot=True):
    """
    Analyseert en visualiseert de superpositie van de TR-velden van Zon, Aarde en Maan:
      1. De Aarde- en Maan-kuilen op de grootschalige TR-helling van de Zon (rond 1 AU).
      2. Het gecombineerde Aarde-Maan profiel (met het zadelpunt / L1-overgangspunt)
         voor zowel het basisveld (eta = 0) als met een subtiele Tijd-Behouds Heuvelrug.
      3. Het lokale TR-profiel door het binnenste van de Maan bij Nieuwe Maan vs Volle Maan
         én het getijden-effect op het aardoppervlak.
    """
    # 1. Bereken de exacte bijdragen op sleutelposities langs de Zon-Aarde-Maan as (bij Volle Maan, x_Maan = +384 400 km)
    psi_sun_at_earth = float(body_psi_profile(AU_M, R_SUN, OMEGA_SUN, eta=0.0))
    psi_sun_at_new_moon = float(body_psi_profile(AU_M - D_EARTH_MOON, R_SUN, OMEGA_SUN, eta=0.0))
    psi_sun_at_full_moon = float(body_psi_profile(AU_M + D_EARTH_MOON, R_SUN, OMEGA_SUN, eta=0.0))

    # Zonne-helling verschil over de baan van de Maan (tussen Nieuwe Maan en Volle Maan):
    delta_tr_sun_across_moon_orbit = psi_sun_at_new_moon - psi_sun_at_full_moon

    # Statisch zadelpunt tussen Aarde en Maan (waar |dPsi_Aarde/dx| = |dPsi_Maan/dx|):
    x_saddle_em_m = D_EARTH_MOON / (1.0 + np.sqrt(M_MOON / M_EARTH))
    d_saddle_from_moon_km = (D_EARTH_MOON - x_saddle_em_m) / 1e3

    # Dynamisch Lagrange L1-punt (inclusief centrifugale baanbeweging van het Aarde-Maan systeem):
    mu_em = M_MOON / (M_EARTH + M_MOON)
    x_L1_em_m = D_EARTH_MOON * (1.0 - (mu_em / 3.0) ** (1.0 / 3.0))

    # Getijden-verschil in TR op het aardoppervlak (sub-lunair punt vs antipodaal punt vs kwadratuur):
    # De lineaire gradiënt geeft de vrije-val versnelling naar de Maan; het tweede-orde (quadrupool) verschil
    # geeft de getijden-uitrekking van het TR-veld op het aardoppervlak:
    delta_tr_tidal_moon = 2.0 * OMEGA_MOON * (R_EARTH**2) / (D_EARTH_MOON**3)
    delta_tr_tidal_sun = 2.0 * OMEGA_SUN * (R_EARTH**2) / (AU_M**3)
    tidal_ratio_moon_sun = delta_tr_tidal_moon / delta_tr_tidal_sun  # ~2.18x (Maan-getij is 2.2x sterker dan Zonne-getij!)

    if verbose:
        print("\n" + "=" * 114)
        print("ETRT SUPERPOSITIE VAN ZON, AARDE EN MAAN: TIJDRITME (TR) OP DE ZONNE-HELLING (sun_earth_moon_tr.py)")
        print("=" * 114)
        print(f"1. Gravitatie-spankrachten (Omega = GM/c²):  Zon = {OMEGA_SUN:,.2f} m | Aarde = {OMEGA_EARTH*1e3:.4f} mm | Maan = {OMEGA_MOON*1e3:.4f} mm")
        print(f"2. Zonne-achtergrondput op 1 AU (Psi_Zon):   {psi_sun_at_earth:.6e} (-{psi_sun_at_earth*SECONDS_PER_DAY*1e6:.2f} μs/dag, 14.2x dieper dan Aardoppervlak!)")
        print(f"3. Zonne-helling over Maanbaan (Nieuw->Vol): ΔTR_Zon = {delta_tr_sun_across_moon_orbit:+.4e} ({delta_tr_sun_across_moon_orbit*SECONDS_PER_DAY*1e6:+.3f} μs/dag over ±384 400 km)")
        print(f"4. Aarde-Maan TR-Zadelpunt (dTR_EM/dx = 0):  x = {x_saddle_em_m/1e3:,.0f} km vanaf Aarde ({d_saddle_from_moon_km:,.0f} km vóór de Maan; dynamisch L1 = {x_L1_em_m/1e3:,.0f} km)")
        print(f"5. Getijden-TR op Aardoppervlak (Maan/Zon):  ΔTR_getij(Maan) = {delta_tr_tidal_moon:.3e} | ΔTR_getij(Zon) = {delta_tr_tidal_sun:.3e} (Ratio Maan/Zon = {tidal_ratio_moon_sun:.2f}x)")
        print("-" * 114)
        print(
            f"{'Locatie langs Zon-Aarde-Maan As (Volle Maan)':<42} | {'x [km]':<11} | "
            f"{'ΔTR_Zon':<13} | {'ΔTR_Aarde':<13} | {'ΔTR_Maan':<13} | {'ΔTR_Totaal':<13} | {'Klok [μs/d]'}"
        )
        print("-" * 114)

        sample_points = [
            ("1. Binnenbaan (-400 000 km richting Zon)", -400000.0e3),
            ("2. Positie Nieuwe Maan (-384 400 km)", -D_EARTH_MOON),
            ("3. Aardoppervlak Dagzijde (-6 371 km)", -R_EARTH),
            ("4. Middelpunt Aarde (x = 0 km)", 0.0),
            ("5. Aardoppervlak Nachtzijde (+6 371 km)", R_EARTH),
            ("6. Geostationaire Baan (+42 164 km)", 42164.0e3),
            ("7. Aarde-Maan TR-Zadelpunt (+346 008 km)", x_saddle_em_m),
            ("8. Maanoppervlak Voorzijde (+382 663 km)", D_EARTH_MOON - R_MOON),
            ("9. Middelpunt Volle Maan (+384 400 km)", D_EARTH_MOON),
            ("10. Maanoppervlak Achterzijde (+386 137 km)", D_EARTH_MOON + R_MOON),
        ]

        for lbl, x_pt in sample_points:
            tr_t, psi_t, p_s, p_e, p_m = superposed_tr_field_1d(x_pt, moon_sign=+1.0, eta=0.0)
            dtr_tot = float(tr_t - 1.0)
            clock_us = dtr_tot * SECONDS_PER_DAY * 1e6
            print(
                f"{lbl:<42} | {x_pt/1e3:<+11.0f} | "
                f"{-float(p_s):<+13.4e} | {-float(p_e):<+13.4e} | {-float(p_m):<+13.4e} | "
                f"{dtr_tot:<+13.4e} | {clock_us:<+11.2f}"
            )
        print("=" * 114)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.5))
        scale_1e9 = 1e9   # Schaal in eenheden van 10^-9 voor Paneel 1 (Zon + Aarde + Maan)
        scale_1e11 = 1e11 # Schaal in eenheden van 10^-11 voor Paneel 2 & 3 (Aarde-Maan & Maan-detail)

        # --- PANEEL 1: DE AARDE- EN MAAN-KUILEN OP DE UITGESTREKTE TR-HELLING VAN DE ZON ---
        ax0 = axes[0]
        x_km_wide = np.linspace(-500000.0, 500000.0, 4000)
        x_m_wide = x_km_wide * 1e3

        tr_full, _, psi_s_w, _, _ = superposed_tr_field_1d(x_m_wide, moon_sign=+1.0, eta=0.0)
        tr_new, _, _, _, _ = superposed_tr_field_1d(x_m_wide, moon_sign=-1.0, eta=0.0)
        tr_sun_only = np.exp(-psi_s_w)

        ax0.plot(
            x_km_wide / 1e3, (tr_sun_only - 1.0) * scale_1e9,
            color="#d95f02", lw=2.2, ls="--", label="Zonne-Helling alleen: ΔTR_Zon(x)"
        )
        ax0.plot(
            x_km_wide / 1e3, (tr_full - 1.0) * scale_1e9,
            color="#2b5c8f", lw=2.5, label="Totaal TR-veld bij Volle Maan (+384 400 km)"
        )
        ax0.plot(
            x_km_wide / 1e3, (tr_new - 1.0) * scale_1e9,
            color="#1b9e77", lw=2.0, ls="-.", label="Totaal TR-veld bij Nieuwe Maan (-384 400 km)"
        )

        # Markeer de Aarde op x = 0 en de twee Maan-posities
        ax0.axvline(0.0, color="black", lw=1.0, ls=":", alpha=0.7)
        ax0.annotate(
            "Aarde-Kuil (x = 0)\nCentrum: -11.01 × 10⁻⁹",
            xy=(0.0, -11.007), xytext=(70.0, -10.92),
            fontsize=8.0, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9)
        )
        ax0.annotate(
            "Nieuwe Maan\n(-384 400 km)",
            xy=(-384.4, (float(superposed_tr_field_1d(-D_EARTH_MOON, moon_sign=-1.0)[0]) - 1.0) * scale_1e9),
            xytext=(-460.0, -10.12),
            fontsize=7.8, fontweight="bold", color="#1b9e77",
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=0.9)
        )
        ax0.annotate(
            "Volle Maan\n(+384 400 km)",
            xy=(384.4, (float(superposed_tr_field_1d(D_EARTH_MOON, moon_sign=+1.0)[0]) - 1.0) * scale_1e9),
            xytext=(210.0, -10.18),
            fontsize=7.8, fontweight="bold", color="#2b5c8f",
            arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=0.9)
        )

        ax0.set_xlim(-500.0, 500.0)
        ax0.set_ylim(-11.08, -9.80)
        ax0.set_title("1. Aarde- & Maan-Kuilen op de Zonne-Helling (1 AU)")
        ax0.set_xlabel("Afstand vanaf Aarde langs Zon-Aarde-as [× 1000 km]")
        ax0.set_ylabel("Totale TR-afwijking (TR(x) - 1) × 10⁹")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="lower left", fontsize=7.6)

        # --- PANEEL 2: HET AARDE-MAAN SYSTEEM EN HET ZADELPUNT (TUSSEN AARDE EN MAAN) ---
        ax1 = axes[1]
        x_km_em = np.linspace(30000.0, 430000.0, 3000)
        x_m_em = x_km_em * 1e3

        # Lokale Aarde + Maan bijdrage (in het vrij-vallende stelsel waarin de lineaire zonne-helling gecompenseerd is)
        psi_e_em = body_psi_profile(x_m_em, R_EARTH, OMEGA_EARTH, eta=0.0)
        psi_m_em = body_psi_profile(np.abs(x_m_em - D_EARTH_MOON), R_MOON, OMEGA_MOON, eta=0.0)
        dtr_em_base = -(psi_e_em + psi_m_em)

        # Vergelijk ook met een subtiele macroscopische heuvelrug-fractie (eta = 0.15)
        psi_e_hill = body_psi_profile(x_m_em, R_EARTH, OMEGA_EARTH, eta=0.15)
        psi_m_hill = body_psi_profile(np.abs(x_m_em - D_EARTH_MOON), R_MOON, OMEGA_MOON, eta=0.15)
        dtr_em_hill = -(psi_e_hill + psi_m_hill)

        ax1.plot(x_km_em / 1e3, -psi_e_em * scale_1e11, color="#d95f02", lw=1.8, ls="--", label="Alleen Aarde-put ΔTR_⊕(x)")
        ax1.plot(x_km_em / 1e3, dtr_em_base * scale_1e11, color="#2b5c8f", lw=2.6, label="Superpositie Aarde + Maan (η = 0)")
        ax1.plot(x_km_em / 1e3, dtr_em_hill * scale_1e11, color="#1b9e77", lw=2.2, ls="-.", label="Superpositie met Heuvelrug-fractie (η = 0.15)")

        # Markeer het zadelpunt op x = 346 008 km en de Maan op x = 384 400 km
        y_saddle = float(-(body_psi_profile(x_saddle_em_m, R_EARTH, OMEGA_EARTH) + body_psi_profile(D_EARTH_MOON - x_saddle_em_m, R_MOON, OMEGA_MOON))) * scale_1e11
        ax1.plot(x_saddle_em_m / 1e6, y_saddle, marker="o", markersize=7.5, color="gold", markeredgecolor="black", zorder=6)
        ax1.axvline(x_saddle_em_m / 1e6, color="gray", lw=1.1, ls=":")
        ax1.annotate(
            f"TR-Zadelpunt (dTR/dx = 0)\nx = {x_saddle_em_m/1e3:,.0f} km",
            xy=(x_saddle_em_m / 1e6, y_saddle), xytext=(150.0, -4.5),
            fontsize=8.0, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9)
        )
        ax1.axvline(D_EARTH_MOON / 1e6, color="#7570b3", lw=1.2, ls=":", label="Maan-centrum (384 400 km)")

        ax1.set_xlim(30.0, 430.0)
        ax1.set_ylim(-12.0, 1.0)
        ax1.set_title("2. Aarde-Maan Superpositie & TR-Zadelpunt (Naar de Maan)")
        ax1.set_xlabel("Afstand vanaf Aarde richting Maan [× 1000 km]")
        ax1.set_ylabel("Lokale Aarde+Maan ΔTR(x) × 10¹¹")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower left", fontsize=7.6)

        # --- PANEEL 3: INGEZOOMD OP DE MAAN-PUT (INWENDIG & UITWENDIG) BIJ NIEUWE VS VOLLE MAAN ---
        ax2 = axes[2]
        dx_moon_km = np.linspace(-10000.0, 10000.0, 1200)
        dx_moon_m = dx_moon_km * 1e3
        r_from_moon_m = np.abs(dx_moon_m)

        # Lokale Maan-put + Aarde-achtergrond op 384 400 km + Zonne-helling verschil tussen Nieuwe en Volle Maan
        psi_m_loc = body_psi_profile(r_from_moon_m, R_MOON, OMEGA_MOON, eta=0.0)
        psi_e_near_moon = body_psi_profile(D_EARTH_MOON + dx_moon_m, R_EARTH, OMEGA_EARTH, eta=0.0)

        # Verschil t.o.v. het gemiddelde zonne-niveau op 1 AU:
        dtr_moon_local_em = -(psi_e_near_moon + psi_m_loc) * scale_1e11
        # Bij Nieuwe Maan ligt de Maan 384 400 km dichter bij de Zon (-2.53e-11 extra TR-verlaging),
        # bij Volle Maan 384 400 km verder van de Zon (+2.53e-11 hogere TR):
        dtr_sun_shift_full = +(psi_sun_at_earth - body_psi_profile(AU_M + D_EARTH_MOON + dx_moon_m, R_SUN, OMEGA_SUN)) * scale_1e11
        dtr_sun_shift_new = +(psi_sun_at_earth - body_psi_profile(AU_M - D_EARTH_MOON + dx_moon_m, R_SUN, OMEGA_SUN)) * scale_1e11

        ax2.axvspan(-R_MOON / 1e3, R_MOON / 1e3, color="#fee090", alpha=0.40, label=f"Binnen de Maan (|Δx| ≤ {R_MOON/1e3:.0f} km)")
        ax2.plot(dx_moon_km, dtr_moon_local_em + dtr_sun_shift_full, color="#2b5c8f", lw=2.5, label="Volle Maan (Verder van Zon: Hoger op Helling)")
        ax2.plot(dx_moon_km, dtr_moon_local_em, color="black", lw=2.0, ls=":", label="Kwarter-Maan (Op 1 AU Zonne-niveau)")
        ax2.plot(dx_moon_km, dtr_moon_local_em + dtr_sun_shift_new, color="#1b9e77", lw=2.5, ls="-.", label="Nieuwe Maan (Dichter bij Zon: Dieper in Put)")

        ax2.set_xlim(-10000.0, 10000.0)
        ax2.set_ylim(-9.5, 2.2)
        ax2.set_title("3. In- en Uitwendige Maan-Put op de Zonne-Helling")
        ax2.set_xlabel("Afstand vanaf Maancentrum Δx [km]")
        ax2.set_ylabel("ΔTR(Δx) t.o.v. 1 AU Zonne-basis × 10¹¹")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower right", fontsize=7.5)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.25)
        plt.savefig("sun_earth_moon_superposition.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'sun_earth_moon_superposition.png'.")

    return {
        "psi_sun_1AU": psi_sun_at_earth,
        "delta_tr_sun_moon_orbit": delta_tr_sun_across_moon_orbit,
        "x_saddle_em_km": x_saddle_em_m / 1e3,
        "x_L1_em_km": x_L1_em_m / 1e3,
        "tidal_ratio_moon_sun": tidal_ratio_moon_sun,
    }


if __name__ == "__main__":
    run_sun_earth_moon_superposition()