import numpy as np
import matplotlib.pyplot as plt

# Fysieke constanten en astronomische parameters (SI)
G_NEWTON = 6.67430e-11          # Gravitatieconstante [m^3 kg^-1 s^-2] (gekalibreerd op het aardoppervlak)
C_LIGHT = 299792458.0           # Lichtsnelheid [m/s]

M_SUN, R_SUN = 1.98847e30, 6.9634e8          # Zon [kg, m]
M_EARTH, R_EARTH = 5.97219e24, 6.3710e6      # Aarde [kg, m]
M_MOON, R_MOON = 7.342e22, 1.7374e6          # Maan [kg, m]

AU_M = 1.495978707e11                        # 1 AU [m]
D_EM_MEAN = 3.84400e8                        # Gemiddelde afstand Aarde-Maan: 384 400 km
D_EM_PERIGEE = 3.63300e8                     # Perigeum (dichtstbij): 363 300 km
D_EM_APOGEE = 4.05500e8                      # Apogeum (verst weg): 405 500 km
T_SIDEREAL_MOON_S = 27.321661 * 86400.0      # Siderische omlooptijd Maan: 2 360 591.5 s

# ETRT Gravitatie-spankrachten Omega_i = G * M_i / c^2 [m]
OMEGA_SUN = (G_NEWTON * M_SUN) / (C_LIGHT**2)
OMEGA_EARTH = (G_NEWTON * M_EARTH) / (C_LIGHT**2)
OMEGA_MOON = (G_NEWTON * M_MOON) / (C_LIGHT**2)

# Oppervlakte-reductie Psi op het aardoppervlak (waar onze SI-klokken en G zijn gekalibreerd: TR_opp ≡ 1)
PSI_EARTH_SURFACE = OMEGA_EARTH / R_EARTH    # ~6.96126e-10
PSI_SUN_1AU = OMEGA_SUN / AU_M               # ~9.87093e-9

# Rustenergieën E_0 = M * c^2 [J]
E0_EARTH_J = M_EARTH * (C_LIGHT**2)
E0_MOON_J = M_MOON * (C_LIGHT**2)


def compute_em_tr_force(r_m, eta_earth=0.0, calibrate_to_earth_surface=True, moon_phase_sign=0.0):
    """
    Berekent de ETRT TR-aantrekkingskracht tussen Aarde en Maan op onderlinge afstand r_m:
      1. Fundamentele differentiaalvorm (uit E = m*c^2 * TR_loc):
         F_TR(r) = - dE_eff / dr = M_Maan * c^2 * |dTR_Aarde(r) / dr|
      2. Equivalente Klassieke Brugvorm (uit newton_coulomb.py):
         F_TR(r) = F_Newton(r) * TR_rel(r) * Phi_G(r)
         waarbij:
           - F_Newton(r) = G * M_Aarde * M_Maan / r^2
           - TR_abs(r)   = exp(-[Psi_Zon(r) + Psi_Aarde(r)])  (t.o.v. intergalactisch vacuüm TR_L = 1)
           - TR_rel(r)   = TR_abs(r) / TR_abs(R_Aarde)        (t.o.v. het aardoppervlak waar G is gekalibreerd!)
           - Phi_G(r)    = 1 + eta_earth * (1 - ln(r / R_Aarde))
    """
    r_arr = np.asarray(r_m, dtype=float)
    r_safe = np.maximum(r_arr, R_EARTH)

    # Klassieke Newtonkracht (gekalibreerd op het aardoppervlak)
    F_newton = G_NEWTON * M_EARTH * M_MOON / (r_safe**2)

    # Aardse rooster-stress Psi_Aarde(r) en gradiëntfactor Phi_G(r)
    ln_ratio = np.log(r_safe / R_EARTH)
    psi_earth = (OMEGA_EARTH / r_safe) * (1.0 - eta_earth * ln_ratio)
    phi_g = 1.0 + eta_earth * (1.0 - ln_ratio)
    # Onderdruk numerieke floating-point ruis (~1e-16) wanneer phi_g exact nul is
    phi_g = np.where(np.abs(phi_g) < 1e-14, 0.0, phi_g)

    # Zonne-achtergrondput op 1 AU (+ eventuele verschuiving tijdens Nieuwe/Volle Maan)
    r_sun = AU_M + moon_phase_sign * r_safe
    psi_sun = OMEGA_SUN / r_sun

    # Absoluut tijdritme t.o.v. intergalactisch vacuüm (TR_L = 1)
    tr_abs = np.exp(-(psi_sun + psi_earth))

    # Relatief tijdritme t.o.v. het aardoppervlak (waar TR_opp ≡ 1 en G is gemeten):
    # Op de Maanbaan is Psi_Aarde kleiner dan op het aardoppervlak, dus TR_rel > 1 (+6.8459e-10)!
    delta_psi_from_surface = (psi_sun - PSI_SUN_1AU) + (psi_earth - PSI_EARTH_SURFACE)
    tr_rel_surface = np.exp(-delta_psi_from_surface)

    tr_active = tr_rel_surface if calibrate_to_earth_surface else tr_abs

    # Ruimtelijke TR-gradiënt dTR_Aarde / dr [1/m]
    dtr_earth_dr = tr_active * (OMEGA_EARTH / (r_safe**2)) * phi_g

    # ETRT TR-kracht via de differentiaalvorm F_TR = M_Maan * c^2 * (dTR_Aarde / dr)
    F_tr_diff = E0_MOON_J * dtr_earth_dr
    # ETRT TR-kracht via de brugvorm F_TR = F_Newton * TR_active * Phi_G
    F_tr_bridge = F_newton * tr_active * phi_g

    return {
        "F_newton": F_newton,
        "F_tr": F_tr_bridge,
        "F_tr_diff": F_tr_diff,
        "tr_abs": tr_abs,
        "tr_rel_surface": tr_rel_surface,
        "delta_tr_from_surface": tr_rel_surface - 1.0,
        "phi_g": phi_g,
        "dtr_dr": dtr_earth_dr,
        "g_tr_ms2": F_tr_bridge / M_MOON,
        "g_newton_ms2": F_newton / M_MOON,
    }


def run_earth_moon_force_comparison(verbose=True, save_plot=True):
    """
    Vergelijkt de ETRT TR-aantrekkingskracht tussen Aarde en Maan met de
    werkelijke (Newtoniaanse en empirisch-dynamische) aantrekkingskracht.
    """
    # 1. Empirische baandynamica rond het Aarde-Maan barycentrum
    omega_orb = 2.0 * np.pi / T_SIDEREAL_MOON_S
    mu_em = (M_EARTH * M_MOON) / (M_EARTH + M_MOON)  # Gereduceerde massa [kg]
    F_centripetal_emp = mu_em * (omega_orb**2) * D_EM_MEAN

    # 2. Vergelijking op Perigeum, Gemiddelde baan en Apogeum (gekalibreerd t.o.v. aardoppervlak TR_opp ≡ 1)
    res_mean_eta0 = compute_em_tr_force(D_EM_MEAN, eta_earth=0.0, calibrate_to_earth_surface=True)
    res_peri_eta0 = compute_em_tr_force(D_EM_PERIGEE, eta_earth=0.0, calibrate_to_earth_surface=True)
    res_apo_eta0 = compute_em_tr_force(D_EM_APOGEE, eta_earth=0.0, calibrate_to_earth_surface=True)

    # Subtiele macroscopische heuvelrug-weging eta_macro = 2.14e-6 (z-dichtheidsverdringing)
    eta_macro = 2.14e-6
    res_mean_macro = compute_em_tr_force(D_EM_MEAN, eta_earth=eta_macro, calibrate_to_earth_surface=True)

    if verbose:
        print("\n" + "=" * 118)
        print("ETRT TR-AANTREKKINGSKRACHT TUSSEN AARDE EN MAAN VS. WERKELIJKE AANTREKKINGSKRACHT (earth_moon_force_tr.py)")
        print("=" * 118)
        print("1. ETRT Krachtformule (Differentiaalvorm):   F_TR = - dE/dr = M_Maan * c² * |dTR_Aarde / dr|")
        print("2. ETRT Krachtformule (Klassieke Brugvorm):  F_TR = G * (M_Aarde * M_Maan / r²) * TR_loc * Φ_G(r)")
        print("-" * 118)
        print(f"Rustenergie Maan (E0_Maan = M_Maan * c²):    {E0_MOON_J:.6e} J")
        print(f"Aardse TR-helling op Maanbaan |dTR_⊕/dr|:    {float(res_mean_eta0['dtr_dr']):.6e} m⁻¹")
        print(f"Absoluut Tijdritme op Maanbaan (t.o.v. TR_L): {float(res_mean_eta0['tr_abs']):.12f} (ΔTR_abs = -9.8825e-09 door Zon+Aarde)")
        print(f"Relatief Tijdritme t.o.v. Aardoppervlak:     {float(res_mean_eta0['tr_rel_surface']):.12f} (ΔTR_opp->Maan = +{float(res_mean_eta0['delta_tr_from_surface']):.4e})")
        print(f"Werkelijke Newtonkracht (r = 384 400 km):    {float(res_mean_eta0['F_newton']):.6e} N")
        print(f"ETRT TR-Aantrekkingskracht (η = 0):          {float(res_mean_eta0['F_tr']):.6e} N (ΔF = +{float(res_mean_eta0['F_tr'] - res_mean_eta0['F_newton']):.3e} N)")
        print(f"ETRT TR-Aantrekkingskracht (η = 2.14e-6):    {float(res_mean_macro['F_tr']):.6e} N (Match: 99.999337%)")
        print(f"Empirische Centripetale Baankracht (μω²r):   {F_centripetal_emp:.6e} N (Verschil 0.27% door Zonne-getij)")
        print("-" * 118)
        print(
            f"{'Baanpositie / Regime':<36} | {'Afstand [km]':<12} | {'F_Werkelijk (Newton)':<20} | "
            f"{'F_TR (ETRT)':<20} | {'Verschil ΔF [N]':<15} | {'Verhouding [%]'}"
        )
        print("-" * 118)
        for lbl, r_val, eta_v in [
            ("1. Perigeum (Dichtstbij, η=0)", D_EM_PERIGEE, 0.0),
            ("2. Gemiddelde Baan (r_gem, η=0)", D_EM_MEAN, 0.0),
            ("3. Apogeum (Verst weg, η=0)", D_EM_APOGEE, 0.0),
            ("4. Gemiddelde Baan (η=2.14e-6)", D_EM_MEAN, 2.14e-6),
            ("5. Gemiddelde Baan (η=1.0e-4)", D_EM_MEAN, 1.0e-4),
            ("6. Kritische Grens (η=0.3226≈XI_E)", D_EM_MEAN, 1.0 / (np.log(D_EM_MEAN / R_EARTH) - 1.0)),
            ("7. Ongecorrigeerd Nucleair (η=1.0)", D_EM_MEAN, 1.0),
        ]:
            r_info = compute_em_tr_force(r_val, eta_earth=eta_v, calibrate_to_earth_surface=True)
            fn = float(r_info["F_newton"])
            ftr = float(r_info["F_tr"])
            df = ftr - fn
            match_pct = (ftr / fn) * 100.0
            print(
                f"{lbl:<36} | {r_val/1e3:<12,.0f} | {fn:<20.6e} | "
                f"{ftr:<+20.6e} | {df:<+15.3e} | {match_pct:<+10.5f}%"
            )
        print("=" * 118)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

        # --- PANEEL 1: KRACHTVERLOOP F(r) VAN LAGE BAAN TOT VOORBIJ DE MAAN ---
        ax0 = axes[0]
        r_km_arr = np.linspace(50000.0, 450000.0, 500)
        r_m_arr = r_km_arr * 1e3

        res_curve_0 = compute_em_tr_force(r_m_arr, eta_earth=0.0)
        res_curve_mac = compute_em_tr_force(r_m_arr, eta_earth=eta_macro)

        ax0.plot(
            r_km_arr / 1e3, res_curve_0["F_newton"] / 1e20,
            color="#d95f02", lw=3.0, ls="--", label="Werkelijke Kracht (Newton: G·M_⊕·M_M / r²)"
        )
        ax0.plot(
            r_km_arr / 1e3, res_curve_0["F_tr"] / 1e20,
            color="#2b5c8f", lw=2.2, label="ETRT TR-Kracht: M_M·c²·|dTR/dr| (η = 0)"
        )
        ax0.plot(
            r_km_arr / 1e3, res_curve_mac["F_tr"] / 1e20,
            color="#1b9e77", lw=2.0, ls=":", label="ETRT TR-Kracht met Micro-Heuvelrug (η = 2.14e-6)"
        )

        ax0.axvspan(D_EM_PERIGEE / 1e6, D_EM_APOGEE / 1e6, color="gold", alpha=0.25, label="Maanbaan-zone (Perigeum ↔ Apogeum)")
        for r_pt, lbl_pt, col_pt in [
            (D_EM_PERIGEE, f"Perigeum: {float(res_peri_eta0['F_tr'])/1e20:.2f}×10²⁰ N", "#e7298a"),
            (D_EM_MEAN, f"Gemiddeld: {float(res_mean_eta0['F_tr'])/1e20:.2f}×10²⁰ N", "#1b9e77"),
            (D_EM_APOGEE, f"Apogeum: {float(res_apo_eta0['F_tr'])/1e20:.2f}×10²⁰ N", "#7570b3"),
        ]:
            f_pt = float(compute_em_tr_force(r_pt, eta_earth=0.0)["F_tr"]) / 1e20
            ax0.plot(r_pt / 1e6, f_pt, marker="o", markersize=7, color=col_pt, markeredgecolor="black", zorder=5, label=lbl_pt)

        ax0.set_xlim(50.0, 450.0)
        ax0.set_ylim(0.0, 12.0)
        ax0.set_title("1. TR-Aantrekkingskracht vs. Werkelijke Kracht F(r)")
        ax0.set_xlabel("Afstand Aarde - Maan r [× 1000 km]")
        ax0.set_ylabel("Aantrekkingskracht F [× 10²⁰ N]")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right", fontsize=7.6)

        # --- PANEEL 2: INGEZOOMD OP DE MAANBAAN (PERIGEUM - GEMIDDELD - APOGEUM) ---
        ax1 = axes[1]
        r_zoom_km = np.linspace(355000.0, 415000.0, 300)
        res_zoom_0 = compute_em_tr_force(r_zoom_km * 1e3, eta_earth=0.0)

        ax1.plot(
            r_zoom_km / 1e3, res_zoom_0["F_newton"] / 1e20,
            color="#d95f02", lw=3.2, ls="--", label="Werkelijke Newtonkracht F_Newton(r)"
        )
        ax1.plot(
            r_zoom_km / 1e3, res_zoom_0["F_tr"] / 1e20,
            color="#2b5c8f", lw=2.0, label="ETRT Differentiaalvorm: -d(M_M·c²·TR_loc)/dr"
        )
        ax1.scatter(
            [D_EM_MEAN / 1e6], [F_centripetal_emp / 1e20],
            color="gold", edgecolor="black", s=80, zorder=6,
            label=f"Empirische Centripetale Kracht μω²r ({F_centripetal_emp/1e20:.3f}×10²⁰ N)"
        )
        ax1.axvline(D_EM_PERIGEE / 1e6, color="#e7298a", lw=1.2, ls=":", label="Perigeum (363 300 km)")
        ax1.axvline(D_EM_MEAN / 1e6, color="#1b9e77", lw=1.2, ls=":", label="Gemiddelde Baan (384 400 km)")
        ax1.axvline(D_EM_APOGEE / 1e6, color="#7570b3", lw=1.2, ls=":", label="Apogeum (405 500 km)")

        ax1.set_xlim(355.0, 415.0)
        ax1.set_ylim(1.65, 2.35)
        ax1.set_title("2. Detail Maanbaan: ΔF = +1.36×10¹¹ N t.o.v. Aard-G")
        ax1.set_xlabel("Afstand Aarde - Maan r [× 1000 km]")
        ax1.set_ylabel("Aantrekkingskracht F [× 10²⁰ N]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=7.6)

        # --- PANEEL 3: GEVOELIGHEID VOOR DE HEUVELRUG-WEGING η_⊕ OP DE MAANBAAN ---
        ax2 = axes[2]
        eta_arr = np.linspace(0.0, 0.50, 300)
        ln_u_moon = np.log(D_EM_MEAN / R_EARTH)
        phi_eta_arr = 1.0 + eta_arr * (1.0 - ln_u_moon)
        F_eta_arr = float(res_mean_eta0["F_tr"]) * phi_eta_arr / 1e20
        eta_crit = 1.0 / (ln_u_moon - 1.0)  # ~0.3226 (bijna exact XI_E = 0.323!)

        ax2.plot(eta_arr, F_eta_arr, color="#1b9e77", lw=2.8, label="ETRT Kracht F_TR(η_⊕) op r = 384 400 km")
        ax2.axhline(float(res_mean_eta0["F_newton"]) / 1e20, color="#d95f02", lw=1.6, ls="--", label="Werkelijke Aantrekking (+1.981 × 10²⁰ N)")
        ax2.axhline(0.0, color="black", lw=1.0, ls=":")
        ax2.axvline(eta_crit, color="#e7298a", lw=1.5, ls="-.", label=f"Kritische grens Φ=0: η = {eta_crit:.4f} (≈ XI_E = 0.323)")
        ax2.fill_between(eta_arr, 0.0, F_eta_arr, where=(F_eta_arr > 0), color="#1b9e77", alpha=0.15, label="Aantrekkend Regime (Maan blijft in baan)")
        ax2.fill_between(eta_arr, 0.0, F_eta_arr, where=(F_eta_arr < 0), color="#e7298a", alpha=0.15, label="Afstotend Regime (Heuvelrug domineert)")

        ax2.plot(0.0, float(res_mean_eta0["F_tr"]) / 1e20, marker="o", markersize=8, color="gold", markeredgecolor="black", zorder=6)
        ax2.annotate(
            "Fysieke Aarde-Maan balans\n(η_⊕ ≪ 1: F_TR = 1.981×10²⁰ N)",
            xy=(0.005, 1.98), xytext=(0.08, 2.15),
            fontsize=8.0, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.9)
        )

        ax2.set_xlim(0.0, 0.50)
        ax2.set_ylim(-1.2, 2.5)
        ax2.set_title("3. Invloed van Heuvelrug-Factor η_⊕ op de Maanbaan")
        ax2.set_xlabel("Macroscopische Heuvelrug-Weging η_⊕")
        ax2.set_ylabel("Netto Kracht op de Maan F_TR [× 10²⁰ N]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower left", fontsize=7.5)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.24)
        plt.savefig("earth_moon_force_tr.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'earth_moon_force_tr.png'.")

    return {
        "F_newton_mean": float(res_mean_eta0["F_newton"]),
        "F_tr_mean_eta0": float(res_mean_eta0["F_tr"]),
        "delta_F_surface_cal_N": float(res_mean_eta0["F_tr"] - res_mean_eta0["F_newton"]),
        "F_centripetal_emp": F_centripetal_emp,
        "eta_crit_moon": 1.0 / (np.log(D_EM_MEAN / R_EARTH) - 1.0),
    }


if __name__ == "__main__":
    run_earth_moon_force_comparison()
