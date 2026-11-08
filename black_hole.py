import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from sun_earth_moon_tr import G_NEWTON, C_LIGHT, M_SUN, R_SUN, OMEGA_SUN

# Omrekeningsfactor radialen naar micro-boogseconden (μas) voor EHT-schaduwmetingen
PARSEC_M = 3.08567758149e16
RAD_TO_UAS = (180.0 / np.pi) * 3600.0 * 1e6

# Fundamentele ETRT/ETT singulariteitsgrens uit 'TR_within_objects.pdf' (Vgl. 51) & 'Technisch Addendum':
#   RR(0) = TR_L * [2 - exp(delta * r_v)] = 0  <==>  delta * r_v = ln(2) ≈ 0.693147
CRITICAL_COLLAPSE_LN2 = np.log(2.0)


def compute_internal_collapse_profile(u_r, delta_rv, tr_l=1.0):
    """
    Berekent het inwendige en uitwendige Time-Rhythm profiel RR(u) (Hoofdstuk 15, Vgl. 44-45)
    met een PLATTE BODEM in het middelpunt (u = r / r_v = 0, waar dTR/dr = 0):
      - Binnen (|u| <= 1):
        RR(u) = TR_L * [1 - exp(delta_rv) + exp(delta_rv * u^2)]
        * In het middelpunt (u = 0) heeft de TR-lijn een platte bodem:
          RR(0) = TR_L * (2 - exp(delta_rv))  en  dRR/du|_{u=0} = 0.
        * Bij toenemende energie (delta_rv) wordt de platte bodem naar beneden gedrukt.
        * De maximale energieconcentratie wordt bereikt wanneer RR(0) = 0, oftewel bij:
          delta_rv = ln(2) ≈ 0.693147 (grotere concentraties zijn uitgesloten in ETRT!).
      - Buiten (|u| > 1):
        Sluit op de rand |u| = 1 aan op het uitwendige herstel richting TR_vac.
    """
    u_abs = np.abs(np.asarray(u_r, dtype=float))
    drv = min(float(delta_rv), CRITICAL_COLLAPSE_LN2)
    u_in = np.minimum(u_abs, 1.0)
    u_ext = np.maximum(u_abs, 1.0)

    # Inwendig Hoofdstuk-15 profiel met platte bodem in r = 0 (genormaliseerd op randwaarde TR_L = 1 op |u|=1,
    # of geschaald zodat TR(|u|->inf) -> 1 en het centrum RR(0) = 2 - exp(drv) bedraagt):
    rr_in = tr_l * (1.0 - np.exp(drv) + np.exp(drv * (u_in**2)))
    # Buiten |u| > 1 loopt TR van TR_L op de rand geleidelijk naar het achtergronds-equilibrium:
    slope_edge = 2.0 * drv * np.exp(drv)
    rr_ext = tr_l * (1.0 + 0.18 * slope_edge * (1.0 - 1.0 / u_ext))

    return np.where(u_abs <= 1.0, rr_in, rr_ext), rr_in


def compute_exterior_black_hole_field(x_over_rs):
    """
    Vergelijkt het uitwendige veld van een Zwart Gat als functie van x = r / R_s
    (waarbij R_s = 2 * G * M / c^2 = 2 * Omega_M de Schwarzschildstraal is):

    1. Standaard Algemene Relativiteitstheorie (Schwarzschild):
       - TR_GR(x) = sqrt(1 - 1/x) voor x >= 1 (imaginair/onfysisch voor x < 1)
       - Versnelling g_GR(x) / g_N(R_s) = 1 / (x^2 * sqrt(1 - 1/x)) -> oneindig op x = 1 en x -> 0!

    2. ETRT Niet-Lineair Z-Grid Veld (vanuit E = m * c^2 * TR_loc en TR_loc = exp(-Omega_M / r)):
       - TR_ETRT(x) = exp(-1 / (2*x))  (blijft overal reëel en glad: TR(R_s) = e^(-1/2) = 0.6065, TR(0) = 0)
       - Versnelling g_ETRT(x) / g_N(R_s) = (1 / x^2) * exp(-1 / (2*x))
         Bereikt een EINDIG MAXIMUM op x_max = 1/4 (r = R_s / 4 = Omega_M / 2)
         en daalt daarbinnen (x -> 0) gladjes naar NUL (geen oneindige kracht-singulariteit!).
    """
    x_arr = np.asarray(x_over_rs, dtype=float)
    x_safe = np.maximum(x_arr, 1e-4)

    # ETRT Tijdritme en genormaliseerde gradiënt-kracht F_TR / F_N(R_s)
    tr_etrt = np.exp(-1.0 / (2.0 * x_safe))
    g_etrt_norm = (1.0 / (x_safe**2)) * tr_etrt

    # Schwarzschild (GR) Tijdritme en kracht
    tr_gr = np.where(x_safe > 1.0, np.sqrt(np.maximum(0.0, 1.0 - 1.0 / x_safe)), 0.0)
    g_newton_norm = 1.0 / (x_safe**2)
    g_gr_norm = np.where(
        x_safe > 1.005,
        1.0 / ((x_safe**2) * np.sqrt(np.maximum(1e-6, 1.0 - 1.0 / x_safe))),
        np.nan,
    )

    return {
        "tr_etrt": tr_etrt,
        "g_etrt_norm": g_etrt_norm,
        "tr_gr": tr_gr,
        "g_newton_norm": g_newton_norm,
        "g_gr_norm": g_gr_norm,
    }


def run_black_hole_model(verbose=True, save_plot=True):
    """
    ETRT Simulatie van Zwarte Gaten, de Eliminatie van Singulariteiten en de EHT-Schaduw (black_hole.py).
    Gebaseerd op:
      1. 'TR_within_objects.pdf' (Hoofdstuk 19, Vgl. 51) & 'Technisch Addendum Macro-schaal Gravitatie':
         De mechanische instortingsgrens δ · r_v = ln(2) ≈ 0.69315 waarbij TR(0) = 0 wordt bereikt
         en z-deeltjes in de kern niet meer kunnen roteren (rooster-extrusie).
      2. 'ETRT.pdf' (Hoofdstuk 3.5) & E = m * c^2 * TR_loc:
         Vervanging van de oneindige 1/r^2 singulariteit door de eindige maximale gradiënt op r = R_s / 4.
      3. Optische Fotonensfeer (r_ph = R_s) en Schaduwdiameter (b_schaduw = e * R_s) vergeleken met de
         Event Horizon Telescope (EHT) metingen van M87* (2019) en Sagittarius A* (2022).
    """
    # 1. Compactheidsladder (δ · r_v) voor astrofysische objecten (uit Technisch Addendum Tabel 3)
    astro_objects = [
        ("1. Aarde (R_⊕ = 6371 km)",              5.70e-7, "Extreem ijl; z-rooster nauwelijks gespannen"),
        ("2. Zon (R_☉ = 696 340 km)",             2.12e-6, "Stabiele hoofdreeksster (ΔTR = 2.12e-6)"),
        ("3. Witte Dwerg (Sirius B)",             3.00e-4, "Elektron-ontaard; ver onder breekpunt"),
        ("4. Neutronenster (PSR J0740+6620)",     0.5500,  "Kritiek; nadert fysiek breekpunt z-rooster"),
        ("5. Kritische ETRT Grens (TR_kern = 0)", CRITICAL_COLLAPSE_LN2, "Exact breekpunt: δ·r_v = ln(2) = 0.6931"),
        ("6. Grotere Concentratie (> ln 2)",      CRITICAL_COLLAPSE_LN2, "Uitgesloten in ETRT: bodem kan niet < TR = 0!"),
    ]

    # 2. Analytische eigenschappen van de ETRT Zwart-Gat Put buiten de kern:
    # - Tijdritme op de klassieke Schwarzschildstraal r = R_s = 2*Omega_M:
    tr_at_rs = float(np.exp(-0.5))  # e^(-1/2) = 0.606531
    # - Straal waar de aantrekkingskracht F_TR = (GMm/r^2) * exp(-Omega_M/r) maximaal is:
    #   d/dr [ r^(-2) * exp(-Omega_M/r) ] = 0  ==>  r_max = Omega_M / 2 = 0.25 * R_s !
    r_max_force_over_rs = 0.2500
    g_max_over_g_rs = float((1.0 / (0.25**2)) * np.exp(-1.0 / (2.0 * 0.25)))  # 16 * e^(-2) = 2.16536

    # 3. Fotonensfeer en EHT-Schaduwstraal vanuit de ETRT Brekingsindex n(r) = TR_loc(r)^(-2) = exp(2*Omega_M/r):
    # - Kritische fotonbaan waar d/dr [ r / n(r) ] = 0  ==>  r_foton = 2 * Omega_M = 1.0000 * R_s
    r_photon_over_rs = 1.0000
    # - Schijnbare schaduwstraal op oneindige afstand: b_schaduw = r_foton * n(r_foton) = e * R_s ≈ 2.7183 * R_s
    b_shadow_etrt_over_rs = float(np.e)                  # 2.71828
    b_shadow_gr_over_rs = float(1.5 * np.sqrt(3.0))      # 2.59808

    # Toetsing aan de twee door de Event Horizon Telescope (EHT) afgebeelde superzware zwarte gaten:
    # A) M87* (EHT Collaboration 2019: M = 6.5e9 M_sun, D = 16.8 Mpc, gemeten ringdiameter = 42.0 ± 3.0 μas)
    M_m87_kg = 6.50e9 * M_SUN
    D_m87_m = 16.80e6 * PARSEC_M
    Rs_m87_m = (2.0 * G_NEWTON * M_m87_kg) / (C_LIGHT**2)
    theta_m87_etrt_uas = (2.0 * b_shadow_etrt_over_rs * Rs_m87_m / D_m87_m) * RAD_TO_UAS
    theta_m87_gr_uas = (2.0 * b_shadow_gr_over_rs * Rs_m87_m / D_m87_m) * RAD_TO_UAS
    theta_m87_emp_uas = 42.00

    # B) Sagittarius A* (EHT Collaboration 2022 / GRAVITY: M = 4.154e6 M_sun, D = 8.277 kpc, gemeten = 51.8 ± 2.3 μas)
    M_sgra_kg = 4.154e6 * M_SUN
    D_sgra_m = 8.277e3 * PARSEC_M
    Rs_sgra_m = (2.0 * G_NEWTON * M_sgra_kg) / (C_LIGHT**2)
    theta_sgra_etrt_uas = (2.0 * b_shadow_etrt_over_rs * Rs_sgra_m / D_sgra_m) * RAD_TO_UAS
    theta_sgra_gr_uas = (2.0 * b_shadow_gr_over_rs * Rs_sgra_m / D_sgra_m) * RAD_TO_UAS
    theta_sgra_emp_uas = 51.80

    if verbose:
        print("\n" + "=" * 118)
        print("ETRT ZWARTE GATEN, ELIMINATIE VAN SINGULARITEITEN & EHT-SCHADUW ANALYSE (black_hole.py)")
        print("=" * 118)
        print("1. Inwendige Limiet:     RR(r) = TR_L · [2 - exp(δ·(r_v - r))] bereikt TR(0) = 0 exact bij δ·r_v = ln(2) ≈ 0.6931.")
        print("                         Voor δ·r_v > ln(2) stopt de z-rotatie in een eindige kern (rooster-extrusie, géén singulariteit).")
        print("2. Uitwendige Limiet:    Vanuit E = m·c²·TR_loc en TR_loc = exp(-Ω_M/r) bereikt de gravitatiekracht haar maximum")
        print("                         op r_max = R_s / 4 (2.17× F_N(R_s)) en dooft daarbinnen (r -> 0) gladjes uit naar NUL.")
        print("3. Fotonensfeer & EHT:   De optische brekingsindex n(r) = TR_loc⁻² geeft r_foton = R_s en schaduwstraal b = e · R_s.")
        print("-" * 118)
        print(f"Kritische ETRT Instortingsdrempel (δ·r_v):     ln(2) = {CRITICAL_COLLAPSE_LN2:.6f}")
        print(f"Tijdritme op Schwarzschildstraal TR(R_s):      e^(-1/2) = {tr_at_rs:.6f} (GR: 0.0000)")
        print(f"Piek-Kracht Positie (dF_TR/dr = 0):            r_max = {r_max_force_over_rs:.4f} R_s (F_max = {g_max_over_g_rs:.4f} × F_N(R_s), F(0) = 0)")
        print(f"Schaduwstraal Verhouding (b_schaduw / R_s):    e = {b_shadow_etrt_over_rs:.4f} (GR: 3√3/2 = {b_shadow_gr_over_rs:.4f})")
        print(f"M87* Schaduwdiameter (6.5×10⁹ M_☉, 16.8 Mpc):  ETRT = {theta_m87_etrt_uas:.2f} μas | GR = {theta_m87_gr_uas:.2f} μas | EHT Meting = {theta_m87_emp_uas:.2f} ± 3.0 μas")
        print(f"Sgr A* Schaduwdiameter (4.15×10⁶ M_☉, 8.3kpc): ETRT = {theta_sgra_etrt_uas:.2f} μas | GR = {theta_sgra_gr_uas:.2f} μas | EHT Meting = {theta_sgra_emp_uas:.2f} ± 2.3 μas")
        print("-" * 118)
        print(f"{'Astrofysisch Object':<38} | {'Product (δ·r_v)':<16} | {'TR_kern (r=0)':<15} | {'Bevroren Kern r_0/r_v':<21} | {'Status in ETRT Rooster'}")
        print("-" * 118)
        for name, drv, status_txt in astro_objects:
            tr_c = max(0.0, 2.0 - np.exp(drv))
            r_core_frac = max(0.0, 1.0 - CRITICAL_COLLAPSE_LN2 / drv) if drv > CRITICAL_COLLAPSE_LN2 else 0.0
            print(f"{name:<38} | {drv:<16.6e} | {tr_c:<15.6f} | {r_core_frac:<21.4f} | {status_txt}")
        print("=" * 118)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

        # --- PANEEL 1: HOOFDSTUK 15 PLATTE BODEM IN HET MIDDELPUNT & MAXIMALE ENERGIE (δ·r_v = ln 2) ---
        ax0 = axes[0]
        u_grid = np.linspace(-1.8, 1.8, 600)

        curves_p1 = [
            (0.15, "#2b5c8f", "--", "Lage Energie (δ·r_v = 0.15: Bodem TR=0.84)"),
            (0.35, "#7570b3", "-.", "Middel-Energie (δ·r_v = 0.35: Bodem TR=0.58)"),
            (0.55, "#d95f02", ":", "Neutronenster (δ·r_v = 0.55: Bodem TR=0.27)"),
            (CRITICAL_COLLAPSE_LN2, "#1b9e77", "-", "Max. Energie (δ·r_v = ln 2 ≈ 0.693: Bodem TR=0!)"),
        ]
        for drv_val, col_c, ls_c, lbl_c in curves_p1:
            tr_prof, _ = compute_internal_collapse_profile(u_grid, drv_val)
            ax0.plot(u_grid, tr_prof, color=col_c, lw=2.6, ls=ls_c, label=lbl_c)

        ax0.axvspan(-1.0, 1.0, color="#fee090", alpha=0.25, label="Binnen Object (-r_v ≤ r ≤ +r_v)")
        ax0.axvline(-1.0, color="black", lw=1.0, ls=":")
        ax0.axvline(1.0, color="black", lw=1.0, ls=":")
        ax0.axhline(0.0, color="#c51b7d", lw=1.4, ls="--", label="Absoluut Bodem-Minimum TR = 0 (Limiet)")
        ax0.plot(0.0, 0.0, marker="o", markersize=7.5, color="gold", markeredgecolor="black", zorder=6)

        ax0.set_xlim(-1.8, 1.8)
        ax0.set_ylim(-0.08, 1.25)
        ax0.set_title("1. Hfdst. 15: Platte Bodem (r=0) & Max. Energie (TR=0)")
        ax0.set_xlabel("Doorsnede door Middelpunt r / r_v")
        ax0.set_ylabel("Inwendig Tijdritme RR(r) / TR_L")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right", fontsize=7.1)

        # --- PANEEL 2: ELIMINATIE VAN DE ONEINDIGE KRACHT-SINGULARITEIT (ETRT VS. GR/NEWTON) ---
        ax1 = axes[1]
        x_arr = np.linspace(0.01, 2.5, 600)
        ext_fields = compute_exterior_black_hole_field(x_arr)

        ax1.plot(x_arr, ext_fields["g_newton_norm"], color="gray", lw=1.8, ls=":", label="Klassiek Newton: 1/x² → ∞ bij r→0")
        ax1.plot(x_arr, ext_fields["g_gr_norm"], color="#d95f02", lw=2.2, ls="--", label="Schwarzschild (GR): → ∞ op Horizon r=R_s")
        ax1.plot(x_arr, ext_fields["g_etrt_norm"], color="#1b9e77", lw=3.0, label="ETRT Kracht: (1/x²)·exp(-1/2x) → Eindig!")
        ax1.plot(x_arr, ext_fields["tr_etrt"], color="#2b5c8f", lw=2.2, ls="-.", label="ETRT Tijdritme TR_loc(r) = exp(-R_s/2r)")

        # Markeer de eindige kracht-piek op r = 0.25 R_s en de gladde uitdoving naar 0 bij r = 0
        ax1.plot(r_max_force_over_rs, g_max_over_g_rs, marker="o", markersize=8, color="gold", markeredgecolor="black", zorder=6)
        ax1.annotate(
            f"Eindig Kracht-Maximum\nr = 0.25 R_s (F_max = {g_max_over_g_rs:.2f} F_s)\nDaarna F → 0 wegens E = mc²·TR_loc!",
            xy=(r_max_force_over_rs, g_max_over_g_rs), xytext=(0.75, 1.80),
            fontsize=7.8, fontweight="bold", color="#1b9e77",
            arrowprops=dict(arrowstyle="->", color="#1b9e77", lw=1.1)
        )
        ax1.axvline(1.0, color="#d95f02", lw=1.2, ls=":", alpha=0.8, label="Schwarzschildstraal r = R_s (Fotonensfeer)")

        ax1.set_xlim(0.0, 2.5)
        ax1.set_ylim(0.0, 3.3)
        ax1.set_title("2. Eliminatie Singulariteit: Eindige Kracht bij r → 0")
        ax1.set_xlabel("Afstand in Schwarzschildstralen x = r / R_s")
        ax1.set_ylabel("Genormaliseerde Kracht F(r)/F_N(R_s) & TR(r)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=7.4)

        # --- PANEEL 3: EHT ZWART-GAT SCHADUWDIAMETER VOOR M87* EN SAGITTARIUS A* ---
        ax2 = axes[2]
        bh_names = ["M87* Schaduw\n(6.5×10⁹ M_☉, 2019)", "Sagittarius A* Schaduw\n(4.15×10⁶ M_☉, 2022)", "Schaduw-Ratio\n(b_schaduw / R_s) × 15"]
        etrt_bars = [theta_m87_etrt_uas, theta_sgra_etrt_uas, b_shadow_etrt_over_rs * 15.0]
        gr_bars = [theta_m87_gr_uas, theta_sgra_gr_uas, b_shadow_gr_over_rs * 15.0]
        emp_bars = [theta_m87_emp_uas, theta_sgra_emp_uas, (42.0 / (theta_m87_gr_uas / b_shadow_gr_over_rs)) * 15.0]

        x_pos = np.arange(len(bh_names))
        w = 0.26
        b1 = ax2.bar(x_pos - w, etrt_bars, width=w, color="#1b9e77", edgecolor="black", label="ETRT Voorspelling (b = e·R_s)")
        b2 = ax2.bar(x_pos, emp_bars, width=w, color="#2b5c8f", edgecolor="black", alpha=0.90, label="EHT Telescoop Meting (Empirisch)")
        b3 = ax2.bar(x_pos + w, gr_bars, width=w, color="#d95f02", edgecolor="black", alpha=0.80, label="Klassiek GR (b = 3√3/2·R_s)")

        for bar_i, val_i, is_rat in zip(b1, etrt_bars, [False, False, True]):
            lbl_v = f"{val_i/15.0:.2f} R_s" if is_rat else f"{val_i:.1f}"
            ax2.text(bar_i.get_x() + bar_i.get_width() / 2.0, val_i + 1.1, lbl_v, ha="center", fontsize=7.5, fontweight="bold", color="#1b9e77")
        for bar_i, val_i, is_rat in zip(b2, emp_bars, [False, False, True]):
            lbl_v = f"{val_i/15.0:.2f} R_s" if is_rat else f"{val_i:.1f}"
            ax2.text(bar_i.get_x() + bar_i.get_width() / 2.0, val_i + 3.2, lbl_v, ha="center", fontsize=7.5, fontweight="bold", color="#2b5c8f")
        for bar_i, val_i, is_rat in zip(b3, gr_bars, [False, False, True]):
            lbl_v = f"{val_i/15.0:.2f} R_s" if is_rat else f"{val_i:.1f}"
            ax2.text(bar_i.get_x() + bar_i.get_width() / 2.0, val_i + 1.1, lbl_v, ha="center", fontsize=7.5, fontweight="bold", color="#d95f02")

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(bh_names, fontsize=8.0)
        ax2.set_ylim(0.0, 66.0)
        ax2.set_title("3. EHT Zwart-Gat Schaduwdiameter: M87* & Sgr A* [μas]")
        ax2.set_ylabel("Schaduwdiameter θ_schaduw [micro-boogseconden μas]")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.legend(loc="upper right", fontsize=7.6)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
        plt.savefig("black_hole_etrt.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'black_hole_etrt.png'.")

    return {
        "critical_collapse_ln2": CRITICAL_COLLAPSE_LN2,
        "tr_at_rs": tr_at_rs,
        "r_max_force_over_rs": r_max_force_over_rs,
        "g_max_over_g_rs": g_max_over_g_rs,
        "b_shadow_etrt_over_rs": b_shadow_etrt_over_rs,
        "b_shadow_gr_over_rs": b_shadow_gr_over_rs,
        "theta_m87_etrt_uas": theta_m87_etrt_uas,
        "theta_m87_emp_uas": theta_m87_emp_uas,
        "theta_sgra_etrt_uas": theta_sgra_etrt_uas,
        "theta_sgra_emp_uas": theta_sgra_emp_uas,
    }


if __name__ == "__main__":
    run_black_hole_model()
