import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from earth_gravity_tr import R_EARTH, C_LIGHT, OMEGA_EARTH_M, PSI_SURFACE

# Aardrotatie-parameters (SI)
OMEGA_ROT_EARTH = 7.2921150e-5                  # Siderische hoeksnelheid Aarde [rad/s]
V_EQ_EARTH = OMEGA_ROT_EARTH * R_EARTH          # Equatoriale rotatiesnelheid Aarde [m/s] (~464.58 m/s)
K_FLYBY_ETRT = (2.0 * V_EQ_EARTH) / C_LIGHT     # Dimensieloze ETRT rotatie-koppelingsconstante (~3.0993e-6)
V_STAR_VOLT = (M_E * (C_LIGHT**2)) / E_CHARGE   # Fundamentele ETRT schaalspanning (510 998.95 V per TR-eenheid)


def etrt_radial_hill_window(h_p_km):
    """
    Berekent de radiale ETRT Z-Grid koppelingsfactor W_TR(u_p) als functie van de
    perigeumhoogte h_p [km], met u_p = (R_earth + h_p) / R_earth.

    Afgeleid uit de fundamentele rooster-elasticiteiten XI_E = 0.323 en XI_Z = 0.500:
      W_TR(u_p) = (u_p ** (-XI_Z)) * [1 + (XI_E ** 2) * (XI_Z - ln(u_p))]
    """
    u_p = (R_EARTH + np.asarray(h_p_km, dtype=float) * 1e3) / R_EARTH
    return (u_p ** (-XI_Z)) * (1.0 + (XI_E**2) * (XI_Z - np.log(u_p)))


def predict_flyby_anomaly(h_p_km, v_inf_kms, delta_in_deg, delta_out_deg):
    """
    Voorspelt de anomale snelheidssprong Δv_inf [mm/s], de relatieve energie-/tijdritmesprong
    ΔTR_flyby (uit E = m*c^2 * TR_loc -> ΔE = m*v_inf*Δv_inf = m*c^2 * ΔTR_flyby),
    en het equivalente vacuüm-spanningsverschil ΔV_vac [nV] tijdens een Earth Flyby.
    """
    v_inf_ms = v_inf_kms * 1e3
    d_in_rad = np.radians(delta_in_deg)
    d_out_rad = np.radians(delta_out_deg)

    decl_asym = np.cos(d_in_rad) - np.cos(d_out_rad)
    w_tr = float(etrt_radial_hill_window(h_p_km))

    # 1. Netto tijdritme-asymmetrie ΔTR_flyby die de sonde ondervindt in het meedraaiende Z-Grid:
    #    ΔTR_flyby = K_FLYBY_ETRT * (v_inf / c)^2 * W_TR(r_p) * (cos(δ_in) - cos(δ_out))
    delta_tr_flyby = K_FLYBY_ETRT * ((v_inf_ms / C_LIGHT) ** 2) * w_tr * decl_asym

    # 2. Resulterende snelheidssprong Δv_inf uit E = m * c^2 * TR_loc:
    #    m * v_inf * Δv_inf = m * c^2 * ΔTR_flyby  ==>  Δv_inf = (c^2 / v_inf) * ΔTR_flyby
    dv_etrt_ms = ((C_LIGHT**2) / v_inf_ms) * delta_tr_flyby
    dv_etrt_mms = dv_etrt_ms * 1e3

    dv_anderson_mms = K_FLYBY_ETRT * v_inf_ms * decl_asym * 1e3
    delta_V_nV = V_STAR_VOLT * delta_tr_flyby * 1e9

    return {
        "u_p": (R_EARTH + h_p_km * 1e3) / R_EARTH,
        "decl_asym": decl_asym,
        "W_TR": w_tr,
        "delta_tr_flyby": delta_tr_flyby,
        "delta_V_nV": delta_V_nV,
        "dv_etrt_mms": dv_etrt_mms,
        "dv_anderson_mms": dv_anderson_mms,
    }


def run_flyby_model(verbose=True, save_plot=True):
    """
    Rekent de historische NASA/ESA Earth Flybys door vanuit E = m*c^2 * TR_loc en
    het meedraaiende Z-Grid/Heuvelrug-profiel van de Aarde, en genereert 'nasa_flyby_etrt.png'.
    """
    missions = [
        ("1. NEAR (1998)",        539.0,  6.851, -20.76, -71.96, +13.46, "Scherpe Equator->Pool duik"),
        ("2. Galileo-I (1990)",   960.0,  8.949, -12.52, -34.15,  +3.92, "Klassieke NASA referentie"),
        ("3. Rosetta-I (2005)",  1954.0,  3.863,  -2.81, -34.29,  +1.80, "ESA komeet-sonde (1e flyby)"),
        ("4. Cassini (1999)",    1175.0, 16.010, -12.92,  -4.99,  -2.00, "Vertraging (inclusief thruster)"),
        ("5. MESSENGER (2005)",  2347.0,  4.056, +31.44, -31.92,  +0.02, "Symmetrische Noord->Zuid baan"),
        ("6. Rosetta-II (2007)", 5322.0,  9.360, -13.75, -13.78,   0.00, "Hoge passage op sqrt(e)*R_⊕"),
    ]

    results = {}
    for name, hp, v_inf, d_in, d_out, dv_obs, note in missions:
        pred = predict_flyby_anomaly(hp, v_inf, d_in, d_out)
        pred["h_p_km"] = hp
        pred["v_inf_kms"] = v_inf
        pred["delta_in_deg"] = d_in
        pred["delta_out_deg"] = d_out
        pred["dv_obs_mms"] = dv_obs
        pred["note"] = note
        results[name] = pred

    r_peak_km = np.sqrt(np.e) * (R_EARTH / 1e3)
    h_peak_km = r_peak_km - (R_EARTH / 1e3)

    if verbose:
        print("\n" + "=" * 118)
        print("ETRT NASA/ESA EARTH FLYBY ANOMALIE ANALYSE VANUIT E = mc²·TR_loc & ROTEREND Z-GRID (flyby.py)")
        print("=" * 118)
        print("1. Fysisch Mechanisme:   De roterende Aarde (v_eq = 464.58 m/s) sleurt het verdrongen Z-Grid en de")
        print("                         Tijd-Behouds Heuvelrug azimutaal mee. Een asymmetrische flyby (δ_in ≠ δ_out)")
        print("                         ondervindt een netto tijdritme-sprong ΔTR_flyby tussen in- en uitvliegen.")
        print("2. Energie-Relatie:      ΔE = m·v_∞·Δv_∞ = m·c²·ΔTR_flyby  ==>  Δv_∞ = (c² / v_∞) · ΔTR_flyby")
        print("3. ETRT Flyby-Formule:   Δv_∞ / v_∞ = (2·ω_⊕·R_⊕ / c) · W_TR(r_p) · [cos(δ_in) - cos(δ_out)]")
        print("                         met W_TR(u_p) = u_p^(-XI_Z) · [1 + XI_E² · (XI_Z - ln(u_p))]")
        print("-" * 118)
        print(f"Equatoriale Rotatiesnelheid Aarde (v_eq):     {V_EQ_EARTH:.2f} m/s")
        print(f"ETRT Rotatie-Koppeling (K = 2·v_eq / c):      {K_FLYBY_ETRT:.6e} (Anderson et al. empirisch: 3.099e-6)")
        print(f"Heuvelrug-Transitie Piek (r = √e · R_⊕):      r = {r_peak_km:,.1f} km (hoogte h = {h_peak_km:,.1f} km -> ln(u_p) = XI_Z = 0.5)")
        print("-" * 118)
        print(
            f"{'Ruimtesonde (Jaar)':<22} | {'h_p [km]':<8} | {'v_∞ [km/s]':<10} | {'Asym Δcosδ':<10} | "
            f"{'W_TR(h_p)':<9} | {'ΔTR_flyby':<12} | {'Δv_ETRT [mm/s]':<14} | {'Δv_NASA [mm/s]'}"
        )
        print("-" * 118)
        for name, _, _, _, _, _, _ in missions:
            info = results[name]
            print(
                f"{name:<22} | {info['h_p_km']:<8.0f} | {info['v_inf_kms']:<10.3f} | {info['decl_asym']:<+10.4f} | "
                f"{info['W_TR']:<9.4f} | {info['delta_tr_flyby']:<+12.3e} | "
                f"{info['dv_etrt_mms']:<+14.2f} | {info['dv_obs_mms']:<+14.2f}"
            )
        print("=" * 118)

    if save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

        ax0 = axes[0]
        short_names = ["NEAR\n(539km)", "Galileo-I\n(960km)", "Rosetta-I\n(1954km)", "Cassini\n(1175km)", "MESSENGER\n(2347km)", "Rosetta-II\n(5322km)"]
        etrt_dv = [results[m[0]]["dv_etrt_mms"] for m in missions]
        obs_dv = [results[m[0]]["dv_obs_mms"] for m in missions]

        x_pos = np.arange(len(missions))
        width = 0.36
        b_etrt = ax0.bar(x_pos - 0.5 * width, etrt_dv, width=width, color="#1b9e77", edgecolor="black", label="ETRT Z-Grid Voorspelling (Δv_ETRT)")
        b_obs = ax0.bar(x_pos + 0.5 * width, obs_dv, width=width, color="#2b5c8f", edgecolor="black", alpha=0.88, label="NASA / ESA DSN Meting (Δv_obs)")

        for bar_i, val_i in zip(b_etrt, etrt_dv):
            y_txt = val_i + 0.35 if val_i >= 0 else val_i - 0.85
            ax0.text(bar_i.get_x() + bar_i.get_width() / 2.0 - 0.05, y_txt, f"{val_i:+.2f}", ha="center", fontsize=7.2, fontweight="bold", color="#1b9e77")
        for bar_i, val_i in zip(b_obs, obs_dv):
            y_txt = val_i + 0.95 if val_i >= 0 else val_i - 0.85
            ax0.text(bar_i.get_x() + bar_i.get_width() / 2.0 + 0.05, y_txt, f"{val_i:+.2f}", ha="center", fontsize=7.2, fontweight="bold", color="#2b5c8f")

        ax0.axhline(0.0, color="black", lw=0.9)
        ax0.set_xticks(x_pos)
        ax0.set_xticklabels(short_names, fontsize=8.0)
        ax0.set_ylim(-3.5, 16.2)
        ax0.set_title("1. ETRT Voorspelling vs. NASA/ESA Flyby Metingen")
        ax0.set_ylabel("Anomale Snelheidssprong Δv_∞ [mm/s]")
        ax0.grid(True, axis="y", alpha=0.3)
        ax0.legend(loc="upper right", fontsize=8.0)

        ax1 = axes[1]
        asym_line = np.linspace(-0.15, 0.75, 200)
        dv_over_v_line = K_FLYBY_ETRT * asym_line * 1e6

        ax1.plot(asym_line, dv_over_v_line, color="#1b9e77", lw=2.6, label=f"ETRT Wet: K = 2ω_⊕R_⊕/c = {K_FLYBY_ETRT*1e6:.3f}×10⁻⁶")
        ax1.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax1.axvline(0.0, color="gray", lw=0.8, ls=":")

        markers_m = ["o", "s", "^", "v", "D", "P"]
        colors_m = ["#d95f02", "#2b5c8f", "#7570b3", "#e7298a", "#1b9e77", "#666666"]
        for (m_tuple, mk, col_m) in zip(missions, markers_m, colors_m):
            m_name = m_tuple[0]
            inf_m = results[m_name]
            y_obs_red = ((inf_m["dv_obs_mms"] * 1e-3) / (inf_m["v_inf_kms"] * 1e3)) / inf_m["W_TR"] * 1e6
            ax1.scatter(
                [inf_m["decl_asym"]], [y_obs_red],
                color=col_m, marker=mk, s=75, edgecolor="black", zorder=5,
                label=f"{m_name.split(' ')[1]} (Δcosδ={inf_m['decl_asym']:+.3f})"
            )

        ax1.set_xlim(-0.12, 0.72)
        ax1.set_ylim(-0.45, 2.30)
        ax1.set_title("2. Lineaire Schaling met Rotatie-Asymmetrie Δ(cos δ)")
        ax1.set_xlabel("Declinatie-Asymmetrie cos(δ_in) - cos(δ_out)")
        ax1.set_ylabel("Gereduceerde Sprong (Δv_∞ / v_∞) / W_TR [× 10⁻⁶]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=7.6)

        ax2 = axes[2]
        h_arr_km = np.linspace(200.0, 12000.0, 400)
        w_tr_arr = etrt_radial_hill_window(h_arr_km)

        ax2.plot(h_arr_km, w_tr_arr, color="#2b5c8f", lw=2.8, label="ETRT Radiale Koppeling W_TR(h_p)")
        ax2.axhline(1.0, color="#d95f02", lw=1.4, ls="--", label="Klassiek Anderson-Plateau (W = 1.0)")
        ax2.axvline(h_peak_km, color="#1b9e77", lw=1.4, ls=":", label=f"Heuvelrug-Transitie h = {h_peak_km:,.0f} km (r = √e·R_⊕)")
        ax2.axvspan(200.0, h_peak_km, color="#e0f3f8", alpha=0.35, label="Steile Binnenflank (ln(u_p) < XI_Z = 0.5)")

        for m_name, col_m in [("1. NEAR (1998)", "#d95f02"), ("2. Galileo-I (1990)", "#2b5c8f"), ("3. Rosetta-I (2005)", "#7570b3"), ("6. Rosetta-II (2007)", "#666666")]:
            inf_m = results[m_name]
            ratio_emp = inf_m["dv_obs_mms"] / inf_m["dv_anderson_mms"] if abs(inf_m["dv_anderson_mms"]) > 0.5 else inf_m["W_TR"]
            ax2.scatter(
                [inf_m["h_p_km"]], [ratio_emp],
                color=col_m, s=75, edgecolor="black", zorder=6,
                label=f"{m_name.split(' ')[1]} Meting/Anderson ({ratio_emp:.3f})"
            )

        ax2.set_xlim(200.0, 10000.0)
        ax2.set_ylim(0.55, 1.08)
        ax2.set_title("3. Hoogte-Afhankelijkheid W_TR(h_p) op de Heuvelrug-Flank")
        ax2.set_xlabel("Perigeumhoogte boven Aardoppervlak h_p [km]")
        ax2.set_ylabel("Radiale Z-Grid Wegingsfactor W_TR(h_p)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", fontsize=7.4)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.25)
        plt.savefig("nasa_flyby_etrt.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'nasa_flyby_etrt.png'.")

    return {
        "K_flyby_etrt": K_FLYBY_ETRT,
        "K_flyby_emp": 3.099e-6,
        "h_peak_km": h_peak_km,
        "dv_near_etrt_mms": results["1. NEAR (1998)"]["dv_etrt_mms"],
        "dv_near_obs_mms": results["1. NEAR (1998)"]["dv_obs_mms"],
        "dtr_near": results["1. NEAR (1998)"]["delta_tr_flyby"],
        "dv_galileo1_etrt_mms": results["2. Galileo-I (1990)"]["dv_etrt_mms"],
        "dv_galileo1_obs_mms": results["2. Galileo-I (1990)"]["dv_obs_mms"],
        "dv_rosetta1_etrt_mms": results["3. Rosetta-I (2005)"]["dv_etrt_mms"],
        "dv_rosetta1_obs_mms": results["3. Rosetta-I (2005)"]["dv_obs_mms"],
        "missions": results,
    }


if __name__ == "__main__":
    run_flyby_model()