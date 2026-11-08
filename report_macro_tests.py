import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

from parameters import *
from newton_coulomb import run_newton_coulomb_model, tr_and_gradient_factor
from earth_gravity_tr import (
    run_earth_gravity_tr_model,
    compute_earth_tr_profile,
    R_EARTH,
    OMEGA_EARTH_M,
    PSI_SURFACE,
    SECONDS_PER_DAY,
    C_LIGHT,
)
from sun_earth_moon_tr import run_sun_earth_moon_superposition, superposed_tr_field_1d, D_EARTH_MOON
from earth_moon_force_tr import (
    run_earth_moon_force_comparison,
    compute_em_tr_force,
    D_EM_PERIGEE,
    D_EM_MEAN,
    D_EM_APOGEE,
)
from flyby import run_flyby_model
from mercury_precession import run_mercury_precession_model
from black_hole import run_black_hole_model
from dark_matter_energy import run_dark_matter_energy_model
from gravitational_waves import run_gravitational_waves_model

G_EARTH_SURFACE_EMP_MS2 = globals().get("G_EARTH_SURFACE_EMP_MS2", 9.80665)


def evaluate_status(val, target, is_qualitative=False, qual_pass=False, tolerance=0.05):
    if is_qualitative:
        return "TRUE" if qual_pass else "FAIL"
    if isinstance(target, str):
        return target
    if abs(target) < 1e-9:
        return "TRUE" if abs(val - target) < 0.05 else "FAIL"

    diff = abs(val - target) / abs(target)
    if diff <= tolerance:
        return f"TRUE ({diff*100:.1f}%)"
    else:
        return f"FAIL ({diff*100:.1f}%)"


def print_table_block(title, subtitle, plot_filename, rows):
    width = 118
    header_fmt = "| {0:<28} | {1:<14} | {2:<14} | {3:<9} | {4:<18} | {5:<18} |"
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print(f"{subtitle:^{width}}")
    print(f"{'[Grafiek: ' + plot_filename + ']':^{width}}")
    print("=" * width)
    print(header_fmt.format("Eigenschap / Missie / Zone", "Modelwaarde", "Fysieke Waarde", "Dimensie", "Doel / Empirisch", "Status"))
    print("-" * width)
    for r in rows:
        print(header_fmt.format(r[0], r[1], r[2], r[3], r[4], r[5]))
    print("=" * width)


def _render_visual_dashboard_and_pdf(
    sections,
    overview_png="report_macro_tests_overview.png",
    bundle_pdf="report_macro_tests_bundel.pdf",
):
    """
    Combineert voor zowel een totaal-overzicht (PNG) als een bundel (PDF) per macro-test
    de numerieke tabel en de bijbehorende 3-paneels grafieken direct bij elkaar.
    """
    n_sec = len(sections)

    with PdfPages(bundle_pdf) as pdf:
        for sec in sections:
            fig = plt.figure(figsize=(16.5, 9.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.42], hspace=0.18)

            ax_tbl = fig.add_subplot(gs[0])
            ax_tbl.axis("off")
            ax_tbl.set_title(
                f"{sec['title']}\n{sec['subtitle']}  —  Grafiek: {sec['plot_file']}",
                fontsize=12.0, fontweight="bold", pad=10, color="#1a365d"
            )

            col_labels = ["Eigenschap / Missie / Zone", "Modelwaarde", "Fysieke Waarde", "Dimensie", "Doel / Empirisch", "Status"]
            cell_text = [[str(c).strip() for c in row] for row in sec["rows"]]

            tbl = ax_tbl.table(
                cellText=cell_text,
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
                colWidths=[0.27, 0.14, 0.14, 0.09, 0.19, 0.16],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.8)
            tbl.scale(1.0, 1.30)

            for (r_idx, c_idx), cell in tbl.get_celld().items():
                if r_idx == 0:
                    cell.set_facecolor("#1a365d")
                    cell.set_text_props(color="white", fontweight="bold")
                else:
                    cell.set_facecolor("#f7fafc" if r_idx % 2 == 1 else "#edf2f7")
                    if c_idx == 0:
                        cell.get_text().set_horizontalalignment("left")
                    if c_idx == 5:
                        txt = cell.get_text().get_text()
                        if txt.startswith("TRUE"):
                            cell.set_text_props(color="#1b7837", fontweight="bold")
                        elif txt.startswith("FAIL"):
                            cell.set_text_props(color="#c51b7d", fontweight="bold")

            ax_img = fig.add_subplot(gs[1])
            ax_img.axis("off")
            if os.path.exists(sec["plot_file"]):
                img = mpimg.imread(sec["plot_file"])
                ax_img.imshow(img)

            fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04, hspace=0.18)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    fig_all = plt.figure(figsize=(20.0, 5.2 * n_sec))
    gs_all = fig_all.add_gridspec(n_sec, 2, width_ratios=[1.05, 1.45], wspace=0.06, hspace=0.22)

    for idx, sec in enumerate(sections):
        ax_t = fig_all.add_subplot(gs_all[idx, 0])
        ax_t.axis("off")
        ax_t.set_title(
            f"{sec['title']}\n{sec['subtitle']}",
            fontsize=10.8, fontweight="bold", loc="left", pad=6, color="#1a365d"
        )

        col_labels = ["Eigenschap / Missie / Zone", "Model", "Fysiek", "Dim", "Empirisch", "Status"]
        cell_text = [[str(c).strip() for c in row] for row in sec["rows"]]

        tbl = ax_t.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
            colWidths=[0.28, 0.14, 0.14, 0.08, 0.19, 0.17],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0)
        tbl.scale(1.0, 1.24)

        for (r_idx, c_idx), cell in tbl.get_celld().items():
            if r_idx == 0:
                cell.set_facecolor("#1a365d")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f7fafc" if r_idx % 2 == 1 else "#edf2f7")
                if c_idx == 5:
                    txt = cell.get_text().get_text()
                    if txt.startswith("TRUE"):
                        cell.set_text_props(color="#1b7837", fontweight="bold")

        ax_i = fig_all.add_subplot(gs_all[idx, 1])
        ax_i.axis("off")
        if os.path.exists(sec["plot_file"]):
            img = mpimg.imread(sec["plot_file"])
            ax_i.imshow(img)
        ax_i.set_title(f"Grafiek: {sec['plot_file']}", fontsize=10.0, color="#2d3748")

    fig_all.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.02, wspace=0.06, hspace=0.22)
    plt.savefig(overview_png, dpi=135)
    plt.close(fig_all)


def generate_macro_tests_report():
    print("\n" + "=" * 118)
    print(f"{'ETRT - GECONSOLIDEERD RAPPORT: MACRO-GRAVITATIE & KOSMISCHE TESTS (report_macro_tests.py)':^118}")
    print("=" * 118)
    print("Bezig met doorrekenen van alle 9 macro- en kosmische modellen...")
    sys.stdout.flush()

    nc_res = run_newton_coulomb_model(verbose=False, save_plot=True)
    eg_res = run_earth_gravity_tr_model(verbose=False, save_plot=True)
    sem_res = run_sun_earth_moon_superposition(verbose=False, save_plot=True)
    emf_res = run_earth_moon_force_comparison(verbose=False, save_plot=True)
    fb_res = run_flyby_model(verbose=False, save_plot=True)
    mp_res = run_mercury_precession_model(verbose=False, save_plot=True)
    bh_res = run_black_hole_model(verbose=False, save_plot=True)
    dme_res = run_dark_matter_energy_model(verbose=False, save_plot=True)
    gw_res = run_gravitational_waves_model(verbose=False, save_plot=True)

    sections = []

    # SECTIE 1: NEWTON-COULOMB
    r0_nc = nc_res["r0"]
    omega_nc = 957.0661 * OMEGA_MACRO_SCALE
    tr_r0, phi_r0, _ = tr_and_gradient_factor(r0_nc, r0_nc, omega_nc, eta=1.0)
    tr_re, phi_re, _ = tr_and_gradient_factor(nc_res["r_euler"], r0_nc, omega_nc, eta=1.0)
    tr_rt, phi_rt, _ = tr_and_gradient_factor(nc_res["r_turn"], r0_nc, omega_nc, eta=1.0)

    rows_nc = [
        ("Kernschaal r_0 (Y_VAC / 2)", f"{r0_nc:^14.4f}", f"{tr_r0:.6f}", "ModelUnit", f"Φ(r_0) = {phi_r0:+.4f}", evaluate_status(phi_r0, 2.0, tolerance=0.01)),
        ("Euler-punt r_e = e·r_0", f"{nc_res['r_euler']:^14.4f}", f"{tr_re:.6f}", "ModelUnit", "TR(e·r_0) = 1.0000", evaluate_status(tr_re, 1.0, tolerance=0.01)),
        ("Heuvelrug-Top e²·r_0", f"{nc_res['r_turn']:^14.4f}", f"{phi_rt:.6f}", "ModelUnit", "Φ(e²·r_0) = 0.0000", evaluate_status(phi_rt, 0.0)),
        ("Equivalentie Notatie 1 vs 2", f"{nc_res['max_diff_12']:^14.2e}", "0.0000", "-", "Brug == Euler-vorm", evaluate_status(nc_res["max_diff_12"], 0.0)),
        ("Equivalentie Notatie 1 vs 3", f"{nc_res['max_diff_13']:^14.2e}", "0.0000", "-", "Brug == -Ω2 dTR1/dr", evaluate_status(nc_res["max_diff_13"], 0.0)),
    ]
    sections.append({
        "title": "MACRO TEST 1: GEÜNIFICEERDE ETRT NEWTON-COULOMB FORMULE (newton_coulomb.py)",
        "subtitle": "Klassieke Brugvorm F_x = C_x [x1*x2/(4πr²)]·TR_loc·Φ_x, Euler-Schaalvorm & Differentiaalvorm -Ω2 dTR1/dr",
        "plot_file": "newton_coulomb_unified.png",
        "rows": rows_nc,
    })

    # SECTIE 2: AARDE TR-PROFIEL
    _, dtr_center, _, _ = compute_earth_tr_profile(0.0, eta=1.0)
    _, dtr_surf, dtr_dr_surf, _ = compute_earth_tr_profile(R_EARTH, eta=0.0)
    _, dtr_iss, _, _ = compute_earth_tr_profile(R_EARTH + 400.0e3, eta=0.0)
    _, dtr_euler, _, _ = compute_earth_tr_profile(np.e * R_EARTH, eta=1.0)
    _, dtr_hill_top, dtr_dr_top, _ = compute_earth_tr_profile((np.e**2) * R_EARTH, eta=1.0)

    clock_center_us = float(dtr_center) * SECONDS_PER_DAY * 1e6
    clock_surf_us = float(dtr_surf) * SECONDS_PER_DAY * 1e6
    clock_surf_emp_us = -PSI_SURFACE * SECONDS_PER_DAY * 1e6
    clock_iss_us = float(dtr_iss) * SECONDS_PER_DAY * 1e6
    clock_iss_emp_us = -(OMEGA_EARTH_M / (R_EARTH + 400.0e3)) * SECONDS_PER_DAY * 1e6
    g0_surf_ms2 = (C_LIGHT**2) * float(dtr_dr_surf)

    rows_eg = [
        ("Aarde Centrum (r = 0 km)", f"{float(dtr_center):^14.4e}", f"{clock_center_us:.2f}", "μs/dag", "< -60.15 (Platte bodem)", evaluate_status(0, 0, is_qualitative=True, qual_pass=(clock_center_us < clock_surf_us))),
        ("Aardoppervlak (r = 6371 km)", f"{float(dtr_surf):^14.4e}", f"{clock_surf_us:.2f}", "μs/dag", f"{clock_surf_emp_us:.2f} (NIST/GPS)", evaluate_status(clock_surf_us, clock_surf_emp_us, tolerance=0.05)),
        ("Oppervlakte-gradiënt g0(R_⊕)", f"{float(dtr_dr_surf):^14.4e}", f"{g0_surf_ms2:.3f}", "m/s^2", f"{G_EARTH_SURFACE_EMP_MS2:.3f} (c²|dTR/dr|)", evaluate_status(g0_surf_ms2, G_EARTH_SURFACE_EMP_MS2, tolerance=0.05)),
        ("ISS Lage Baan (h = 400 km)", f"{float(dtr_iss):^14.4e}", f"{clock_iss_us:.2f}", "μs/dag", f"{clock_iss_emp_us:.2f}", evaluate_status(clock_iss_us, clock_iss_emp_us, tolerance=0.05)),
        ("Euler-Nulpunt (e·R_⊕)", f"{float(dtr_euler):^14.4e}", f"{eg_res['r_euler_km']:.1f}", "km", "17318.2 (ΔTR = 0)", evaluate_status(float(dtr_euler), 0.0)),
        ("Heuvelrug-Top (e²·R_⊕)", f"{float(dtr_hill_top):^14.4e}", f"{eg_res['r_hill_top_km']:.1f}", "km", "47075.7 (dTR/dr = 0)", evaluate_status(float(dtr_dr_top), 0.0)),
    ]
    sections.append({
        "title": "MACRO TEST 2: TIJDRITME (TR) & GRADIËNT IN EN ROND DE AARDE (earth_gravity_tr.py)",
        "subtitle": "Laag-voor-laag TR-reductie in de aardkern (r ≤ R_⊕), oppervlakte-gradiënt g0 en de Tijd-Behouds Heuvelrug",
        "plot_file": "earth_tr_profile.png",
        "rows": rows_eg,
    })

    # SECTIE 3: SUPERPOSITIE ZON, AARDE EN MAAN
    dtr_sun_1AU = -sem_res["psi_sun_1AU"]
    clock_sun_1AU_us = dtr_sun_1AU * SECONDS_PER_DAY * 1e6
    dtr_sun_moon_mnd = sem_res["delta_tr_sun_moon_orbit"]
    clock_sun_moon_us = dtr_sun_moon_mnd * SECONDS_PER_DAY * 1e6
    x_saddle_km = sem_res["x_saddle_em_km"]
    u_saddle_earth = (x_saddle_km * 1e3) / R_EARTH
    tidal_ratio_ms = sem_res["tidal_ratio_moon_sun"]
    tr_moon_center = float(superposed_tr_field_1d(D_EARTH_MOON, moon_sign=+1.0, eta=0.0)[0] - 1.0)
    clock_moon_center_us = tr_moon_center * SECONDS_PER_DAY * 1e6

    rows_sem = [
        ("Zonne-put op 1 AU (ΔTR_Zon)", f"{dtr_sun_1AU:^14.4e}", f"{clock_sun_1AU_us:.2f}", "μs/dag", "-852.85 (14.2x Aarde)", evaluate_status(clock_sun_1AU_us, -852.85, tolerance=0.05)),
        ("Maanbaan Zonne-Modulatie", f"{dtr_sun_moon_mnd:^14.4e}", f"+{clock_sun_moon_us:.3f}", "μs/dag", "+4.383 (Vol - Nieuw)", evaluate_status(clock_sun_moon_us, 4.383, tolerance=0.05)),
        ("Aarde-Maan TR-Zadelpunt", f"{u_saddle_earth:^14.4f}", f"{x_saddle_km:.0f}", "km", "346033 (dTR_EM/dx=0)", evaluate_status(x_saddle_km, 346033.0, tolerance=0.05)),
        ("Dynamisch Lagrange L1-punt", f"{sem_res['x_L1_em_km']/6371.0:^14.4f}", f"{sem_res['x_L1_em_km']:.0f}", "km", "323137 (Incl. ω²r)", evaluate_status(sem_res["x_L1_em_km"], 323137.0, tolerance=0.05)),
        ("Maan-Centrum (Volle Maan)", f"{tr_moon_center:^14.4e}", f"{clock_moon_center_us:.2f}", "μs/dag", "-855.44 (Zon+Aarde+M)", evaluate_status(clock_moon_center_us, -855.44, tolerance=0.05)),
        ("Getijden-Ratio Maan / Zon", f"{tidal_ratio_ms:^14.4f}", f"{tidal_ratio_ms:.3f}", "-", "2.176 (ΔTR_getij)", evaluate_status(tidal_ratio_ms, 2.176, tolerance=0.05)),
    ]
    sections.append({
        "title": "MACRO TEST 3: SUPERPOSITIE VAN ZON, AARDE EN MAAN OP DE ZONNE-HELLING (sun_earth_moon_tr.py)",
        "subtitle": "Samengesteld tijdritme TR_tot = TR_L exp(-[Ψ_Zon + Ψ_Aarde + Ψ_Maan]), TR-zadelpunt en getijden-kromming",
        "plot_file": "sun_earth_moon_superposition.png",
        "rows": rows_sem,
    })

    # SECTIE 4: KRACHT AARDE-MAAN
    em_peri = compute_em_tr_force(D_EM_PERIGEE, eta_earth=0.0)
    em_mean = compute_em_tr_force(D_EM_MEAN, eta_earth=0.0)
    em_apo = compute_em_tr_force(D_EM_APOGEE, eta_earth=0.0)

    rows_emf = [
        ("Maanbaan TR-Helling |dTR/dr|", f"{float(em_mean['dtr_dr']):^14.4e}", f"{float(em_mean['g_tr_ms2']):.3e}", "m/s^2", f"{float(em_mean['g_newton_ms2']):.3e} (a_Maan)", evaluate_status(float(em_mean["g_tr_ms2"]), float(em_mean["g_newton_ms2"]), tolerance=0.05)),
        ("Kracht Perigeum (363 300 km)", f"{float(em_peri['F_tr']):^14.4e}", f"{float(em_peri['F_tr']):.3e}", "N", f"{float(em_peri['F_newton']):.3e} (Newton)", evaluate_status(float(em_peri["F_tr"]), float(em_peri["F_newton"]), tolerance=0.05)),
        ("Kracht Gemiddeld (384 400km)", f"{float(em_mean['F_tr']):^14.4e}", f"{float(em_mean['F_tr']):.3e}", "N", f"{float(em_mean['F_newton']):.3e} (Newton)", evaluate_status(float(em_mean["F_tr"]), float(em_mean["F_newton"]), tolerance=0.05)),
        ("Kracht Apogeum (405 500 km)", f"{float(em_apo['F_tr']):^14.4e}", f"{float(em_apo['F_tr']):.3e}", "N", f"{float(em_apo['F_newton']):.3e} (Newton)", evaluate_status(float(em_apo["F_tr"]), float(em_apo["F_newton"]), tolerance=0.05)),
        ("Verschil ΔF t.o.v. Aard-G", f"{float(em_mean['delta_tr_from_surface']):^14.4e}", f"+{emf_res['delta_F_surface_cal_N']:.3e}", "N", "+1.356e+11 (ΔTR_rel)", evaluate_status(emf_res["delta_F_surface_cal_N"], 1.35586e11, tolerance=0.05)),
        ("Centripetale Baankracht Maan", f"{float(em_mean['F_tr']):^14.4e}", f"{float(em_mean['F_tr']):.3e}", "N", f"{emf_res['F_centripetal_emp']:.3e} (μω²r)", evaluate_status(float(em_mean["F_tr"]), emf_res["F_centripetal_emp"], tolerance=0.05)),
        ("Kritische Grens Φ=0 (η_crit)", f"{emf_res['eta_crit_moon']:^14.4f}", f"{emf_res['eta_crit_moon']:.4f}", "-", f"{XI_E:.4f} (≈ XI_E)", evaluate_status(emf_res["eta_crit_moon"], XI_E, tolerance=0.05)),
    ]
    sections.append({
        "title": "MACRO TEST 4: ETRT TR-AANTREKKINGSKRACHT AARDE-MAAN VS. WERKELIJKE KRACHT (earth_moon_force_tr.py)",
        "subtitle": "Toetsing van F_TR = M_Maan·c²·|dTR_⊕/dr| op Perigeum, Gemiddelde baan en Apogeum (gekalibreerd op TR_opp ≡ 1)",
        "plot_file": "earth_moon_force_tr.png",
        "rows": rows_emf,
    })

    # SECTIE 5: NASA FLYBY
    rows_fb = [
        ("Rotatie-Koppeling K_ETRT", f"{fb_res['K_flyby_etrt']:^14.4e}", f"{fb_res['K_flyby_etrt']:.3e}", "-", f"{fb_res['K_flyby_emp']:.3e} (2ωR/c)", evaluate_status(fb_res["K_flyby_etrt"], fb_res["K_flyby_emp"], tolerance=0.05)),
        ("Heuvelrug-Transitie h_piek", f"{np.sqrt(np.e):^14.4f}", f"{fb_res['h_peak_km']:.1f}", "km", "4133.0 (r = √e·R_⊕)", evaluate_status(fb_res["h_peak_km"], 4133.0, tolerance=0.05)),
    ]
    for m_name, m_info in fb_res["missions"].items():
        short_m = m_name.replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "").replace("6. ", "")
        dv_mod = m_info["dv_etrt_mms"]
        dv_emp = m_info["dv_obs_mms"]
        if "Cassini" in short_m:
            st_str = evaluate_status(0, 0, is_qualitative=True, qual_pass=(dv_mod < 0 and dv_emp < 0))
        elif abs(dv_emp) < 0.1:
            st_str = evaluate_status(0, 0, is_qualitative=True, qual_pass=(abs(dv_mod - dv_emp) < 0.05))
        else:
            st_str = evaluate_status(dv_mod, dv_emp, tolerance=0.05)

        rows_fb.append((
            f"Flyby - {short_m}"[:28],
            f"ΔTR={m_info['delta_tr_flyby']:+.2e}",
            f"{dv_mod:+.2f}",
            "mm/s",
            f"{dv_emp:+.2f} (h={m_info['h_p_km']:.0f}km)",
            st_str,
        ))
    sections.append({
        "title": "MACRO TEST 5: NASA/ESA EARTH FLYBY ANOMALIEËN VANUIT E = mc²·TR_loc (flyby.py)",
        "subtitle": "Snelheidssprong Δv_∞ = (c²/v_∞)·ΔTR_flyby door azimutale Z-Grid meesleuring en radiale Heuvelrug-weging W_TR(h_p)",
        "plot_file": "nasa_flyby_etrt.png",
        "rows": rows_fb,
    })

    # SECTIE 6: MERCURIUS PRECESSIE
    merc_b = mp_res["bodies"]["1. Mercurius"]
    rows_mp = [
        ("Mercurius ΔTR_swing (p↔a)", f"{merc_b['dtr_swing']:^14.4e}", f"{merc_b['clock_swing_us_day']:.1f}", "μs/dag", "+51.8% in perihelium", evaluate_status(0, 0, is_qualitative=True, qual_pass=(merc_b["stiffness_ratio"] > 1.5))),
        ("Mercurius Precessie/Omloop", f"{merc_b['dphi_orb_arcsec']:^14.4f}", f"{merc_b['dphi_orb_arcsec']:.4f}", "''/omloop", "0.1035 (×415.2/eeuw)", evaluate_status(merc_b["dphi_orb_arcsec"], 0.1035, tolerance=0.05)),
    ]
    for b_name, b_info in mp_res["bodies"].items():
        short_b = b_name.replace("1. ", "").replace("2. ", "").replace("3. ", "").replace("4. ", "").replace("5. ", "")
        rows_mp.append((
            f"Precessie - {short_b}"[:28],
            f"e = {b_info['ecc']:.4f}",
            f"+{b_info['dphi_rk4_arcsec_cent']:.2f}",
            "''/eeuw",
            f"+{b_info['dphi_emp_arcsec_cent']:.2f} (a={b_info['a_Gm']:.0f}Gm)",
            evaluate_status(b_info["dphi_rk4_arcsec_cent"], b_info["dphi_emp_arcsec_cent"], tolerance=0.05),
        ))
    sections.append({
        "title": "MACRO TEST 6: PERIHELIUMPRECESSIE VAN MERCURIUS & BINNENPLANETEN (mercury_precession.py)",
        "subtitle": "Niet-lineaire Z-Grid stijfheid (d²u/dφ² + u = 1/p + 3·Ω_Zon·u²) in de steile Zonne-TR-put (+42.98''/eeuw)",
        "plot_file": "mercury_precession_etrt.png",
        "rows": rows_mp,
    })

    # SECTIE 7: ZWARTE GATEN
    rows_bh = [
        ("Instortingsgrens (δ·r_v)", f"{bh_res['critical_collapse_ln2']:^14.6f}", "ln(2) = 0.6931", "-", "0.693147 (TR_kern=0)", evaluate_status(bh_res["critical_collapse_ln2"], np.log(2.0), tolerance=0.01)),
        ("Tijdritme op Horizon R_s", f"{bh_res['tr_at_rs']:^14.6f}", f"{bh_res['tr_at_rs']:.4f}", "-", "e^(-1/2) (Geen breuk)", evaluate_status(bh_res["tr_at_rs"], np.exp(-0.5), tolerance=0.01)),
        ("Eindige Kracht-Piek r_max", f"{bh_res['r_max_force_over_rs']:^14.4f}", f"{bh_res['g_max_over_g_rs']:.4f} F_N(R_s)", "R_s", "0.2500 R_s (F(0)=0)", evaluate_status(bh_res["r_max_force_over_rs"], 0.25, tolerance=0.01)),
        ("Schaduwstraal b / R_s", f"{bh_res['b_shadow_etrt_over_rs']:^14.4f}", f"{bh_res['b_shadow_etrt_over_rs']:.4f}", "R_s", "2.75 ± 0.15 (EHT)", evaluate_status(bh_res["b_shadow_etrt_over_rs"], 2.75, tolerance=0.05)),
        ("EHT Schaduw M87* (2019)", f"b = e·R_s", f"{bh_res['theta_m87_etrt_uas']:.2f}", "μas", f"{bh_res['theta_m87_emp_uas']:.2f} ± 3.0 (EHT)", evaluate_status(bh_res["theta_m87_etrt_uas"], bh_res["theta_m87_emp_uas"], tolerance=0.05)),
        ("EHT Schaduw Sgr A* (2022)", f"b = e·R_s", f"{bh_res['theta_sgra_etrt_uas']:.2f}", "μas", f"{bh_res['theta_sgra_emp_uas']:.2f} ± 2.3 (EHT)", evaluate_status(bh_res["theta_sgra_etrt_uas"], bh_res["theta_sgra_emp_uas"], tolerance=0.05)),
    ]
    sections.append({
        "title": "MACRO TEST 7: ZWARTE GATEN, ELIMINATIE SINGULARITEIT & EHT-SCHADUW (black_hole.py)",
        "subtitle": "Platte bodem in r=0 met max. energie bij δ·r_v = ln(2) ≈ 0.693 (TR=0), eindige kracht op R_s/4 & EHT-schaduw",
        "plot_file": "black_hole_etrt.png",
        "rows": rows_bh,
    })

    # SECTIE 8: DONKERE MATERIE & DONKERE ENERGIE
    rows_dme = [
        ("Rooster-Versnelling a0_ETRT", f"{dme_res['a0_etrt_ms2']:^14.4e}", f"{dme_res['a0_etrt_ms2']:.3e}", "m/s^2", f"{dme_res['a0_emp_ms2']:.2e} (c·H0/2e)", evaluate_status(dme_res["a0_etrt_ms2"], dme_res["a0_emp_ms2"], tolerance=0.05)),
        ("Melkweg Zonnebaan (8.2 kpc)", f"{dme_res['v_sun_etrt_kms']:^14.1f}", f"{dme_res['v_sun_etrt_kms']:.1f}", "km/s", f"{dme_res['v_sun_emp_kms']:.1f} (Gaia DR3)", evaluate_status(dme_res["v_sun_etrt_kms"], dme_res["v_sun_emp_kms"], tolerance=0.05)),
        ("Melkweg Buitenrand (25 kpc)", f"{dme_res['v_mw_25kpc_etrt_kms']:^14.1f}", f"{dme_res['v_mw_25kpc_etrt_kms']:.1f}", "km/s", f"{dme_res['v_mw_25kpc_emp_kms']:.1f} (Vlak, 0 DM)", evaluate_status(dme_res["v_mw_25kpc_etrt_kms"], dme_res["v_mw_25kpc_emp_kms"], tolerance=0.05)),
        ("Tully-Fisher - NGC 3198", f"{dme_res['galaxies']['3. NGC 3198 (Referentie)']['v_etrt_kms']:^14.1f}", f"{dme_res['galaxies']['3. NGC 3198 (Referentie)']['v_etrt_kms']:.1f}", "km/s", "150.0 (SPARC)", evaluate_status(dme_res["galaxies"]["3. NGC 3198 (Referentie)"]["v_etrt_kms"], 150.0, tolerance=0.05)),
        ("Tully-Fisher - Andromeda", f"{dme_res['galaxies']['5. Andromeda (M31)']['v_etrt_kms']:^14.1f}", f"{dme_res['galaxies']['5. Andromeda (M31)']['v_etrt_kms']:.1f}", "km/s", "250.0 (M31)", evaluate_status(dme_res["galaxies"]["5. Andromeda (M31)"]["v_etrt_kms"], 250.0, tolerance=0.05)),
        ("Omslagpunt Φ_G=0 (e²·R_halo)", f"{np.e**2:^14.4f}", f"{dme_res['R_zero_gravity_etrt_Mpc']:.2f}", "Mpc", f"{dme_res['R_zero_gravity_emp_Mpc']:.2f} (Lokale Groep)", evaluate_status(dme_res["R_zero_gravity_etrt_Mpc"], dme_res["R_zero_gravity_emp_Mpc"], tolerance=0.05)),
        ("Materie-Fractie Ω_m (=XI_E)", f"{dme_res['Omega_m_etrt']:^14.4f}", f"{dme_res['Omega_m_etrt']*100:.2f}", "%", f"{dme_res['Omega_m_emp']*100:.2f}% (Planck 2018)", evaluate_status(dme_res["Omega_m_etrt"], dme_res["Omega_m_emp"], tolerance=0.05)),
        ("Donkere Energie Ω_Λ (1-XI_E)", f"{dme_res['Omega_Lambda_etrt']:^14.4f}", f"{dme_res['Omega_Lambda_etrt']*100:.2f}", "%", f"{dme_res['Omega_Lambda_emp']*100:.2f}% (Planck 2018)", evaluate_status(dme_res["Omega_Lambda_etrt"], dme_res["Omega_Lambda_emp"], tolerance=0.05)),
    ]
    sections.append({
        "title": "MACRO TEST 8: DONKERE MATERIE (ROTATIECURVEN) & DONKERE ENERGIE (dark_matter_energy.py)",
        "subtitle": "Vlakke rotatiecurven uit Heuvelrug-spanning (a0 = c·H0/2e), kosmische afstoting (Φ_G < 0) & Ω_m = XI_E = 32.3%",
        "plot_file": "dark_matter_energy_etrt.png",
        "rows": rows_dme,
    })

    # SECTIE 9: ZWAARTEKRACHTSGOLVEN (GW150914)
    rows_gw = [
        ("Golfsnelheid Z-Grid (v_GW/c)", f"{gw_res['v_gw_over_c_etrt']:^14.4f}", f"{gw_res['v_gw_over_c_etrt']:.4f}", "-", "1.0000 (GW170817)", evaluate_status(gw_res["v_gw_over_c_etrt"], gw_res["v_gw_over_c_emp"], tolerance=0.01)),
        ("GW150914 Chirp-Massa M_c", f"{gw_res['m_chirp_src']:^14.2f}", f"{gw_res['m_chirp_src']:.2f}", "M_sun", f"{gw_res['m_chirp_emp']:.2f} (LIGO)", evaluate_status(gw_res["m_chirp_src"], gw_res["m_chirp_emp"], tolerance=0.05)),
        ("Merger-Afstand (R_m = √e·R_s)", f"{np.sqrt(np.e):^14.4f}", f"{gw_res['R_merger_det_km']:.1f}", "km", "351.4 (Geen r=0 sing.)", evaluate_status(0, 0, is_qualitative=True, qual_pass=(gw_res["R_merger_det_km"] > 0))),
        ("Merger Piek-Frequentie f_GW", f"{gw_res['f_gw_peak_Hz']:^14.2f}", f"{gw_res['f_gw_peak_Hz']:.2f}", "Hz", f"{gw_res['f_gw_emp_Hz']:.1f} (LIGO Hanford)", evaluate_status(gw_res["f_gw_peak_Hz"], gw_res["f_gw_emp_Hz"], tolerance=0.05)),
        ("Piek TR-Rimpel op Aarde h_TR", f"{gw_res['h_peak_etrt']:^14.3e}", f"{gw_res['h_peak_etrt']:.3e}", "-", f"{gw_res['h_peak_emp']:.2e} (LIGO)", evaluate_status(gw_res["h_peak_etrt"], gw_res["h_peak_emp"], tolerance=0.05)),
        ("LIGO 4-km Arm-Shift ΔL_eff", f"{gw_res['delta_V_ligo_V']:^14.2e}", f"{gw_res['delta_L_ligo_m']:.3e}", "m", f"{gw_res['delta_L_emp_m']:.2e} (L·h_TR)", evaluate_status(gw_res["delta_L_ligo_m"], gw_res["delta_L_emp_m"], tolerance=0.05)),
        ("Uitgestraald Massadefect ΔM", f"{gw_res['E_gw_radiated_J']:^14.2e}", f"{gw_res['delta_m_etrt_sun']:.2f}", "M_sun", f"{gw_res['delta_m_emp_sun']:.2f} ± 0.5 (LIGO)", evaluate_status(gw_res["delta_m_etrt_sun"], gw_res["delta_m_emp_sun"], tolerance=0.05)),
    ]
    sections.append({
        "title": "MACRO TEST 9: ZWAARTEKRACHTSGOLVEN ALS 3D Z-GRID TIJDRITME-GOLF & LIGO GW150914 (gravitational_waves.py)",
        "subtitle": "Quadrupolaire spiraalgolf δTR(r,t) met v_GW = c, begrensde merger op R = √e·R_s (149.6 Hz) & ΔM = 2.93 M_☉",
        "plot_file": "gravitational_waves_etrt.png",
        "rows": rows_gw,
    })

    for sec in sections:
        print_table_block(sec["title"], sec["subtitle"], sec["plot_file"], sec["rows"])

    overview_png = "report_macro_tests_overview.png"
    bundle_pdf = "report_macro_tests_bundel.pdf"
    _render_visual_dashboard_and_pdf(sections, overview_png=overview_png, bundle_pdf=bundle_pdf)

    print("\n" + "=" * 118)
    print("GEZAMENLIJK VISUEEL MACRO-OVERZICHT OPGESLAGEN ALS:")
    print(f"  1. '{overview_png}' (Alle 9 de macro-tabellen + 9 grafieken naast elkaar in één hoog-resolutie dashboard)")
    print(f"  2. '{bundle_pdf}'  (9-pagina rapport met per pagina de numerieke tabel én de bijbehorende grafieken)")
    print("=" * 118)


if __name__ == "__main__":
    generate_macro_tests_report()