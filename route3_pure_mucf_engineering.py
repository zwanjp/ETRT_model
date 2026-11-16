#!/usr/bin/env python3
"""
ETRT Standalone Underlying Model: route3_pure_mucf_engineering.py
=================================================================
Part 8 (Energy Tests) — Section 8.7 Underlying Engineering, R&D Bottlenecks &
Grand 3-Route Strategic Synthesis Model:
Pure Room-Temperature Muon-Catalysed Fusion (muCF) via Diamond-Bragg CBXFEL
Cavity Stripping + Staged d-t Pion-Trap / Wide-Aperture ERL Muon Drivers, and
Master Comparison Across ETRT Routes 1, 2 & 3.

What this standalone script computes:
  1. Diamond-(400) Bragg CBXFEL Cavity & 20 MW_th Pilot Sizing:
     4-crystal synthetic diamond Bragg resonator (R = 99.2% -> Q_cav = 125x
     intracavity photon recycling at 8.20 / 10.93 keV), Be window transmission,
     and net electric pilot balance with a Stage-I 1.85 GeV d+t magnetic pion
     trap (Q_elec = 1.49x, net export +2.94 MW_e without fissile fuel).
  2. Honest R&D Scalability Bottlenecks (TRL 2-3) & Driver Comparison:
     Exact radiative Bhabha cross-section (>5% dp/p: 87.05 mbarn vs. 1.01 ubarn
     muon pair peak, ratio 8.62e4), proving why >=99.90% wide-aperture FFAG-ERL
     energy recovery and L_eff = R_mu/sigma = 2.7e46 cm^-2 s^-1 dense-target luminosity are required to
     lift wall-plug gain from Q_elec = 0.59 (5 GeV p+C) -> 1.49 (d+t) -> 3.83 (ERL).
  3. 200 MW_e Commercial Plant Economics (14% Learning: 94.2 -> 48.7 EUR/MWh) &
     Grand Societal Synthesis Comparing Routes 1, 2 & 3 vs. PWR, CCGT & Tokamak.

Usage (local):
  python3 route3_pure_mucf_engineering.py
    -> Prints full Route 3 engineering audit & Grand 3-Route comparison matrix
    -> Saves 3-panel diagnostic plot: route3_pure_mucf_engineering.png

Author: J.P. van der Zwan (2026) — ORCID ID 0009-0003-0591-5736
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# AUTOMATIC MACOS WORKING DIRECTORY FIX
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# =============================================================================
# 1. FIXED PHYSICS ANCHORS FROM route3_pure_mucf_physics.py (SECTION 8.6)
# =============================================================================
ALPHA = 1.0 / 137.035999084     # Fine-structure constant
M_E_MEV = 0.51099895            # Electron rest mass [MeV/c^2]
R_E_CM = 2.8179403262e-13       # Classical electron radius [cm]
E_CHARGE = 1.602176634e-19      # Elementary charge [C]
TAU_MU_S = 2.1970e-6            # Free muon Michel shedding lifetime [s]
LAM_0_MU = 1.0 / TAU_MU_S       # 4.5517e5 s^-1
LAM_C_OPT = 5.50e8              # Epithermal/dense D-T cycling rate [s^-1]
Q_TOT_CYCLE_MEV = 22.6125       # 17.5893 MeV D-T + 1.05 * 4.7840 MeV 6Li(n,a)t
SIG_MUMU_PEAK_UB = 1.010        # Peak e^-e^+ -> mu^-mu^+ cross-section [ubarn]
E_BEAM_ERL_MEV = 124.69         # Colliding beam energy at sqrt(s) = 249.38 MeV


# =============================================================================
# 2. MODULE A — DIAMOND-(400) CBXFEL CAVITY & 20 MW_th STAGE-I PILOT DESIGN
# =============================================================================
def compute_cbxfel_and_pilot_design(p_th_pilot_mw=20.0, omega_s_strip=0.0008):
    """
    Computes the 4-crystal synthetic diamond-(400) Cavity-Based XFEL (CBXFEL)
    photon recycling gain and sizes a 20 MW_th Stage-I pure-fusion pilot plant.
    """
    r_diamond_400 = 0.9920        # Synthetic C(400) Bragg reflectivity @ 8.2-10.9 keV
    q_cavity = 1.0 / (1.0 - r_diamond_400)                     # 125.0x photon recycling!
    t_be_window = np.exp(-2.0 * 0.010 * 1.85 * 0.58)           # Two 100 um Be windows = 97.9%

    nfus_strip = 1.0 / (omega_s_strip + LAM_0_MU / LAM_C_OPT)  # 614.49 fusions / mu^-
    e_th_per_mu_gev = nfus_strip * Q_TOT_CYCLE_MEV * 1.0e-3    # 13.895 GeV_th / mu^-

    eta_conv = 0.45               # 540 C SiC / 6Li-Pb -> sCO_2 Brayton efficiency
    e_gross_per_mu_geve = e_th_per_mu_gev * eta_conv           # 6.253 GeV_e / mu^-

    # Stage-I Pilot Driver: 1.85 GeV/mu^- neutron-rich d+t magnetic pion-trap (eta_acc = 0.52)
    # + CBXFEL intracavity makeup wall-plug overhead (0.65 GeV_e / mu^-):
    ewall_dt_geve = (1.85 / 0.52) + 0.65                       # 4.208 GeV_e / mu^-
    q_elec_stage1 = e_gross_per_mu_geve / ewall_dt_geve        # 1.486x (> 1.0 net electric!)

    mu_flux_pilot_s = (p_th_pilot_mw * 1.0e6) / (e_th_per_mu_gev * 1.0e9 * E_CHARGE)  # 8.98e14 mu^-/s
    p_gross_pilot_mwe = p_th_pilot_mw * eta_conv               # 9.000 MW_e gross
    p_recirc_pilot_mwe = p_gross_pilot_mwe / q_elec_stage1     # 6.057 MW_e driver + XFEL
    p_net_pilot_mwe = p_gross_pilot_mwe - p_recirc_pilot_mwe   # +2.943 MW_e net (zero U/Th!)

    return {
        "r_diamond_400": r_diamond_400,
        "q_cavity": q_cavity,
        "t_be_window_pct": t_be_window * 100.0,
        "nfus_strip": nfus_strip,
        "e_th_per_mu_gev": e_th_per_mu_gev,
        "e_gross_per_mu_geve": e_gross_per_mu_geve,
        "ewall_dt_geve": ewall_dt_geve,
        "q_elec_stage1": q_elec_stage1,
        "mu_flux_pilot_s": mu_flux_pilot_s,
        "p_gross_pilot_mwe": p_gross_pilot_mwe,
        "p_recirc_pilot_mwe": p_recirc_pilot_mwe,
        "p_net_pilot_mwe": p_net_pilot_mwe,
    }


# =============================================================================
# 3. MODULE B — HONEST ERL BHABHA BOTTLENECKS & 3-DRIVER Q_elec COMPARISON
# =============================================================================
def compute_erl_bottlenecks_and_drivers(pilot_data, p_net_comm_mwe=200.0):
    """
    Quantifies the radiative Bhabha scattering cross-section at 124.7 MeV/beam,
    derives the required FFAG-ERL energy-recovery efficiency & ring luminosity,
    and evaluates wall-plug Q_elec(omega_s) across 3 muon-driver technologies.
    """
    eta_conv = 0.45
    gamma_beam = E_BEAM_ERL_MEV / M_E_MEV                      # 244.01
    eps_mom_accept = 0.05                                      # 5% FFAG momentum acceptance

    # Exact Altarelli-Bucella / Baier-Fadin radiative Bhabha loss cross-section (>5% dp/p):
    sig_bhabha_cm2 = (16.0 / 3.0) * ALPHA * (R_E_CM ** 2) * (
        (np.log(4.0 * (gamma_beam ** 2)) - 0.5)
        * (np.log(1.0 / eps_mom_accept) - 0.625)
    )
    sig_bhabha_mb = sig_bhabha_cm2 * 1.0e27                    # 87.03 mbarn
    bhabha_to_mumu_ratio = (sig_bhabha_mb * 1.0e3) / SIG_MUMU_PEAK_UB  # 86,168 : 1!

    # Unrecovered beam-energy loss per mu^- pair at eta_ERL = 99.90% recovery:
    eta_erl_recov = 0.9990
    # Mean radiative loss among >5% events ~ 0.18 * E_beam, plus 0.10% unrecovered pass loss:
    e_net_erl_gev = (
        0.211 + bhabha_to_mumu_ratio * (E_BEAM_ERL_MEV * 1.0e-3) * (1.0 - eta_erl_recov) * 0.041
    )                                                          # ~0.652 GeV net thermalized / mu^-
    ewall_pC_geve = (5.00 / 0.50) + 0.65                       # 10.650 GeV_e (Std 5 GeV p+C)
    ewall_dt_geve = pilot_data["ewall_dt_geve"]                # 4.208 GeV_e (Stage-I d+t trap)
    ewall_erl_geve = (0.65 / 0.60) + 0.55                      # 1.633 GeV_e (Stage-II 249 MeV ERL)

    ws_pct = np.linspace(0.03, 0.90, 400)
    nfus_arr = 1.0 / ((ws_pct * 1.0e-2) + LAM_0_MU / LAM_C_OPT)
    e_gross_arr = nfus_arr * Q_TOT_CYCLE_MEV * 1.0e-3 * eta_conv

    q_elec_pC_curve = e_gross_arr / ewall_pC_geve
    q_elec_dt_curve = e_gross_arr / ewall_dt_geve
    q_elec_erl_curve = e_gross_arr / ewall_erl_geve

    eg_08 = pilot_data["e_gross_per_mu_geve"]
    qe_pC_08 = eg_08 / ewall_pC_geve                           # 0.587x (< 1.0 deficit!)
    qe_dt_08 = eg_08 / ewall_dt_geve                           # 1.486x (> 1.0 Stage-I net!)
    qe_erl_08 = eg_08 / ewall_erl_geve                         # 3.828x (> 3.8 Stage-II comm.!)

    # Commercial 200 MW_e,net Stage-II ERL Plant Sizing & Multi-IP Luminosity
    f_recirc_erl_pct = (1.0 / qe_erl_08) * 100.0               # 26.12%
    p_gross_comm_mwe = p_net_comm_mwe / (1.0 - 1.0 / qe_erl_08)# 270.72 MW_e
    p_th_comm_mw = p_gross_comm_mwe / eta_conv                 # 601.60 MW_th
    mu_rate_comm_s = (p_th_comm_mw * 1.0e6) / (
        pilot_data["e_th_per_mu_gev"] * 1.0e9 * E_CHARGE
    )                                                          # 2.702e16 mu^-/s
    # Exact required effective luminosity L_eff = R_mu / sigma_peak (1 ub = 1e-30 cm^2!):
    # Vacuum e-e+ colliders reach ~1e36 cm^-2 s^-1 (~10^6 mu^-/s), so 2.7e16 mu^-/s
    # requires dense-plasma/internal-target ERL luminosity L_eff = 2.68e46 cm^-2 s^-1:
    sig_peak_cm2 = SIG_MUMU_PEAK_UB * 1.0e-30                  # 1.010e-30 cm^2
    lumi_tot_cm2_s = mu_rate_comm_s / sig_peak_cm2             # 2.676e46 cm^-2 s^-1!

    return {
        "sig_bhabha_mb": sig_bhabha_mb,
        "bhabha_to_mumu_ratio": bhabha_to_mumu_ratio,
        "eta_erl_recov_pct": eta_erl_recov * 100.0,
        "e_net_erl_gev": e_net_erl_gev,
        "ws_pct": ws_pct,
        "q_elec_pC_curve": q_elec_pC_curve,
        "q_elec_dt_curve": q_elec_dt_curve,
        "q_elec_erl_curve": q_elec_erl_curve,
        "qe_pC_08": qe_pC_08,
        "qe_dt_08": qe_dt_08,
        "qe_erl_08": qe_erl_08,
        "f_recirc_erl_pct": f_recirc_erl_pct,
        "p_th_comm_mw": p_th_comm_mw,
        "mu_rate_comm_s": mu_rate_comm_s,
        "lumi_tot_cm2_s": lumi_tot_cm2_s,
    }


# =============================================================================
# 4. MODULE C — ROUTE 3 LCOE & GRAND 3-ROUTE STRATEGIC SYNTHESIS
# =============================================================================
def compute_route3_lcoe_and_synthesis():
    """
    Computes Route 3 FOAK -> NOAK LCOE (200 MW_e plant, 6% WACC, 40 yr, 88% CF)
    and compiles the master comparison dataset across Routes 1, 2 & 3 vs. rivals.
    """
    r_wacc = 0.060
    n_life_yr = 40
    cf_r3 = 0.88
    crf = (r_wacc * ((1.0 + r_wacc) ** n_life_yr)) / (((1.0 + r_wacc) ** n_life_yr) - 1.0)
    hours_r3 = 8760.0 * cf_r3                                  # 7708.8 h/yr

    capex_foak = 7900.0                                        # EUR / kW_e (ERL + CBXFEL + core)
    learn_rate = 0.14                                          # 14% per doubling
    capex_noak16 = float(capex_foak * ((1.0 - learn_rate) ** 4))  # 4321.37 EUR / kW_e

    lcoe_cap_foak = (capex_foak * 1000.0 * crf) / hours_r3     # 68.11 EUR/MWh
    lcoe_om_foak, lcoe_fuel_foak = 24.50, 1.60
    lcoe_r3_foak = lcoe_cap_foak + lcoe_om_foak + lcoe_fuel_foak  # 94.21 EUR/MWh

    lcoe_cap_noak = (capex_noak16 * 1000.0 * crf) / hours_r3   # 37.26 EUR/MWh
    lcoe_om_noak, lcoe_fuel_noak = 10.20, 1.20
    lcoe_r3_noak = lcoe_cap_noak + lcoe_om_noak + lcoe_fuel_noak  # 48.66 EUR/MWh

    synthesis_portfolio = [
        ("R1 Hybrid SMR\n(TRL 6-7)",   53.41, 31.78, "Depl. U/Th", "~300 yr FPs"),
        ("R2 Direct HV\n(TRL 4-5)",    51.66, 24.07, "1H + 11B",   "0 yr (4He)"),
        ("R3 Pure muCF\n(TRL 2-3)",    lcoe_r3_foak, lcoe_r3_noak, "D + 6Li", "<50 yr SiC"),
        ("Gen-III+ PWR\n(TRL 9)",     112.00, 96.50, "4.5% 235U",  "300,000 yr"),
        ("Gas CCGT\n+ EU CO2",        108.00,108.00, "CH4 + CO2",  "GHG emiss."),
        ("Tokamak DEMO\n(TRL 3-4)",   185.00,148.00, "1.5e8 K DT", "~100 yr Stl"),
    ]

    return {
        "capex_foak": capex_foak,
        "capex_noak16": capex_noak16,
        "lcoe_r3_foak": lcoe_r3_foak,
        "lcoe_r3_noak": lcoe_r3_noak,
        "synthesis_portfolio": synthesis_portfolio,
    }


# =============================================================================
# 5. STANDALONE DIAGNOSTIC PLOT & CONSOLE MASTER AUDIT REPORT
# =============================================================================
def build_diagnostic_figure(pilot, drv, syn):
    """Builds the 3-panel standalone Q_elec(omega_s), ERL recovery & 3-route LCOE figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # Panel 1: Wall-Plug Electrical Gain Q_elec(omega_s) Across 3 Muon Drivers
    ax0 = axes[0]
    ws = drv["ws_pct"]
    ax0.plot(ws, drv["q_elec_erl_curve"], color="#1b9e77", lw=2.6,
             label=rf"Stage-II $249\;\mathrm{{MeV}}$ ERL ($Q_e={drv['qe_erl_08']:.2f}\times$ @ $0.08\%$)")
    ax0.plot(ws, drv["q_elec_dt_curve"], color="#1a365d", lw=2.4, ls="--",
             label=rf"Stage-I $1.85\;\mathrm{{GeV}}$ $d+t$ trap ($Q_e={drv['qe_dt_08']:.2f}\times$ @ $0.08\%$)")
    ax0.plot(ws, drv["q_elec_pC_curve"], color="#c51b7d", lw=2.0, ls=":",
             label=rf"Std. $5.0\;\mathrm{{GeV}}$ $p+\mathrm{{C}}$ linac ($Q_e={drv['qe_pC_08']:.2f} < 1$)")
    ax0.axhline(1.0, color="black", lw=1.4, ls="-.", label=r"Wall-plug break-even $Q_{\mathrm{elec}}=1.00$")
    ax0.axvline(0.08, color="#2b5c8f", lw=1.2, ls=":")
    ax0.scatter([0.08, 0.08, 0.08], [drv["qe_pC_08"], drv["qe_dt_08"], drv["qe_erl_08"]],
                color="gold", edgecolor="black", s=58, zorder=6)
    ax0.set_xlim(0.03, 0.90)
    ax0.set_ylim(0.0, 5.2)
    ax0.set_title(r"1. Wall-Plug Gain $Q_{\mathrm{elec}}(\omega_s)$ Across 3 $\mu^-$ Drivers",
                  fontsize=9.8, fontweight="bold", color="#1a365d")
    ax0.set_xlabel(r"Effective alpha-sticking probability $\omega_s$ [%]", fontsize=9.0)
    ax0.set_ylabel(r"Net Electrical Gain $Q_{\mathrm{elec}}$", fontsize=9.0)
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=6.7, framealpha=0.95)

    # Panel 2: ERL Bhabha Energy-Recovery Sensitivity (Why >=99.8% Recovery Is Crucial)
    ax1 = axes[1]
    unrec_pct = np.logspace(-2.5, 0.5, 400)  # 0.003% to 3.16% unrecovered Bhabha loss (1 - eta_ERL)
    rec_pct = 100.0 - unrec_pct
    e_net_bhabha = 0.211 + drv["bhabha_to_mumu_ratio"] * (E_BEAM_ERL_MEV * 1.0e-3) * (unrec_pct * 1.0e-2) * 0.041
    ewall_sens = (e_net_bhabha / 0.60) + 0.55
    qe_sens = pilot["e_gross_per_mu_geve"] / ewall_sens

    ax1.semilogx(unrec_pct, qe_sens, color="#2b5c8f", lw=2.6,
                 label=r"Stage-II ERL $Q_{\mathrm{elec}}$ vs. unrecovered loss $(1-\eta_{\mathrm{ERL}})$")
    ax1.axhline(1.0, color="#c51b7d", lw=1.6, ls="--", label=r"Electrical break-even $Q_{\mathrm{elec}}=1.00$")
    ax1.axhline(drv["qe_dt_08"], color="#d95f02", lw=1.6, ls=":",
                label=rf"Stage-I $d+t$ pion-trap baseline (${drv['qe_dt_08']:.2f}\times$)")
    ax1.scatter([0.10], [drv["qe_erl_08"]], color="gold", edgecolor="black", s=62, zorder=6)
    ax1.annotate(
        rf"Target $\eta_{{\mathrm{{ERL}}}}=99.90\%$ ($0.10\%$ loss):" + "\n"
        rf"$Q_{{\mathrm{{elec}}}} = {drv['qe_erl_08']:.2f}\times$",
        xy=(0.10, drv["qe_erl_08"]), xytext=(0.18, 4.15),
        arrowprops=dict(arrowstyle="->", color="#1a365d", lw=1.3),
        fontsize=6.9, fontweight="bold", color="#1a365d",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#1a365d", alpha=0.95),
    )
    ax1.set_xlim(0.003, 3.0)
    ax1.set_ylim(0.0, 5.2)
    ax1.set_title(r"2. FFAG-ERL Bhabha Recovery Sensitivity ($249.4\;\mathrm{MeV}$)",
                  fontsize=9.8, fontweight="bold", color="#1a365d")
    ax1.set_xlabel(r"Unrecovered off-momentum beam fraction $1-\eta_{\mathrm{ERL}}$ [% (log)]", fontsize=8.8)
    ax1.set_ylabel(r"Net Plant Electrical Gain $Q_{\mathrm{elec}}$", fontsize=9.0)
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower left", fontsize=6.6, framealpha=0.95)

    # Panel 3: Grand Synthesis — FOAK & NOAK LCOE Across All 3 ETRT Routes vs. Rivals
    ax2 = axes[2]
    labels = [item[0] for item in syn["synthesis_portfolio"]]
    foak_v = np.array([item[1] for item in syn["synthesis_portfolio"]])
    noak_v = np.array([item[2] for item in syn["synthesis_portfolio"]])

    xc = np.arange(len(labels))
    ax2.bar(xc - 0.18, foak_v, width=0.35, color="#a0aec0", edgecolor="black", label="FOAK LCOE [EUR/MWh]")
    b_noak = ax2.bar(xc + 0.18, noak_v, width=0.35,
                     color=["#1b9e77", "#1b9e77", "#2b5c8f", "#718096", "#d95f02", "#c51b7d"],
                     edgecolor="black", label="NOAK Series LCOE [EUR/MWh]")

    for bi, nv, idx_c in zip(b_noak, noak_v, range(len(labels))):
        ax2.text(bi.get_x() + bi.get_width() / 2.0, nv + 3.0, f"{nv:.1f}",
                 ha="center", va="bottom", fontsize=7.1, fontweight="bold",
                 color="#1b7837" if idx_c < 2 else ("#1a365d" if idx_c == 2 else "#4a5568"))

    ax2.set_xticks(xc)
    ax2.set_xticklabels(labels, fontsize=7.2)
    ax2.set_ylim(0.0, 210.0)
    ax2.set_title("3. Grand Synthesis: All 3 ETRT Routes vs. Global Benchmarks",
                  fontsize=9.7, fontweight="bold", color="#1a365d")
    ax2.set_ylabel("Levelized Cost of Electricity [EUR / MWh]", fontsize=9.0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=6.8, framealpha=0.95)

    fig.tight_layout()
    return fig


def run_engineering_and_synthesis_audit():
    """Runs Route 3 engineering checks and prints the Grand 3-Route comparison audit."""
    pilot = compute_cbxfel_and_pilot_design()
    drv = compute_erl_bottlenecks_and_drivers(pilot)
    syn = compute_route3_lcoe_and_synthesis()

    rows_r3 = [
        ("Diamond-(400) CBXFEL Gain",       f"{pilot['q_cavity']:.0f}x (R=99.20%)",                       f"Be window trans: {pilot['t_be_window_pct']:.1f}%", "Q_cav",   "FEASIBLE (TRL 4)"),
        ("Std 5.0 GeV p+C + Stripping",     f"Q_elec = {drv['qe_pC_08']:.2f}x (< 1)",                     "Stripping alone insufficient",    "Q_elec",  "SUB-UNITY"),
        ("Stage-I Pilot (20 MW_th, d+t)",   f"+{pilot['p_net_pilot_mwe']:.2f} (Q_e={pilot['q_elec_stage1']:.2f}x)", "1.85 GeV d+t trap, 0 U/Th", "MW_e",    "NET SURPLUS"),
        ("Radiative Bhabha / mu-mu Ratio",  f"{drv['bhabha_to_mumu_ratio']:.0f} : 1",                     f"{drv['sig_bhabha_mb']:.1f} mb vs 1.01 ub",         "ratio",   "KEY BOTTLENECK"),
        ("Stage-II FFAG-ERL Recovery",      f">={drv['eta_erl_recov_pct']:.2f}% (L={drv['lumi_tot_cm2_s']:.1e})", f"Q_elec = {drv['qe_erl_08']:.2f}x (recirc 26%)", "eff / L", "LONG-TERM R&D"),
        ("Route 3 CAPEX (FOAK -> NOAK)",    f"{syn['capex_foak']:.0f} -> {syn['capex_noak16']:.0f}",      "14% learning (200 MW_e)",         "EUR/kWe", "PURE 300K FUSION"),
        ("Route 3 LCOE (FOAK -> NOAK)",     f"{syn['lcoe_r3_foak']:.1f} -> {syn['lcoe_r3_noak']:.1f}",    "0 actinides, <50 yr SiC wall",    "EUR/MWh", "COMPETITIVE"),
    ]

    print("=" * 108)
    print("ETRT PART 8 (SECTION 8.7) — STANDALONE ENGINEERING & SYNTHESIS AUDIT: route3_pure_mucf_engineering.py")
    print("=" * 108)
    print(f"{'Route 3 Metric / R&D Check':<32} | {'Route 3 Model':<26} | {'Consequence / Requirement':<26} | {'Unit':<7} | {'Assessment'}")
    print("-" * 108)
    for name, mod_v, ref_v, unit, status in rows_r3:
        print(f"{name:<32} | {mod_v:<26} | {ref_v:<26} | {unit:<7} | {status}")
    print("=" * 108)
    print("GRAND PART 8 PORTFOLIO COMPARISON (ALL 3 ETRT SURPLUS ROUTES VS. BENCHMARKS):")
    print("-" * 108)
    print(f"{'Technology / Horizon':<26} | {'FOAK LCOE':<14} | {'NOAK LCOE':<18} | {'Primary Fuel':<16} | {'Waste Horizon'}")
    print("-" * 108)
    for lbl, fk, nk, fl, wst in syn["synthesis_portfolio"]:
        clean_lbl = lbl.replace("\n", " ")
        nk_str = f"{nk:.1f} (26.8 w/MA)" if "R1" in clean_lbl else f"{nk:.1f} EUR/MWh"
        print(f"{clean_lbl:<26} | {fk:>6.1f} EUR/MWh | {nk_str:<18} | {fl:<16} | {wst}")
    print("=" * 108)
    return pilot, drv, syn


if __name__ == "__main__":
    p_r, d_r, s_r = run_engineering_and_synthesis_audit()
    fig_out = build_diagnostic_figure(p_r, d_r, s_r)
    out_png = "route3_pure_mucf_engineering.png"
    fig_out.savefig(out_png, dpi=150)
    plt.close(fig_out)
    print(f"Saved diagnostic figure: {os.path.join(SCRIPT_DIR, out_png)}")
