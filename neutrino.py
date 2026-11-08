import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model
from neutron import run_neutron_model
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from planck import run_planck_model


def run_neutrino_model(p_metrics=None, n_metrics=None, e_metrics=None, pl_metrics=None, verbose=True, save_plot=True):
    """
    ETRT 3D Z-Grid Model voor het Neutrino (neutrino.py).
    Modelleert de niet-gesloten topologische fase-slip (betaverval n -> p + e- + bar(nu)_e)
    met Q0 = 0, J_spin = sqrt(3)/2 * hbar, infinitesimale massa (~meV) en 3-smaaks
    kleur-/smaak-oscillaties (nu_e <-> nu_mu <-> nu_tau) afgeleid uit XI_E en XI_Z.
    """
    if p_metrics is None or n_metrics is None or e_metrics is None:
        if verbose:
            print("1. Ophalen van referentiemodellen (Proton, Neutron, Elektron, Planck)...")
        p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
        n_metrics = _extract_model_metrics(run_neutron_model, verbose=False)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=False)

    if pl_metrics is None:
        pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    M_p_phys = p_metrics["M0_phys"]
    M_n_phys = n_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    mass_conv_MeV = M_E_MEV / M_e_phys if M_e_phys > 1e-12 else 0.0

    # 1. Neutrino Lading en Spin
    Q_nu = 0.0000              # Zuiver neutraal (geen nettolading)
    spin_nu = J_SPIN           # Fermion spin-1/2 (sqrt(3)/2 in model-eenheden)

    # 2. Neutrino Massaschaal in eV (afgeleid uit M_e, ALPHA en Y_ASSYM / Y_VAC)
    m1_eV = 0.5 * (ALPHA**2) * M_E_MEV * 1e6 * (Y_ASSYM / Y_VAC)
    m2_eV = m1_eV / (XI_E**2)
    tr_up = ((Y_VAC / Y_UP)**3) * (((Y_UP / 2.0) / ((Y_UP / 2.0) + XI_Z))**2)
    delta_tr_up = abs(tr_up - 1.0)
    m3_eV = m2_eV / delta_tr_up
    sum_m_eV = m1_eV + m2_eV + m3_eV

    dm2_21 = m2_eV**2 - m1_eV**2
    dm2_31 = m3_eV**2 - m1_eV**2

    # 3. 3-Smaaks PMNS Mengingsmatrix U afgeleid uit de Z-Grid elasticiteitsparameters XI_E en XI_Z:
    # - Zonne-menging (x-y vlak):         sin^2(theta_12) = XI_E = 0.3230 (Emp: 0.307)
    # - Atmosferische menging (y-z vlak): sin^2(theta_23) = XI_Z = 0.5000 (Emp: 0.546)
    # - Reactor-menging (x-z koppeling):  sin^2(theta_13) = (XI_Z - XI_E)^2 / sqrt(2) = 0.0222 (Emp: 0.0220)
    s12_sq = XI_E
    s23_sq = XI_Z
    s13_sq = ((XI_Z - XI_E) ** 2) / np.sqrt(2.0)

    t12 = np.arcsin(np.sqrt(s12_sq))
    t23 = np.arcsin(np.sqrt(s23_sq))
    t13 = np.arcsin(np.sqrt(s13_sq))

    c12, s12 = np.cos(t12), np.sin(t12)
    c23, s23 = np.cos(t23), np.sin(t23)
    c13, s13 = np.cos(t13), np.sin(t13)

    U_pmns = np.array([
        [ c12 * c13,                  s12 * c13,                  s13],
        [-s12 * c23 - c12 * s23 * s13, c12 * c23 - s12 * s23 * s13, s23 * c13],
        [ s12 * s23 - c12 * c23 * s13,-c12 * s23 - s12 * c23 * s13, c23 * c13],
    ])

    # 4. Exacte 3-Smaaks Oscillatiekansen P(nu_e -> nu_e, nu_mu, nu_tau) langs afstand L [km]
    # Voor E_nu = 3.0 MeV (reactor-neutrino piekenergie) over L = 0 .. 60 km:
    E_nu_MeV = 3.0
    L_km = np.linspace(0.0, 60.0, 1500)
    m_sq_vec = np.array([m1_eV**2, m2_eV**2, m3_eV**2])

    phases = 2533.87 * np.outer(m_sq_vec, L_km / E_nu_MeV)
    exp_phases = np.exp(-1j * phases)

    P_ee = np.abs((U_pmns[0, :, None] * U_pmns[0, :, None]).T @ exp_phases).flatten() ** 2
    P_emu = np.abs((U_pmns[0, :, None] * U_pmns[1, :, None]).T @ exp_phases).flatten() ** 2
    P_etau = np.abs((U_pmns[0, :, None] * U_pmns[2, :, None]).T @ exp_phases).flatten() ** 2

    if verbose:
        print("\n" + "=" * 78)
        print("ETRT 3D Z-GRID NEUTRINO & 3-SMAAKS OSCILLATIE ANALYSE (neutrino.py)")
        print("=" * 78)
        print(f"Neutrino Lading (Q_nu):                   {Q_nu:.4f} C (Neutraal)")
        print(f"Neutrino Spin (s):                        {spin_nu:.4f} (Fermion spin-1/2)")
        print(f"Berekende Massa m_1 (lichtste):           {m1_eV*1e3:.4f} meV ({m1_eV:.5f} eV)")
        print(f"Berekende Massa m_2 (zonne-schaal):       {m2_eV*1e3:.4f} meV ({m2_eV:.5f} eV)")
        print(f"Berekende Massa m_3 (atmosferisch):       {m3_eV*1e3:.4f} meV ({m3_eV:.5f} eV)")
        print(f"Som van de neutrinomassa's (Σ m_nu):      {sum_m_eV*1e3:.2f} meV ({sum_m_eV:.4f} eV)")
        print("-" * 78)
        print(f"Berekend massaverschil Δm^2_21:           {dm2_21:.2e} eV^2 (Emp: 7.53e-5 eV^2)")
        print(f"Berekend massaverschil Δm^2_31:           {dm2_31:.2e} eV^2 (Emp: 2.45e-3 eV^2)")
        print(f"ETRT Mengingshoeken (sin^2 θ_12,23,13):   ({s12_sq:.4f}, {s23_sq:.4f}, {s13_sq:.4f})")
        print(f"Kleurwisseling Extrema (E_nu = 3 MeV):    Min P_ee = {np.min(P_ee)*100:.1f}%, Max P_eμ = {np.max(P_emu)*100:.1f}%, Max P_eτ = {np.max(P_etau)*100:.1f}%")
        print("=" * 78)

    if save_plot:
        c_e = "#1b9e77"    # Groen  (nu_e / nu_1)
        c_mu = "#d95f02"   # Oranje (nu_mu / nu_2)
        c_tau = "#7570b3"  # Paars  (nu_tau / nu_3)

        rgb_e = np.array(to_rgb(c_e))
        rgb_mu = np.array(to_rgb(c_mu))
        rgb_tau = np.array(to_rgb(c_tau))

        # Bereken voor elk punt L langs de baan de gemengde RGB-kleur van de neutrino-toestand:
        # RGB(L) = P_ee(L)*Groen + P_emu(L)*Oranje + P_etau(L)*Paars
        mixed_rgb = (
            P_ee[:, None] * rgb_e[None, :]
            + P_emu[:, None] * rgb_mu[None, :]
            + P_etau[:, None] * rgb_tau[None, :]
        )
        mixed_rgb = np.clip(mixed_rgb, 0.0, 1.0)

        fig = plt.figure(figsize=(18.0, 5.4))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.35, 1.35], wspace=0.24)

        # --- PANEEL 1: NEUTRINO MASSASPECTRUM [meV] ---
        ax0 = fig.add_subplot(gs[0])
        masses = [m1_eV * 1e3, m2_eV * 1e3, m3_eV * 1e3]
        labels = ["$\\nu_1$\n(Lichtste)", "$\\nu_2$\n(Zonne-schaal)", "$\\nu_3$\n(Atmosferisch)"]
        colors = [c_e, c_mu, c_tau]
        ax0.bar(labels, masses, color=colors, width=0.52, edgecolor="black", alpha=0.92)
        ax0.set_title("ETRT Voorspelde Neutrinomassa's [meV]")
        ax0.set_ylabel("Massa [meV]")
        ax0.set_ylim(0, 86)
        ax0.grid(True, alpha=0.3, axis="y")
        for i, m in enumerate(masses):
            ax0.text(i, m + 1.5, f"{m:.2f} meV", ha="center", fontweight="bold", fontsize=9.5)

        # --- PANEEL 2: 3-SMAAKS OSCILLATIECURVEN + MEEKLEURENDE GOLF ---
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(L_km, P_emu, color=c_mu, lw=1.8, alpha=0.85, label="P(ν_e → ν_μ) Muon-smaak (Oranje)")
        ax1.plot(L_km, P_etau, color=c_tau, lw=1.8, alpha=0.85, label="P(ν_e → ν_τ) Tau-smaak (Paars)")

        # Teken P_ee als een dikke meekleurende RGB-curve die van Groen -> Oranje/Paars -> Groen wisselt
        points = np.array([L_km, P_ee]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, colors=mixed_rgb[:-1], linewidths=3.4, zorder=4)
        ax1.add_collection(lc)
        ax1.plot([], [], color=c_e, lw=3.2, label="P(ν_e → ν_e) Elektron-smaak (Meekleurend)")

        # Voeg onderin (-0.15 tot -0.04) een continue RGB-kleurenbalk toe die de actuele neutrino-kleur toont
        ribbon_img = np.tile(mixed_rgb[None, :, :], (20, 1, 1))
        ax1.imshow(
            ribbon_img, extent=[L_km[0], L_km[-1], -0.15, -0.04],
            aspect="auto", origin="lower", zorder=3
        )
        ax1.axhline(-0.04, color="black", lw=0.7)
        ax1.axhline(-0.15, color="black", lw=0.7)
        ax1.text(
            0.5 * L_km[-1], -0.095, "Actuele Smaak-Kleurwisseling langs Pad L (Groen ↔ Oranje/Paars)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.35, edgecolor="none")
        )

        ax1.set_title(f"3-Smaaks Neutrino-Oscillatie & Kleurwisseling (E_ν = {E_nu_MeV:.0f} MeV)")
        ax1.set_xlabel("Afstand L [km]")
        ax1.set_ylabel("Overlevings- / Overgangskans P(ν_e → ν_α)")
        ax1.set_xlim(L_km[0], L_km[-1])
        ax1.set_ylim(-0.16, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right", fontsize=8.2)

        # --- PANEEL 3: GESTAPELDE SMAAK-KLEURWAAIER (100% SAMENSTELLING) ---
        ax2 = fig.add_subplot(gs[2])
        ax2.stackplot(
            L_km, P_ee * 100.0, P_emu * 100.0, P_etau * 100.0,
            labels=["ν_e fractie (Groen)", "ν_μ fractie (Oranje)", "ν_τ fractie (Paars)"],
            colors=[c_e, c_mu, c_tau], alpha=0.88, edgecolor="black", lw=0.4
        )
        ax2.set_title("Continue Kleur-Samenstelling van de Neutrino-Golf [%]")
        ax2.set_xlabel("Afstand L [km]")
        ax2.set_ylabel("Smaak-Samenstelling [%] (Som = 100%)")
        ax2.set_xlim(L_km[0], L_km[-1])
        ax2.set_ylim(0.0, 100.0)
        ax2.grid(True, alpha=0.25, color="white", ls="--")
        ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.15, wspace=0.24)
        plt.savefig("neutrino_oscillation.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie met kleurwisselingen opgeslagen als 'neutrino_oscillation.png'.")

    return {
        "Q_nu": Q_nu,
        "spin_nu": spin_nu,
        "m1_eV": m1_eV,
        "m2_eV": m2_eV,
        "m3_eV": m3_eV,
        "sum_m_eV": sum_m_eV,
        "dm2_21": dm2_21,
        "dm2_31": dm2_31,
        "s12_sq": s12_sq,
        "s23_sq": s23_sq,
        "s13_sq": s13_sq,
        "P_ee": P_ee,
        "P_emu": P_emu,
        "P_etau": P_etau,
        "L_km": L_km,
    }


if __name__ == "__main__":
    run_neutrino_model()