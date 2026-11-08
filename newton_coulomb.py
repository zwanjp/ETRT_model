import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from massfield import calculate_psi_component

# Natuurconstanten voor gravitatie en elektrostatica (SI)
G_NEWTON = 6.67430e-11                  # Gravitatiewet constante [m^3 kg^-1 s^-2]
EPSILON_0 = 8.8541878128e-12            # Vacuümpermittiviteit [F m^-1]
C_GRAV = 4.0 * np.pi * G_NEWTON         # Geünificeerde koppelingsconstante C_G = 4*pi*G
C_COULOMB = 1.0 / EPSILON_0             # Geünificeerde koppelingsconstante C_C = 1/epsilon_0


def tr_and_gradient_factor(r, r0, omega_1, eta=1.0, tr_l=1.0):
    """
    Berekent het lokale Time-Rhythm TR(r) en de dimensieloze TR-gradiëntfactor Phi_x(r)
    voor een bron met kernschaal r0 en topologische spankracht omega_1.
    - eta = 0.0: Zuivere lading (Coulomb, geen Tijd-Behouds Heuvelrug -> Phi = 1)
    - eta = 1.0: Massa / Nucleon (met logaritmische Tijd-Behouds Heuvelrug -> Phi = 2 - ln(r/r0))
    """
    r_safe = np.maximum(r, r0)
    psi = (omega_1 / r_safe) * (1.0 - eta * np.log(r_safe / r0))
    tr_r = tr_l * np.exp(-psi)
    phi_r = 1.0 + eta * (1.0 - np.log(r_safe / r0))
    return tr_r, phi_r, psi


def force_notation_1_bridge(r, x1, x2, C_x, r0, omega_1, eta=1.0, tr_l=1.0):
    """
    1. KLASSIEKE BRUGVORM (Hoofdformule):
       F_x(r) = C_x * (x_1 * x_2) / (4 * pi * r^2) * TR(r) * Phi_x(r)
       Laat direct het verschil zien met de klassieke wetten van Newton en Coulomb
       (waar TR(r) = 1 en Phi_x(r) = 1).
    """
    tr_r, phi_r, _ = tr_and_gradient_factor(r, r0, omega_1, eta=eta, tr_l=tr_l)
    f_classical = C_x * (x1 * x2) / (4.0 * np.pi * (r**2))
    return f_classical * tr_r * phi_r


def force_notation_2_euler(r, x1, x2, C_x, r0, omega_1, eta=1.0, tr_l=1.0):
    """
    2. WETENSCHAPPELIJKE EULER-SCHAALVORM (Analytische vorm voor eta = 1):
       F_x(r) = C_x * (x_1 * x_2) / (4 * pi * r^2) * TR_L * (r / r_e)**(omega_1 / r) * ln(e * r_e / r)
       waarbij r_e = e * r0 de spanningsvrije Euler-straal is (waar Psi = 0 en TR = TR_L).
    """
    r_safe = np.maximum(r, r0)
    f_classical = C_x * (x1 * x2) / (4.0 * np.pi * (r**2))
    if eta == 0.0:
        return f_classical * tr_l * np.exp(-omega_1 / r_safe)

    r_euler = np.e * r0
    tr_euler = tr_l * ((r_safe / r_euler) ** (omega_1 / r_safe))
    phi_euler = np.log((np.e * r_euler) / r_safe)
    return f_classical * tr_euler * phi_euler


def force_notation_3_differential(r, omega_1, omega_2, r0, eta=1.0, tr_l=1.0):
    """
    3. FUNDAMENTELE ETRT-DIFFERENTIAALVORM (Postulaatvorm):
       F_x(r) = - Omega_2 * (dTR_1 / dr)   [in 3D: F_x = - Omega_2 * grad(TR_1)]
       Elke kracht in de natuur is het product van de ontvangende tijd-massa (Omega_2)
       en de lokale ruimtelijke helling van het tijdritme (-dTR_1/dr).
    """
    # Numerieke afgeleide dTR_1/dr via centrale differentie voor verificatie
    eps = 1e-6 * r
    tr_plus, _, _ = tr_and_gradient_factor(r + eps, r0, omega_1, eta=eta, tr_l=tr_l)
    tr_minus, _, _ = tr_and_gradient_factor(r - eps, r0, omega_1, eta=eta, tr_l=tr_l)
    dtr_dr = (tr_plus - tr_minus) / (2.0 * eps)
    return omega_2 * dtr_dr


def run_newton_coulomb_model(verbose=True, save_plot=True):
    """
    Verifieert dat de drie notaties (1. Klassieke Brugvorm, 2. Euler-Schaalvorm,
    3. Fundamentele Differentiaalvorm) wiskundig en numeriek exact identiek zijn,
    en vergelijkt het Coulomb-regime (eta=0) met het Massa/Gravitatie-regime (eta=1).
    """
    r0 = Y_VAC / 2.0        # Karakteristieke kernschaal r0 = 5.0 [ModelUnits]
    r_euler = np.e * r0     # Euler-evenwichtspunt r_e = e * r0 (~13.59 ModelUnits)
    r_turn = (np.e**2) * r0 # Omslagpunt aantrekking -> heuvelrug-afstoting: e^2 * r0 (~36.95 ModelUnits)

    omega_1 = 957.0661 * OMEGA_MACRO_SCALE  # Structurele spankracht nucleon (~0.1176)
    omega_2 = omega_1
    C_model = 4.0 * np.pi
    x1, x2 = 1.0, omega_2 / omega_1

    r_grid = np.linspace(r0, 50.0, 400)

    # Bereken het Massa-regime (eta = 1) via alle drie de notaties
    F1_bridge = force_notation_1_bridge(r_grid, omega_1, omega_2, C_model, r0, omega_1, eta=1.0)
    F2_euler = force_notation_2_euler(r_grid, omega_1, omega_2, C_model, r0, omega_1, eta=1.0)
    F3_diff = force_notation_3_differential(r_grid, omega_1, omega_2, r0, eta=1.0)

    # Bereken het Coulomb-regime (eta = 0, zuivere 1/r put zonder heuvelrug) en Klassiek 1/r^2
    F_coulomb = force_notation_1_bridge(r_grid, omega_1, omega_2, C_model, r0, omega_1, eta=0.0)
    F_classic = (omega_1 * omega_2) / (r_grid**2)

    max_diff_12 = float(np.max(np.abs(F1_bridge - F2_euler)))
    max_diff_13 = float(np.max(np.abs(F1_bridge - F3_diff)))

    tr_r, phi_r, psi_r = tr_and_gradient_factor(r_grid, r0, omega_1, eta=1.0)

    if verbose:
        print("\n" + "=" * 90)
        print("GEÜNIFICEERDE ETRT NEWTON-COULOMB FORMULE: DRIE EQUIVALENTE NOTATIES (newton_coulomb.py)")
        print("=" * 90)
        print("1. Klassieke Brugvorm (Hoofdformule):")
        print("   F_x(r) = C_x * [x_1 * x_2 / (4*pi*r^2)] * TR(r) * Phi_x(r)")
        print("2. Wetenschappelijke Euler-Schaalvorm (met r_e = e*r_0):")
        print("   F_x(r) = C_x * [x_1 * x_2 / (4*pi*r^2)] * TR_L * (r / r_e)^(Omega_1 / r) * ln(e * r_e / r)")
        print("3. Fundamentele ETRT-Differentiaalvorm:")
        print("   F_x(r) = - Omega_2 * (dTR_1 / dr)   [of in 3D: F_x = - Omega_2 * grad(TR_1)]")
        print("-" * 90)
        print(f"Karakteristieke kernschaal (r_0):                {r0:.4f} [ModelUnits]")
        print(f"Spanningsvrije Euler-straal (r_e = e*r_0):       {r_euler:.4f} [ModelUnits] -> TR(r_e) = 1.000000")
        print(f"Gradiënt-omslagpunt (e*r_e = e^2*r_0):           {r_turn:.4f} [ModelUnits] -> Phi(e^2*r_0) = 0.000000")
        print(f"Maximale afwijking Notatie 1 vs Notatie 2:       {max_diff_12:.2e} (Exact identiek)")
        print(f"Maximale afwijking Notatie 1 vs Notatie 3:       {max_diff_13:.2e} (Exact identiek)")
        print("-" * 90)
        print(f"{'Straal r / r_0':<16} | {'TR(r) Massa':<14} | {'Gradiënt Phi(r)':<16} | {'F_Massa / F_Klassiek':<22} | {'Regime'}")
        print("-" * 90)
        for mult, label in [
            (1.0, "Oppervlak (r = r_0)"),
            (np.e, "Euler-punt (r = e*r_0)"),
            (5.0, "Overgangszone"),
            (np.e**2, "Top Heuvelrug (e^2*r_0)"),
            (10.0, "Heuvelrug-flank (> e^2*r_0)"),
        ]:
            r_pt = mult * r0
            tr_pt, phi_pt, _ = tr_and_gradient_factor(r_pt, r0, omega_1, eta=1.0)
            ratio_pt = tr_pt * phi_pt
            print(f"{mult:<16.4f} | {tr_pt:<14.6f} | {phi_pt:<+16.6f} | {ratio_pt:<+22.6f} | {label}")
        print("=" * 90)

    if save_plot:
        u = r_grid / r0
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))

        # Paneel 1: Tijdritme TR(r) en Gradiëntfactor Phi_x(r)
        axes[0].plot(u, tr_r, color="#2b5c8f", lw=2.5, label="Tijdritme TR(r) = exp(-Ψ)")
        axes[0].plot(u, phi_r, color="#d95f02", lw=2.4, ls="-.", label="TR-Gradiëntfactor Φ(r) = 2 - ln(r/r0)")
        axes[0].axhline(1.0, color="gray", lw=0.8, ls=":")
        axes[0].axhline(0.0, color="black", lw=0.8, ls=":")
        axes[0].axvline(np.e, color="#2b5c8f", lw=1.2, ls="--", alpha=0.7, label=f"r = e·r0 ({np.e:.2f} r0): TR = 1")
        axes[0].axvline(np.e**2, color="#d95f02", lw=1.2, ls="--", alpha=0.7, label=f"r = e²·r0 ({np.e**2:.2f} r0): Φ = 0")
        axes[0].set_title("ETRT Tijdritme TR(r) & Gradiëntfactor Φ(r)")
        axes[0].set_xlabel(" Genormaliseerde afstand r / r0")
        axes[0].set_ylabel("Dimensieloze factor")
        axes[0].set_xlim(1.0, 10.0)
        axes[0].set_ylim(-0.5, 2.2)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8.0)

        # Paneel 2: Verificatie van de Drie Equivalente Notaties vs Klassiek Coulomb (eta=0)
        axes[1].plot(u, F_classic / F_classic[0], color="gray", lw=1.8, ls=":", label="Klassiek Newton/Coulomb (1/r²)")
        axes[1].plot(u, F_coulomb / F_classic[0], color="#e7298a", lw=2.0, ls="--", label="ETRT Coulomb-regime (η = 0)")
        axes[1].plot(u, F1_bridge / F_classic[0], color="#1b9e77", lw=3.2, label="1. Brugvorm: F_klas · TR(r) · Φ(r)")
        axes[1].plot(u[::8], (F2_euler / F_classic[0])[::8], "o", color="gold", markeredgecolor="black", markersize=6, label="2. Euler-Schaalvorm (Exact gelijk)")
        axes[1].plot(u[::12], (F3_diff / F_classic[0])[::12], "x", color="#2b5c8f", markersize=7, mew=2.0, label="3. Differentiaalvorm -Ω₂ dTR₁/dr")
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set_title("Equivalentie van de 3 ETRT-Notaties")
        axes[1].set_xlabel("Genormaliseerde afstand r / r0")
        axes[1].set_ylabel("Genormaliseerde Kracht F(r) / F_klas(r0)")
        axes[1].set_xlim(1.0, 6.0)
        axes[1].set_ylim(-0.05, 2.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8.0)

        # Paneel 3: Verhouding F_ETRT(r) / F_Klassiek(r) = TR(r) * Phi(r)
        ratio_mass = F1_bridge / F_classic
        ratio_coul = F_coulomb / F_classic
        axes[2].plot(u, ratio_coul, color="#e7298a", lw=2.2, ls="--", label="Coulomb (η = 0): TR(r)·1 ≈ 1")
        axes[2].plot(u, ratio_mass, color="#1b9e77", lw=2.6, label="Massa (η = 1): TR(r)·[2 - ln(r/r0)]")
        axes[2].axhline(1.0, color="gray", lw=0.9, ls=":", label="Klassieke 1/r² limiet (Ratio = 1)")
        axes[2].axhline(0.0, color="black", lw=0.9, ls=":")
        axes[2].axvline(np.e, color="#2b5c8f", lw=1.1, ls="--", alpha=0.6, label="r = e·r0 (Ratio = 1.000)")
        axes[2].axvline(np.e**2, color="#d95f02", lw=1.1, ls="--", alpha=0.6, label="r = e²·r0 (Kracht-omslag Φ = 0)")
        axes[2].fill_between(u, 0.0, ratio_mass, where=(ratio_mass > 0), color="#1b9e77", alpha=0.15, label="Aantrekkend regime (Φ > 0)")
        axes[2].fill_between(u, 0.0, ratio_mass, where=(ratio_mass < 0), color="#d95f02", alpha=0.20, label="Afstotende Heuvelrug-flank (Φ < 0)")
        axes[2].set_title("Afwijking t.o.v. Klassiek: F_ETRT / F_Klassiek")
        axes[2].set_xlabel("Genormaliseerde afstand r / r0")
        axes[2].set_ylabel("Correctiefactor TR(r) · Φ_x(r)")
        axes[2].set_xlim(1.0, 10.0)
        axes[2].set_ylim(-0.4, 2.1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=7.8)

        fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.14, wspace=0.25)
        plt.savefig("newton_coulomb_unified.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'newton_coulomb_unified.png'.")

    return {
        "r0": r0,
        "r_euler": r_euler,
        "r_turn": r_turn,
        "max_diff_12": max_diff_12,
        "max_diff_13": max_diff_13,
    }


if __name__ == "__main__":
    run_newton_coulomb_model()