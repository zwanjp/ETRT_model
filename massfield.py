import numpy as np
import sys

try:
    from zgrid import ZGrid
    from parameters import *
except ImportError:
    from .zgrid import ZGrid
    from .parameters import *

# Maximale energie-/massaconcentratie in ETRT (Hoofdstuk 15, Vgl. 44-45):
# Wanneer RR(0) = TR_L * (2 - exp(delta * r_v)) = 0, bereikt de platte bodem in het
# middelpunt TR = 0 bij delta * r_v = ln(2) ≈ 0.693147. Grotere concentraties zijn uitgesloten.
MAX_DELTA_RV_ETRT = np.log(2.0)


def calculate_internal_time_mass(r_v, delta=None, r_z=Y_VAC, tr_l=1.0):
    """
    Berekent het inwendige tijdritme-profiel RR(r) met platte bodem in het middelpunt (r = 0)
    en de bijbehorende 'Time-Mass' (TM_rv) volgens Hoofdstuk 15 ('Reduced time-rhythm within objects'):
      - Vgl. (44): RC = TR_L * exp(delta * r_v)  (met fysische bovengrens RC <= 2*TR_L, dus delta*r_v <= ln(2))
      - 3D Sferische vorm van Vgl. (45) met platte bodem in r = 0 (dRR/dr|_{r=0} = 0):
        RR(r) = TR_L * [1 - exp(delta * r_v) + exp(delta * r^2 / r_v)]  voor 0 <= r <= r_v
      - Op r = 0:   RR(0) = TR_L * (2 - exp(delta * r_v)) >= 0  (platte bodem; bereikt 0 bij delta*r_v = ln(2))
      - Op r = r_v: RR(r_v) = TR_L
    """
    if delta is None:
        delta_tr_q = 1.0 - (r_v / (r_v + XI_Z)) ** 2
        delta = np.log(1.0 + delta_tr_q) / r_v if r_v > 1e-12 else 0.0

    # Begrens delta * r_v op de maximale ETRT energieconcentratie ln(2) ≈ 0.693147 (waar RR(0) = 0)
    delta_rv_raw = delta * r_v
    delta_rv = min(delta_rv_raw, MAX_DELTA_RV_ETRT)
    delta_capped = delta_rv / r_v if r_v > 1e-12 else 0.0

    rc_factor = np.exp(delta_rv)
    RC = tr_l * rc_factor
    RR_center_flat = max(0.0, tr_l * (2.0 - rc_factor))
    RR_center_mult = tr_l * np.exp(-delta_rv)

    # Vgl. (47): Totale Time-Mass op de rand r = r_v
    TM_rv = tr_l * (r_v / r_z) * (1.0 - rc_factor)

    # Sferische 3D-koppeling naar de macroscopische spankracht omega
    omega_theory = (4.0 / 3.0) * abs(TM_rv)

    return {
        "r_v": r_v,
        "r_z": r_z,
        "delta": delta_capped,
        "delta_rv": delta_rv,
        "max_delta_rv_limit": MAX_DELTA_RV_ETRT,
        "RC": RC,
        "RR_center_eq45": RR_center_flat,
        "RR_center_mult": RR_center_mult,
        "TM_rv": TM_rv,
        "omega_theory": omega_theory,
    }


def evaluate_internal_rr_profile(r, r_v, delta_rv, tr_l=1.0):
    """
    Evalueert het inwendige tijdritme RR(r) (voor |r| <= r_v) met een platte bodem in het
    middelpunt (r = 0, dRR/dr = 0). Bij toenemende energie (delta_rv -> ln 2) wordt de
    bodem naar beneden gedrukt tot het absolute minimum RR(0) = 0 bij delta_rv = ln(2).
    Grotere energieconcentraties (delta_rv > ln 2) zijn in ETRT uitgesloten.
    """
    drv = min(float(delta_rv), MAX_DELTA_RV_ETRT)
    u_sq = np.minimum((np.asarray(r, dtype=float) / r_v) ** 2, 1.0)
    return tr_l * (1.0 - np.exp(drv) + np.exp(drv * u_sq))


def calculate_psi_component(X, Y, Z, center, r0, omega, eta=1.0):
    """
    Berekent de lokale rooster-stress (Psi) voor één topologisch centrum volgens ETRT.
    Combineert:
      1. Binnen het object (0 <= r < r0, Hoofdstuk 15 met platte bodem in r = 0):
         Psi_in(r) = (omega / r0) * [2 - exp(-(r0^2 - r^2) / (2 * r0^2))]
         - In het middelpunt (r = 0): dPsi_in/dr = 0 (horizontale, platte bodem!)
           en Psi_in(0) = (omega / r0) * (2 - e^(-1/2)) ≈ 1.3935 * (omega / r0).
         - Op de rand (r = r0): Psi_in(r0) = omega / r0 en dPsi_in/dr = -omega / r0^2
           (naadloze C1-aansluiting zonder knik op het buitenveld!).
      2. Buiten het object (r >= r0, Hoofdstuk 2.4, 3.5 & 16):
         Psi_ext(r) = (omega / r) * [1 - eta * ln(r / r0)]
    """
    if isinstance(center, (tuple, list, np.ndarray)):
        x0, y0, z0 = center
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
    else:
        r = np.asarray(X, dtype=float)
        r0 = float(Y)
        omega = float(Z)

    r_safe = np.where(r < r0, r0, r)

    # Buitenveld (r >= r0): standaard TR-put (1/r) + logaritmische Tijd-Behouds Heuvelrug
    psi_well = omega / r_safe
    hill_factor = eta * np.log(r_safe / r0)
    psi_hill = (omega / r_safe) * hill_factor
    psi_ext = psi_well - psi_hill

    # Binnenveld (0 <= r < r0): platte, komvormige bodem in het middelpunt (dPsi/dr = 0 op r = 0)
    r_in = np.minimum(r, r0)
    psi_in = (omega / r0) * (2.0 - np.exp(-((r0**2 - r_in**2) / (2.0 * (r0**2)))))

    psi_total = np.where(r < r0, psi_in, psi_ext)
    return psi_total


def get_macroscopic_tr_field(X, Y, Z, particles, r0=5.0):
    """
    Berekent het absolute Time-Rhythm (TR) veld voor een verzameling deeltjes (macro-massa)
    via de niet-lineaire exponentiële compositie-regel: TR(x) = exp(-sum(Psi_i(x))).
    """
    psi_total = np.zeros_like(X, dtype=float)

    for p in particles:
        r0_p = p.get("r0", r0)
        psi_total += calculate_psi_component(X, Y, Z, p["center"], r0_p, p["omega"])

    TR_field = np.exp(-psi_total)
    return TR_field, psi_total


def apply_tr_to_energy(E_raw, TR_field):
    """
    De effectieve energie-dichtheid schaalt met het lokale tijdritme (E = m * c^2 * TR_loc).
    """
    return E_raw * TR_field


def apply_external_tr_modulation(
    X, Y, Z, E_tot_raw, center_p, center_n, r_bag_p, r_bag_n, M_p_raw, M_n_raw, omega_scale=OMEGA_MACRO_SCALE
):
    """
    Past in een samengestelde kern (zoals Deuterium) het externe TR-veld en de
    Tijd-Behouds Heuvelrug van de naburige nucleonen toe.
    """
    r_center_p = np.sqrt((X - center_p[0]) ** 2 + (Y - center_p[1]) ** 2 + (Z - center_p[2]) ** 2)
    r_center_n = np.sqrt((X - center_n[0]) ** 2 + (Y - center_n[1]) ** 2 + (Z - center_n[2]) ** 2)

    psi_ext_p = calculate_psi_component(X, Y, Z, center_p, r_bag_p, M_p_raw * omega_scale)
    psi_ext_n = calculate_psi_component(X, Y, Z, center_n, r_bag_n, M_n_raw * omega_scale)

    mask_ext_p = np.where(r_center_p > r_bag_p, psi_ext_p, 0.0)
    mask_ext_n = np.where(r_center_n > r_bag_n, psi_ext_n, 0.0)

    TR_mutual = np.exp(-(mask_ext_p + mask_ext_n))
    return apply_tr_to_energy(E_tot_raw, TR_mutual), TR_mutual