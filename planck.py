import numpy as np

from parameters import *
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics


def run_planck_model(e_metrics=None, verbose=True):
    """
    ETRT Afleiding en Verificatie van de Constante van Planck (hbar en h)
    vanuit de fundamentele Z-Grid elasticiteitsparameters XI_E, XI_Z en Y_VAC.
    """
    if e_metrics is None:
        if verbose:
            print("1. Ophalen van het 3D Elektron-Hopfion referentiemodel (electron.py)...")
            print("-" * 70)
        e_metrics = _extract_model_metrics(run_electron_model, verbose=verbose)
        if verbose:
            print("-" * 70)

    M0_raw_e = e_metrics["M0_raw"]
    M0_phys_e = e_metrics["M0_phys"]
    I_eff_e = e_metrics["I_eff"]
    R_rms_sq_e = e_metrics["R_rms_sq"]
    R_rms_e_model = np.sqrt(R_rms_sq_e)

    # 2. Lengte-schaal kalibratie vanuit het elektron-Hopfion
    r_e_m = R_E_FM * 1e-15
    m_per_unit = r_e_m / R_rms_e_model if R_rms_e_model > 1e-12 else 0.0

    xi_e_phys_m = XI_E * m_per_unit
    xi_z_phys_m = XI_Z * m_per_unit
    y_vac_phys_m = Y_VAC * m_per_unit

    # 3. Geometrische ETRT schaalrelatie tussen R_rms_e, XI_Z, Y_VAC en ALPHA:
    # Uit de 3D Hopfion-relaxatie met XI_E = 0.323 volgt R_rms_e / XI_Z ~ sqrt(Y_VAC / ALPHA)
    ratio_grid = R_rms_e_model / XI_Z
    ratio_theory = np.sqrt(Y_VAC / ALPHA)

    # In ModelUnits is het elementaire spin-actiekwantum hbar_model:
    hbar_model = ratio_theory / ratio_grid if ratio_grid > 1e-12 else 1.0
    h_model = 2.0 * np.pi * hbar_model

    # 4. Fysische afleiding van de gereduceerde constante van Planck (hbar) uit XI_Z en XI_E:
    # hbar_ETRT = m_e * c * xi_z_phys * sqrt(Y_VAC) / (ALPHA^(3/2))
    #           = m_e * c * xi_e_phys * (XI_Z / XI_E) * sqrt(Y_VAC) / (ALPHA^(3/2))
    hbar_etrt_Js = M_E * C * xi_z_phys_m * np.sqrt(Y_VAC) / (ALPHA**1.5)
    h_etrt_Js = 2.0 * np.pi * hbar_etrt_Js

    # Bijbehorende gereduceerde Compton-golflengte van het elektron uit XI_Z:
    lambda_c_bar_model = XI_Z * np.sqrt(Y_VAC) / (ALPHA**1.5)
    lambda_c_bar_phys_m = hbar_etrt_Js / (M_E * C)
    lambda_c_bar_emp_m = HBAR / (M_E * C)

    # Verwachte elektron-oppervlaktemassa M0_phys ~ ALPHA / XI_Z
    M0_phys_theory = ALPHA / XI_Z

    if verbose:
        print("\n" + "=" * 72)
        print("ETRT KWANTUM-ACTIE ANALYSE: CONSTANTE VAN PLANCK (planck.py)")
        print("=" * 72)
        print(f"Fundamentele defect-elasticiteit (XI_E):  {XI_E:.4f} ({xi_e_phys_m:.4e} m)")
        print(f"Lokale Z-Grid elasticiteit (XI_Z):        {XI_Z:.4f} ({xi_z_phys_m:.4e} m)")
        print(f"Vacuüm roosterafstand (Y_VAC):            {Y_VAC:.4f} ({y_vac_phys_m:.4e} m)")
        print("-" * 72)
        print(f"3D Z-Grid Schaalverhouding (R_e / XI_Z):  {ratio_grid:.4f}")
        print(f"Theoretische Schaalwortel √(Y_VAC / α):   {ratio_theory:.4f}")
        print(f"Fysieke Elektronmassa M0_phys (Z-Grid):   {M0_phys_e:.6f} (α / XI_Z = {M0_phys_theory:.6f})")
        print(f"Gereduceerde Compton-schaal (λ_bar_C):    {lambda_c_bar_phys_m:.4e} m (Emp: {lambda_c_bar_emp_m:.4e} m)")
        print("-" * 72)
        print(f"Model Actiekwantum (hbar_model):          {hbar_model:.4f} [ModelUnit] (Verwacht: 1.0000)")
        print(f"VOORSPELDE GEREDUCEERDE PLANCK (ħ_ETRT):  {hbar_etrt_Js:.6e} J·s")
        print(f"Empirische Gereduceerde Planck (ħ_emp):   {HBAR:.6e} J·s")
        print(f"VOORSPELDE VOLLEDIGE PLANCK (h_ETRT):     {h_etrt_Js:.6e} J·s (Emp: {2*np.pi*HBAR:.6e} J·s)")
        print(f"Relatieve nauwkeurigheid:                 {(1.0 - abs(hbar_etrt_Js - HBAR)/HBAR)*100:.2f}%")
        print("=" * 72)

    return {
        "XI_E": XI_E,
        "XI_Z": XI_Z,
        "xi_e_phys_m": xi_e_phys_m,
        "xi_z_phys_m": xi_z_phys_m,
        "ratio_grid": ratio_grid,
        "ratio_theory": ratio_theory,
        "hbar_model": hbar_model,
        "h_model": h_model,
        "hbar_phys_Js": hbar_etrt_Js,
        "h_phys_Js": h_etrt_Js,
        "lambda_c_bar_model": lambda_c_bar_model,
        "lambda_c_bar_phys_m": lambda_c_bar_phys_m,
        "lambda_c_bar_emp_m": lambda_c_bar_emp_m,
    }


if __name__ == "__main__":
    run_planck_model()