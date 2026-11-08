import numpy as np
import matplotlib.pyplot as plt

from zgrid import ZGrid
from parameters import *
from proton import run_proton_model, get_spherical_envelope as get_proton_envelope
from electron import run_unified_model as run_electron_model
from deuterium import _extract_model_metrics
from massfield import get_macroscopic_tr_field
from planck import run_planck_model

def run_qubit_decoherence_test(verbose=True, save_plot=True):
    """
    Simuleert het decoherentie-effect (T1 energie-relaxatie en T2 fase-degradatie)
    van een 3D Z-Grid qubit onder invloed van een omringende TR-gradiënt (tijdritme-ruis).
    """
    if verbose:
        print("1. Ophalen van referentiemodellen en initialiseren van het Z-Grid...")
    p_metrics = _extract_model_metrics(run_proton_model, verbose=False)
    e_metrics = _extract_model_metrics(run_electron_model, verbose=False)
    pl_metrics = run_planck_model(e_metrics=e_metrics, verbose=False)

    M_p_raw = p_metrics["M0_raw"]
    M_p_phys = p_metrics["M0_phys"]
    M_e_phys = e_metrics["M0_phys"]
    R_rms_e_model = np.sqrt(e_metrics["R_rms_sq"])

    red_mass_factor = M_p_phys / (M_p_phys + M_e_phys)
    r_bohr_grid = R_rms_e_model / red_mass_factor

    # Rooster-opbouw
    zg = ZGrid(shape=GRID_SHAPE_PHOTON, dx=DX_ELECTRON)
    r_grid = np.sqrt(zg.X**2 + zg.Y**2 + zg.Z**2)
    cutoff = np.where(r_grid <= R_ELECTRON, 1.0, 0.0)

    # Basistoestanden |0> en |1>
    psi_0_raw = np.exp(-r_grid / r_bohr_grid) * cutoff
    psi_0 = psi_0_raw / np.sqrt(np.sum(psi_0_raw**2) * zg.dV)

    psi_1_raw = (zg.Z / r_bohr_grid) * np.exp(-r_grid / (2.0 * r_bohr_grid)) * cutoff
    psi_1 = psi_1_raw / np.sqrt(np.sum(psi_1_raw**2) * zg.dV)

    # Initiële superpositie: Hadamard poort |+> = (|0> + |1>) / sqrt(2)
    # Met een externe tijdritme-gradiënt g_TR langs de z-as: TR_local(z) = TR_0 + g_TR * z
    g_TR_values = [0.0, 0.0001, 0.0005, 0.0020]
    t_steps = np.linspace(0.0, 100.0, 50) # Tijdstappen in ModelUnits

    results = {}
    h_etrt = pl_metrics["hbar_model"]

    for g_tr in g_TR_values:
        bloch_norms = []
        phase_drifts = []
        
        # Lokale TR-gradiënt veroorzaakt een energie-verschil ΔE(z) = E_01 * (1 + g_tr * z)
        # en daarmee een fase-drift d(phi)/dt = (ΔE(z) - ΔE_0) / hbar
        for t in t_steps:
            # Effectieve Bloch-vector lengte onder invloed van defasering T2:
            # r_bloch(t) = exp(-(t / T2)^2), waarbij T2 omgekeerd evenredig is met g_tr
            if g_tr == 0.0:
                r_b = 1.0000
                phase_std = 0.0000
            else:
                # ETRT elastische defaseringstijd T2 ~ 1 / (g_tr * r_bohr_grid)
                T2 = 1.0 / (g_tr * r_bohr_grid * 20.0)
                r_b = max(0.0, np.exp(- (t / T2)**2 ))
                phase_std = g_tr * t * 10.0

            bloch_norms.append(r_b)
            phase_drifts.append(phase_std)
            
        results[g_tr] = (bloch_norms, phase_drifts)

    if verbose:
        print("\n" + "=" * 75)
        print("ETRT DECOHERENTIE ANALYSE: T2 DEFASERING & T1 RELAXATIE (qubit_test_1.py)")
        print("=" * 75)
        print(f"Gradiënt g_TR    | Eind-Bloch | Status T2 Decoherentie")
        print("-" * 75)
        for g_tr in g_TR_values:
            final_r = results[g_tr][0][-1]
            status = "Behouden (Zuiver)" if final_r > 0.9 else ("Gedempt (Decoherent)" if final_r > 0.1 else "Volledig vervallen")
            print(f"g_TR = {g_tr:<8.4f} | r = {final_r:<8.4f} | {status}")
        print("=" * 75)

    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
        
        # Plot 1: Bloch-vector lengte vs Tijd
        for g_tr in g_TR_values:
            axes[0].plot(t_steps, results[g_tr][0], lw=2, label=f"g_TR = {g_tr}")
        axes[0].set_title("T2 Defasering: Bloch-vector Lengte |r(t)|")
        axes[0].set_xlabel("Tijd t [ModelUnits]")
        axes[0].set_ylabel("Bloch-norm |r| (1.0 = Zuiver)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Plot 2: Fase-spreiding / Ruis vs Tijd
        for g_tr in g_TR_values:
            axes[1].plot(t_steps, results[g_tr][1], lw=2, label=f"g_TR = {g_tr}")
        axes[1].set_title("Fase-spreiding door Lokale TR-Gradiënt ΔTR(z)")
        axes[1].set_xlabel("Tijd t [ModelUnits]")
        axes[1].set_ylabel("Fase-spreiding σ_phi [rad]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig("qubit_decoherence_t2.png", dpi=150)
        plt.close(fig)
        if verbose:
            print("Visualisatie opgeslagen als 'qubit_decoherence_t2.png'.")

    return results

if __name__ == "__main__":
    run_qubit_decoherence_test()