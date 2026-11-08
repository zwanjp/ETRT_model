import matplotlib.pyplot as plt
import numpy as np

def plot_1d_slice(z_grid, torsion_density, title="Z-Grid Veldenprofiel (Doorsnede X-as)"):
    """
    Snijdt het 3D grid precies door het midden (Y- en Z-as) en plot 
    het verloop over de X-as voor TR, Torsie en Energie.
    """
    print(f"\nGenereren van grafiek: {title}...")
    
    # Bepaal het midden van het rooster
    mid_y = z_grid.shape[1] // 2
    mid_z = z_grid.shape[2] // 2
    
    x_axis = z_grid.x
    
    # 1D slices extraheren
    tr_slice = z_grid.TR[:, mid_y, mid_z]
    torsion_slice = torsion_density[:, mid_y, mid_z]
    energy_slice = z_grid.get_energy_density()[:, mid_y, mid_z]
    
    # Plot instellen (3 onder elkaar)
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', color='darkblue')
    
    # 1. TR Verloop (Time Rhythm)
    axs[0].plot(x_axis, tr_slice, color='blue', lw=2)
    axs[0].axhline(1.0, color='gray', linestyle='--') # Vacuüm referentie
    axs[0].set_ylabel('Time Rhythm (TR)', fontweight='bold')
    axs[0].set_title('Topologische Vervorming in de Ruimtetijd')
    axs[0].grid(True, alpha=0.4)
    # Voorkom dat extreme pieken de as onleesbaar maken
    axs[0].set_ylim(max(0, np.min(tr_slice) - 0.2), min(3.0, np.max(tr_slice) + 0.2))
    
    # 2. Torsiedruk (De Bron)
    axs[1].plot(x_axis, torsion_slice, color='purple', lw=2)
    axs[1].axhline(0, color='gray', linestyle='--')
    axs[1].set_ylabel('Torsiedruk', fontweight='bold')
    axs[1].set_title('Ingebrachte Topologische Lading (+ Compressie, - Expansie)')
    axs[1].grid(True, alpha=0.4)
    
    # 3. Energie-dichtheid (De resulterende Massa)
    axs[2].plot(x_axis, energy_slice, color='red', lw=2)
    axs[2].axhline(0, color='gray', linestyle='--')
    axs[2].set_ylabel('Energie (Massa)', fontweight='bold')
    axs[2].set_title('Elastische Energiedichtheid (E = afgeleide veld)')
    axs[2].set_xlabel('Afstand in het Z-Grid (X-as)', fontweight='bold')
    axs[2].grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.show()