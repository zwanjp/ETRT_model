import numpy as np
import matplotlib.pyplot as plt

try:
    from parameters import *
except ImportError:
    from .parameters import *

# =====================================================================
# 1. FYSIEKE SCHAAL BEREKENEN (Vanuit Centrale Parameters)
# =====================================================================
# Bereken de werkelijke schaal van het Z-grid op basis van de unificatie
m_z_kg = M_E * (C_Z / C_E)
y0_m = HBAR / (m_z_kg * C)
xi_m = y0_m / BETA_GEOM  # Universele kernschaal in werkelijke meters

# Voor een leesbare grafiek drukken we de x-as uit in Zeptometers (10^-21 m)
# xi_m = ~1e-23 m, dus ongeveer 0.01 Zeptometer
ZEPTO = 1e21
xi_zm = xi_m * ZEPTO
xi_shell_zm = xi_zm * XI_SHELL_FACTOR

# =====================================================================
# 2. DATA GENEREREN (Echte Fysieke Dimensies)
# =====================================================================
# We plotten tot 15 keer de kernschaal om de rimpel goed te zien
r_zm = np.linspace(0.0001, 15 * xi_zm, 1000)
r_m = r_zm / ZEPTO  # terug naar meters voor de wiskunde

# A. Z-DEELTJE (De harde vacuüm pixel - Betonnen muur)
# GEEN nul-kalibratie (brentq) meer! We gebruiken de keiharde Z_RIDGE_AMPLITUDE
core_z = 1.0 - np.exp(-(r_m / xi_m)**2)
shell_u2 = (r_m / (xi_m * XI_SHELL_FACTOR))**2
shell_z = Z_RIDGE_AMPLITUDE * shell_u2 * np.exp(-shell_u2)
tr_z_particle = core_z + shell_z

# B. HET VRIJE ELEKTRON (1D doorsnede van het Hopfion defect)
# Het uitgesmeerde profiel r/(r+xi)
tr_electron = r_m / (r_m + xi_m)

# =====================================================================
# 3. GRAFIEK OPMAKEN
# =====================================================================
plt.figure(figsize=(10, 6))

# Lijnen
plt.plot(r_zm, tr_electron, label=rf'Vrij Elektron (Zacht defect, $\xi={xi_zm:.4f}$ zm)', 
         color='blue', linestyle='--')
plt.plot(r_zm, tr_z_particle, label=f'Z-deeltje (Harde Kern + Heuvelrug, A={Z_RIDGE_AMPLITUDE})', color='red', linewidth=2)

# Referentielijn Vacuüm
plt.axhline(1.0, color='black', linestyle=':', label='Wiskundig Vacuüm (TR=1)')

# Markeer de Heuvelrug
r_max_idx = np.argmax(tr_z_particle)
tr_max = tr_z_particle[r_max_idx]
r_max_zm = r_zm[r_max_idx]

plt.plot(r_max_zm, tr_max, 'ko')
plt.annotate(f'Top Heuvelrug\nTR ≈ {tr_max:.2f}\nbij r = {r_max_zm:.4f} zm', 
             xy=(r_max_zm, tr_max), 
             xytext=(r_max_zm + (2*xi_zm), tr_max - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=10)

# Vormgeving
plt.title('TR-profiel in Absolute Fysieke Schaal: Elektron vs. Z-cel', fontsize=14, pad=15)
plt.xlabel('Radiale afstand (r) [Zeptometers, $10^{-21}$ m]', fontsize=12)
plt.ylabel('Time Rhythm (TR) [Dimensieloos]', fontsize=12)

# Laat de as netjes lopen tot het einde van onze data
plt.xlim(0, 15 * xi_zm)
plt.ylim(0, max(1.25, tr_max + 0.2))

plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.show()
