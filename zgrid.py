import numpy as np
from scipy.ndimage import laplace
try:
    from parameters import *
except ImportError:
    from .parameters import *

class ZGrid:
    """
    Het fundamentele vacuümweefsel (V3 Cosserat-rooster).
    Ondersteunt statische topologische defecten (elektronen, quarks) in de niet-lineaire zone,
    én dynamische transversale golven (fotonen) in de lineaire zone.
    """
    def __init__(self, shape=GRID_SHAPE_ELECTRON, dx=DX_ELECTRON, k_tr=K_TR, beta_couple=BETA_COUPLE, rho_z=RHO_Z):
        self.shape = shape
        self.dx = dx
        self.k_tr = k_tr
        self.beta_couple = beta_couple
        
        # Traagheid (massadichtheid) van het vacuüm voor golffoortplanting
        self.rho_z = rho_z 
        
        # Cartesisch rooster genereren
        nx, ny, nz = shape
        origin = -0.5 * dx * np.array([nx - 1, ny - 1, nz - 1])
        
        self.x = origin[0] + dx * np.arange(nx)
        self.y = origin[1] + dx * np.arange(ny)
        self.z = origin[2] + dx * np.arange(nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        
        self.dV = self.dx**3
        
        # Initialiseer het TR-veld als een onverstoord vacuüm (TR = 1.0)
        self.TR = np.ones(self.shape, dtype=float)

    # =========================================================================
    # MOTOR 1: STATISCHE RELAXATIE (Voor Quarks en Nucleonen)
    # =========================================================================
    def relax(self, external_pressure=None, dt=0.05, iterations=500):
        """
        Zoekt het elastische rustpunt via Gradient Descent.
        Gebruikt de ASYMMETRISCHE (1/TR) volume-vergelijking zodat
        expansie en compressie hun correcte natuurkundige gewicht krijgen.
        """
        if external_pressure is None:
            external_pressure = np.zeros_like(self.TR)
        
        for _ in range(iterations):
            laplacian = laplace(self.TR) / (self.dx**2)
            
            # Voorkom delen door absolute nul in de asymmetrische afgeleide
            TR_safe = np.maximum(self.TR, 0.0001)
            
            # ASYMMETRISCHE AFGELEIDE: d/dTR [ (TR-1)^2 / TR ] = (TR^2 - 1) / TR^2
            # Dit vervangt de oude symmetrische 2*(TR-1) en de rigide Lambda_Z muur
            dV_dTR = LAMBDA_ELEKTRON * ((TR_safe**2 - 1.0) / (TR_safe**2))
            
            # Krachtenbalans per cel
            force = (self.k_tr * laplacian - dV_dTR - self.beta_couple * external_pressure)
            
            self.TR += dt * force
            np.clip(self.TR, 0.0001, 20.0, out=self.TR) 

    def get_energy_density(self):
        """
        Berekent de lokale elastische energie (massa).
        Energie = Gradiënt + Asymmetrische Potentiaal
        """
        dTR_dx, dTR_dy, dTR_dz = np.gradient(self.TR, self.dx, self.dx, self.dx)
        e_grad = 0.5 * self.k_tr * (dTR_dx**2 + dTR_dy**2 + dTR_dz**2)
        
        TR_safe = np.maximum(self.TR, 0.0001)
        u = self.TR - 1.0
        
        # ASYMMETRISCHE VOLUMETRISCHE ENERGIE: E_pot = Lambda * Alpha * (u^2 / TR)
        # Zorgt ervoor dat TR < 1 (expansie) relatief zwaarder weegt per grid-cel
        e_pot = LAMBDA_ELEKTRON * ((u**2) / TR_safe)
        
        return e_grad + e_pot

    # =========================================================================
    # MOTOR 2: DYNAMISCHE GOLVEN (Voor het Foton en de lichtsnelheid c)
    # =========================================================================
    def propagate_wave(self, TR_oud, TR_nu, dt=0.04):
        """
        Voert EXACT één tijdstap uit van de d'Alembert golfvergelijking in 3D.
        Gebruikt traagheid (rho_z) en implementeert 3D Absorberende Randen (ABC).
        """
        laplaciaan = laplace(TR_nu) / (self.dx**2)
        kracht = self.k_tr * laplaciaan
        versnelling = kracht / self.rho_z
        TR_nieuw = 2.0 * TR_nu - TR_oud + versnelling * (dt**2)
        
        c_theorie = np.sqrt(self.k_tr / self.rho_z)
        c_factor = c_theorie * (dt / self.dx)
        abc = (c_factor - 1.0) / (c_factor + 1.0)
        
        TR_nieuw[0, :, :] = TR_nu[1, :, :] + abc * (TR_nieuw[1, :, :] - TR_nu[0, :, :])
        TR_nieuw[-1, :, :] = TR_nu[-2, :, :] + abc * (TR_nieuw[-2, :, :] - TR_nu[-1, :, :])
        TR_nieuw[:, 0, :] = TR_nu[:, 1, :] + abc * (TR_nieuw[:, 1, :] - TR_nu[:, 0, :])
        TR_nieuw[:, -1, :] = TR_nu[:, -2, :] + abc * (TR_nieuw[:, -2, :] - TR_nu[:, -1, :])
        TR_nieuw[:, :, 0] = TR_nu[:, :, 1] + abc * (TR_nieuw[:, :, 1] - TR_nu[:, :, 0])
        TR_nieuw[:, :, -1] = TR_nu[:, :, -2] + abc * (TR_nieuw[:, :, -2] - TR_nu[:, :, -1])
        
        return TR_nieuw