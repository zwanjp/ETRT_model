import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def tr_single_particle_sq(r_sq, xi_core, xi_shell, A):
    """Glad enkel-node TR-profiel."""
    core = np.exp(-r_sq / (2.0 * xi_core**2))
    shell = np.exp(-r_sq / (2.0 * xi_shell**2))
    return 1.0 - A * (core - shell)

def get_periodic_diff(dr, L):
    """Nearest-image conventie."""
    return dr - L * np.round(dr / L)

def quadratic_curvature(x, y):
    """Fit y(x) = a*x^2 + b*x + c en geeft d2y/dx2 = 2a terug."""
    coeff = np.polyfit(x, y, deg=2)
    return 2.0 * coeff[0], coeff

# --- Deformatie Hulpfuncties ---

def deformation_gradient_uniaxial(eps_xx=0.0, eps_yy=0.0):
    return np.array([[1.0 + eps_xx, 0.0], [0.0, 1.0 + eps_yy]])

def deformation_gradient_shear(gamma_xy=0.0):
    return np.array([[1.0, 0.5 * gamma_xy], [0.5 * gamma_xy, 1.0]])

def deformation_gradient_isotropic(eps=0.0):
    return np.array([[1.0 + eps, 0.0], [0.0, 1.0 + eps]])

class ZGridElasticity2D:
    """
    2D-periodieke Z-gridcel met vervormbare celmatrix H en interne relaxatie.
    """

    def __init__(self, L, N, params):
        self.L = float(L)
        self.N = int(N)
        self.params = params
        
        # Referentie celmatrix H0 (vierkant)
        self.H0 = np.array([[self.L, 0.0], [0.0, self.L]], dtype=float)
        
        # Basis in fractionele coördinaten
        self.basis_frac = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=float)
        
        # Grid in fractionele coördinaten (u, v) in [0, 1)
        ticks = np.linspace(0, 1, self.N, endpoint=False)
        self.U, self.V = np.meshgrid(ticks, ticks, indexing="xy")

    def cell_matrix(self, F=None):
        if F is None: F = np.eye(2)
        return F @ self.H0

    def cell_area(self, F=None):
        return abs(np.linalg.det(self.cell_matrix(F)))

    def get_basis_positions(self, delta_s=None, F=None):
        """Geeft Cartesische posities van de basisnodes."""
        if delta_s is None: delta_s = np.zeros(2)
        basis = self.basis_frac.copy()
        basis[1] = np.mod(basis[1] + np.asarray(delta_s, dtype=float), 1.0)
        
        H = self.cell_matrix(F)
        # s @ H.T geeft Cartesische x, y
        return basis @ H.T

    def compute_tr_field(self, node_positions, H):
        """Bouwt TR veld op in de vervormde cel."""
        tr_total = np.ones((self.N, self.N), dtype=float)
        
        # Roosterpunten in Cartesische coördinaten: [U, V] @ H.T
        # We flatten voor de berekening
        grid_frac = np.stack([self.U.ravel(), self.V.ravel()], axis=1)
        grid_cart = grid_frac @ H.T
        
        X_cart = grid_cart[:, 0].reshape((self.N, self.N))
        Y_cart = grid_cart[:, 1].reshape((self.N, self.N))

        for pos in node_positions:
            # Voor deformatie moeten we periodieke afstand in fracties doen 
            # en dan naar Cartesisch transformeren via H
            du = get_periodic_diff(self.U - (pos @ np.linalg.inv(H).T)[0], 1.0)
            dv = get_periodic_diff(self.V - (pos @ np.linalg.inv(H).T)[1], 1.0)
            
            # d_cart = [du, dv] @ H.T
            dx_c = du * H[0, 0] + dv * H[0, 1]
            dy_c = du * H[1, 0] + dv * H[1, 1]
            r_sq = dx_c**2 + dy_c**2

            tr_node = tr_single_particle_sq(
                r_sq, self.params["xi_core"], self.params["xi_shell"], self.params["A"]
            )
            tr_total *= tr_node

        return tr_total

    def compute_gradient_energy_deformed(self, tr, H):
        """Graddiënten in fractionele ruimte omzetten naar fysieke ruimte."""
        du_val = 1.0 / self.N
        
        dtr_du = (np.roll(tr, -1, axis=1) - np.roll(tr, 1, axis=1)) / (2.0 * du_val)
        dtr_dv = (np.roll(tr, -1, axis=0) - np.roll(tr, 1, axis=0)) / (2.0 * du_val)

        H_inv_T = np.linalg.inv(H).T
        grad_x = H_inv_T[0, 0] * dtr_du + H_inv_T[0, 1] * dtr_dv
        grad_y = H_inv_T[1, 0] * dtr_du + H_inv_T[1, 1] * dtr_dv

        return 0.5 * (grad_x**2 + grad_y**2)

    def total_energy(self, delta_s=None, F=None):
        if delta_s is None: delta_s = np.zeros(2)
        if F is None: F = np.eye(2)

        H = self.cell_matrix(F)
        node_positions = self.get_basis_positions(delta_s, F)
        
        tr = self.compute_tr_field(node_positions, H)
        e_grad = self.compute_gradient_energy_deformed(tr, H)
        
        chi = self.params.get("chi", 0.0)
        e_pot = chi * (1.0 - tr**2)**2

        dA = self.cell_area(F) / (self.N**2)
        return np.sum(e_grad + e_pot) * dA

    def relax_internal_mode(self, F=None, start_delta_s=None):
        if F is None: F = np.eye(2)
        if start_delta_s is None: start_delta_s = np.zeros(2)

        res = minimize(
            lambda ds: self.total_energy(ds, F),
            x0=np.asarray(start_delta_s),
            method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-11}
        )
        return {"F": F, "delta_s": res.x, "energy": res.fun, "success": res.success}

    def scan_deformation_mode(self, deformation_function, amplitudes):
        results = []
        last_ds = np.zeros(2)
        for amp in amplitudes:
            F = deformation_function(amp)
            relaxed = self.relax_internal_mode(F, start_delta_s=last_ds)
            last_ds = relaxed["delta_s"]
            results.append({
                "amplitude": amp, "energy": relaxed["energy"], 
                "delta_s": relaxed["delta_s"], "area": self.cell_area(F)
            })
        return results

    def extract_elastic_constants(self, scan_xx, scan_iso, scan_shear):
        A0 = self.cell_area()
        E0 = self.total_energy()

        def get_dW(scan):
            x = np.array([r["amplitude"] for r in scan])
            dW = np.array([(r["energy"] - E0) / A0 for r in scan])
            return x, dW

        eps_x, dW_x = get_dW(scan_xx)
        C11, _ = quadratic_curvature(eps_x, dW_x)

        gam, dW_s = get_dW(scan_shear)
        C66, _ = quadratic_curvature(gam, dW_s)

        eps_i, dW_i = get_dW(scan_iso)
        curv_iso, _ = quadratic_curvature(eps_i, dW_i)
        C12 = 0.5 * curv_iso - C11

        return {"C11": C11, "C12": C12, "C66": C66}

# --- Script Uitvoering ---

if __name__ == "__main__":
    params = {"xi_core": 0.5, "xi_shell": 1.2, "A": 0.4146, "chi": 0.25}
    model = ZGridElasticity2D(L=5.2, N=64, params=params)

    amps = np.linspace(-0.02, 0.02, 7)

    print("Scans uitvoeren...")
    s_xx = model.scan_deformation_mode(lambda e: deformation_gradient_uniaxial(eps_xx=e), amps)
    s_iso = model.scan_deformation_mode(lambda e: deformation_gradient_isotropic(e), amps)
    s_shear = model.scan_deformation_mode(lambda g: deformation_gradient_shear(gamma_xy=g), amps)

    C = model.extract_elastic_constants(s_xx, s_iso, s_shear)

    print("=== Effectieve 2D-elastische constanten ===")
    print(f"C11: {C['C11']:.6e}")
    print(f"C12: {C['C12']:.6e}")
    print(f"C66: {C['C66']:.6e}")

    # Diagnostiek voor Shear
    print("--- Shear Scan Diagnostiek ---")
    print(" gamma       Delta E      ds_x         ds_y")
    for r in s_shear:
        print(f"{r['amplitude']:8.4f} {r['energy']-s_shear[3]['energy']:12.4e} {r['delta_s'][0]:12.4e} {r['delta_s'][1]:12.4e}")
