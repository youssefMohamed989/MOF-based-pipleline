"""
Stage 4 — FDTD Electromagnetic Simulation + Active Learning Cost Optimizer
===========================================================================
Implements:
  (A) A semi-analytic LSPR spectrum model (Mie theory + quasi-static
      dipole coupling) to predict LSPR peak wavelength and enhancement
      factor for a gold nanoparticle dimer as a function of geometry.
  (B) A Gaussian Process active learning loop that finds the optimal
      geometry (radius, gap) in far fewer evaluations than a grid search.

In production, replace lspr_spectrum() with a MEEP or Lumerical call.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Physical Constants
# ─────────────────────────────────────────────
C_LIGHT  = 3.0e8     # m/s
EPS_INF  = 9.5       # gold high-freq dielectric
OMEGA_P  = 1.36e16   # rad/s  (gold plasma frequency)
GAMMA_D  = 1.05e14   # rad/s  (gold Drude damping)
EPS_MED  = 1.77      # medium dielectric (water ≈ 1.333²)


# ─────────────────────────────────────────────
# Drude Model for Gold Permittivity
# ─────────────────────────────────────────────
def gold_permittivity(wavelength_nm: float) -> complex:
    """Returns complex permittivity of gold via Drude model."""
    omega = 2 * np.pi * C_LIGHT / (wavelength_nm * 1e-9)
    eps = EPS_INF - OMEGA_P**2 / (omega**2 + 1j * GAMMA_D * omega)
    return eps


# ─────────────────────────────────────────────
# Quasi-static LSPR Spectrum (Mie + dipole coupling)
# ─────────────────────────────────────────────
def lspr_spectrum(
    radius_nm: float,
    gap_nm: float,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Computes LSPR extinction cross-section spectrum for a gold nanoparticle
    dimer using quasi-static Mie theory with near-field dipole coupling.

    Parameters
    ----------
    radius_nm   : NP radius in nm
    gap_nm      : gap between NPs in nm
    wavelengths : array of wavelengths in nm

    Returns
    -------
    C_ext : normalized extinction spectrum (a.u.)
    """
    r = radius_nm * 1e-9
    d = (2 * radius_nm + gap_nm) * 1e-9   # center-to-center distance

    C_ext = np.zeros(len(wavelengths))

    for i, lam in enumerate(wavelengths):
        lam_m = lam * 1e-9
        eps_m = gold_permittivity(lam)
        eps_bg = EPS_MED

        # Quasi-static polarizability (Clausius-Mossotti)
        alpha_0 = 4 * np.pi * r**3 * (eps_m - eps_bg) / (eps_m + 2 * eps_bg)

        # Dipole coupling correction (longitudinal dimer mode)
        # Coupled polarizability: alpha_eff = alpha_0 / (1 - beta * alpha_0)
        # where beta = 1/(4*pi*eps_0*eps_bg) * 2/(d^3)
        eps_0 = 8.854e-12  # F/m
        beta = 2.0 / (4 * np.pi * eps_0 * eps_bg * d**3)
        alpha_eff = alpha_0 / (1.0 - beta * alpha_0 + 1e-30)

        k = 2 * np.pi * np.sqrt(eps_bg) / lam_m
        # Extinction = Im(alpha_eff) * k / eps_0
        C_ext[i] = np.abs(np.imag(alpha_eff) * k)

    # Normalize to peak = 1
    if C_ext.max() > 0:
        C_ext /= C_ext.max()
    return C_ext


def lspr_peak(radius_nm: float, gap_nm: float) -> dict:
    """Returns peak wavelength and max enhancement for a given geometry."""
    wl = np.linspace(400, 800, 400)
    spec = lspr_spectrum(radius_nm, gap_nm, wl)
    peak_idx = np.argmax(spec)
    return {
        "radius_nm": radius_nm,
        "gap_nm":    gap_nm,
        "peak_wl_nm": float(wl[peak_idx]),
        "enhancement": float(spec[peak_idx]),
        "fwhm_nm": _fwhm(wl, spec),
    }


def _fwhm(wl: np.ndarray, spec: np.ndarray) -> float:
    """Compute full-width at half-maximum of spectrum."""
    half = spec.max() / 2
    above = np.where(spec >= half)[0]
    if len(above) < 2:
        return 0.0
    return float(wl[above[-1]] - wl[above[0]])


# ─────────────────────────────────────────────
# Conventional Grid Search
# ─────────────────────────────────────────────
def grid_search(
    radii: np.ndarray,
    gaps: np.ndarray,
    target_wl: float = 650.0,
):
    """
    Exhaustive grid search: evaluates all (radius, gap) combinations.
    Returns results DataFrame sorted by proximity to target wavelength.
    """
    records = []
    n_evals = 0
    for r, g in product(radii, gaps):
        result = lspr_peak(r, g)
        result["cost"] = abs(result["peak_wl_nm"] - target_wl)
        records.append(result)
        n_evals += 1
    df = pd.DataFrame(records).sort_values("cost")
    print(f"  Grid search: {n_evals} evaluations")
    return df, n_evals


# ─────────────────────────────────────────────
# Active Learning Optimizer
# ─────────────────────────────────────────────
class LSPROptimizer:
    """
    Bayesian optimization of LSPR geometry using a GP surrogate.
    Minimizes |peak_wl - target_wl| (a proxy for sensor sensitivity).
    """
    def __init__(self, target_wl: float = 650.0, kappa: float = 2.0):
        self.target_wl = target_wl
        self.kappa = kappa
        kernel = Matern(nu=2.5, length_scale=10.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            normalize_y=True, random_state=42
        )
        self.sc = StandardScaler()
        self.X_obs: list = []
        self.y_obs: list = []
        self.history: list = []

    def _objective(self, radius_nm: float, gap_nm: float) -> float:
        """Negative closeness to target wavelength (we maximize)."""
        p = lspr_peak(radius_nm, gap_nm)
        return -abs(p["peak_wl_nm"] - self.target_wl)   # higher = better

    def _fit(self):
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        X_sc = self.sc.fit_transform(X)
        self.gp.fit(X_sc, y)

    def _ucb(self, X_cand: np.ndarray) -> np.ndarray:
        X_sc = self.sc.transform(X_cand)
        mu, sigma = self.gp.predict(X_sc, return_std=True)
        return mu + self.kappa * sigma

    def optimize(
        self,
        radii: np.ndarray,
        gaps: np.ndarray,
        n_initial: int = 4,
        n_rounds: int = 10,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        all_points = np.array([[r, g] for r, g in product(radii, gaps)])

        # Random initial samples
        init_idx = rng.choice(len(all_points), size=n_initial, replace=False)
        for idx in init_idx:
            r, g = all_points[idx]
            y = self._objective(r, g)
            self.X_obs.append([r, g])
            self.y_obs.append(y)
            self.history.append({"round": 0, "radius_nm": r, "gap_nm": g,
                                  "objective": y, "type": "initial"})

        # AL rounds
        for rnd in range(1, n_rounds + 1):
            self._fit()
            ucb = self._ucb(all_points)
            # Exclude already evaluated
            evaluated = set(map(tuple, self.X_obs))
            mask = np.array([tuple(p) not in evaluated for p in all_points])
            if not mask.any():
                break
            best_idx = np.argmax(np.where(mask, ucb, -1e9))
            r, g = all_points[best_idx]
            y = self._objective(r, g)
            self.X_obs.append([r, g])
            self.y_obs.append(y)
            self.history.append({"round": rnd, "radius_nm": r, "gap_nm": g,
                                  "objective": y, "type": "AL"})

        return pd.DataFrame(self.history)

    def best(self) -> dict:
        best_idx = np.argmax(self.y_obs)
        r, g = self.X_obs[best_idx]
        result = lspr_peak(r, g)
        result["objective"] = self.y_obs[best_idx]
        return result


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
def plot_results(spec_data: dict, al_history: pd.DataFrame,
                 grid_df: pd.DataFrame, al_best: dict, n_grid: int):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#04080f")

    # ── Panel A: LSPR spectra for different geometries
    ax = axes[0]
    ax.set_facecolor("#0a1120")
    cmap = plt.cm.cool
    for i, (label, (wl, spec)) in enumerate(spec_data.items()):
        c = cmap(i / max(len(spec_data) - 1, 1))
        ax.plot(wl, spec, color=c, linewidth=1.5, label=label)
    ax.set_xlabel("Wavelength (nm)", color="#cdd9e5")
    ax.set_ylabel("Normalized Extinction (a.u.)", color="#cdd9e5")
    ax.set_title("LSPR Spectra vs Geometry", color="#fff", fontsize=10)
    ax.tick_params(colors="#5a7a9a")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color("#1a2e4a")
    ax.legend(fontsize=7, facecolor="#0d1829", edgecolor="#1a2e4a",
              labelcolor="#cdd9e5")

    # ── Panel B: AL convergence vs grid search cost
    ax2 = axes[1]
    ax2.set_facecolor("#0a1120")
    # AL cumulative best
    al_cum_best = []
    best_so_far = -np.inf
    for v in al_history["objective"]:
        if v > best_so_far: best_so_far = v
        al_cum_best.append(best_so_far)
    ax2.plot(range(1, len(al_cum_best)+1), [-v for v in al_cum_best],
             color="#39ff14", linewidth=2, label="AI-Accelerated")
    ax2.axhline(grid_df["cost"].min(), color="#5a7a9a",
                linestyle="--", linewidth=1.5, label="Grid best")
    ax2.axvline(n_grid, color="#ff6b35", linestyle=":",
                linewidth=1, label=f"Grid cost ({n_grid} evals)")
    ax2.set_xlabel("Number of Evaluations", color="#cdd9e5")
    ax2.set_ylabel("|peak_wl − target| (nm)", color="#cdd9e5")
    ax2.set_title("Active Learning vs Grid Search Cost", color="#fff", fontsize=10)
    ax2.tick_params(colors="#5a7a9a")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax2.spines[sp].set_color("#1a2e4a")
    ax2.legend(fontsize=8, facecolor="#0d1829", edgecolor="#1a2e4a",
               labelcolor="#cdd9e5")

    # ── Panel C: Optimal spectrum
    ax3 = axes[2]
    ax3.set_facecolor("#0a1120")
    wl = np.linspace(400, 800, 400)
    best_spec = lspr_spectrum(al_best["radius_nm"], al_best["gap_nm"], wl)
    ax3.fill_between(wl, best_spec, alpha=0.25, color="#cc2222")
    ax3.plot(wl, best_spec, color="#cc2222", linewidth=2)
    ax3.axvline(al_best["peak_wl_nm"], color="#ffd60a",
                linestyle="--", linewidth=1.5,
                label=f"Peak: {al_best['peak_wl_nm']:.0f} nm")
    ax3.set_xlabel("Wavelength (nm)", color="#cdd9e5")
    ax3.set_ylabel("Normalized Extinction (a.u.)", color="#cdd9e5")
    ax3.set_title(f"Optimal Spectrum\nr={al_best['radius_nm']} nm, "
                  f"gap={al_best['gap_nm']} nm", color="#fff", fontsize=10)
    ax3.tick_params(colors="#5a7a9a")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    for sp in ["bottom","left"]:
        ax3.spines[sp].set_color("#1a2e4a")
    ax3.legend(fontsize=8, facecolor="#0d1829", edgecolor="#1a2e4a",
               labelcolor="#cdd9e5")

    plt.tight_layout()
    plt.savefig("stage4_fdtd.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  ✓ Saved: stage4_fdtd.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 4 — FDTD Electromagnetic Simulation")
    print("=" * 60)

    # Parameter space
    radii = np.arange(20, 80, 10)   # nm
    gaps  = np.arange(5,  40, 5)    # nm
    TARGET_WL = 650.0               # nm (desired LSPR peak for GPC3 sensing)
    wl_arr = np.linspace(400, 800, 400)

    # ── Sample spectra for visualization
    print("\n[1] Computing sample LSPR spectra...")
    spec_data = {}
    for r, g in [(30, 10), (50, 10), (50, 20), (70, 10)]:
        key = f"r={r}nm, gap={g}nm"
        spec_data[key] = (wl_arr, lspr_spectrum(r, g, wl_arr))
        p = lspr_peak(r, g)
        print(f"  {key} → peak {p['peak_wl_nm']:.1f} nm, FWHM {p['fwhm_nm']:.1f} nm")

    # ── Conventional grid search
    print("\n[2] Running conventional grid search...")
    grid_df, n_grid = grid_search(radii, gaps, target_wl=TARGET_WL)
    print(f"  Best grid result:")
    print(grid_df.head(3).to_string(index=False))

    # ── Active learning optimization
    print("\n[3] Running Active Learning optimization...")
    opt = LSPROptimizer(target_wl=TARGET_WL, kappa=2.0)
    al_history = opt.optimize(radii, gaps, n_initial=4, n_rounds=10)
    al_best = opt.best()
    n_al = len(al_history)
    print(f"  AL used {n_al} evaluations vs {n_grid} for grid")
    print(f"  Cost reduction: {n_grid/n_al:.1f}×")
    print(f"  Best geometry: radius={al_best['radius_nm']} nm, gap={al_best['gap_nm']} nm")
    print(f"  Best peak: {al_best['peak_wl_nm']:.1f} nm (target {TARGET_WL} nm)")

    # ── Plot
    print("\n[4] Generating plots...")
    plot_results(spec_data, al_history, grid_df, al_best, n_grid)

    # Export
    grid_df.to_csv("stage4_grid_results.csv", index=False)
    al_history.to_csv("stage4_al_history.csv", index=False)
    pd.DataFrame([al_best]).to_csv("stage4_optimal_geometry.csv", index=False)
    print("\n✓ Stage 4 complete.")
    print(f"  → Optimal geometry exported to stage4_optimal_geometry.csv")
