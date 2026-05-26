"""
Stage 2 — ML-Guided Structure Screening with Active Learning
=============================================================
Trains a Gaussian Process surrogate model on MOF feature vectors
to predict biosensor binding score. Uses Upper Confidence Bound (UCB)
acquisition to suggest the next candidate to synthesise/test.

Requires: stage1_mof_synthesis.py output CSVs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Simulated Experimental Binding Scores
# (would come from wet-lab assay in real use)
# ─────────────────────────────────────────────
EXPERIMENTAL_SCORES: dict = {
    "Zn-GA MOF":    0.92,   # top performer → selected
    "Eu-Based MOF": 0.74,
    "Zn-BDC MOF":  0.61,
    "Cu-TCPP MOF": 0.48,
    "Fe-MIL-101":  0.38,
}


# ─────────────────────────────────────────────
# Gaussian Process Ranker
# ─────────────────────────────────────────────
class GPRanker:
    def __init__(self, kappa: float = 1.96):
        """
        kappa: exploration–exploitation trade-off for UCB.
               Higher kappa → more exploration.
        """
        self.kappa = kappa
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.X_obs: np.ndarray | None = None
        self.y_obs: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_obs = X
        self.y_obs = y
        X_sc = self.scaler.fit_transform(X)
        self.gp.fit(X_sc, y)

    def predict(self, X: np.ndarray):
        X_sc = self.scaler.transform(X)
        mu, sigma = self.gp.predict(X_sc, return_std=True)
        return mu, sigma

    def ucb_score(self, X: np.ndarray) -> np.ndarray:
        """Upper Confidence Bound acquisition function."""
        mu, sigma = self.predict(X)
        return mu + self.kappa * sigma

    def rank(self, X: np.ndarray, names: list) -> pd.DataFrame:
        mu, sigma = self.predict(X)
        ucb = self.ucb_score(X)
        df = pd.DataFrame({
            "MOF": names,
            "predicted_score": mu.round(4),
            "uncertainty": sigma.round(4),
            "UCB": ucb.round(4),
        }).sort_values("UCB", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "rank"
        return df


# ─────────────────────────────────────────────
# Active Learning Loop Simulation
# ─────────────────────────────────────────────
def active_learning_simulation(
    X: np.ndarray,
    y: np.ndarray,
    names: list,
    n_initial: int = 2,
    n_rounds: int = 3,
) -> pd.DataFrame:
    """
    Simulates an active learning loop:
    - Start with n_initial observed samples
    - Each round: fit GP → pick best unobserved by UCB → 'observe' its score
    - Returns history of discoveries
    """
    rng = np.random.default_rng(0)
    observed_idx = list(rng.choice(len(names), size=n_initial, replace=False))
    unobserved_idx = [i for i in range(len(names)) if i not in observed_idx]

    ranker = GPRanker(kappa=1.96)
    history = []

    for round_num in range(1, n_rounds + 1):
        X_obs = X[observed_idx]
        y_obs = y[observed_idx]
        ranker.fit(X_obs, y_obs)

        if not unobserved_idx:
            break

        X_unobs = X[unobserved_idx]
        ucb_vals = ranker.ucb_score(X_unobs)
        best_local = int(np.argmax(ucb_vals))
        best_global = unobserved_idx[best_local]

        history.append({
            "round": round_num,
            "suggested": names[best_global],
            "true_score": float(y[best_global]),
            "n_observed": len(observed_idx),
        })

        observed_idx.append(best_global)
        unobserved_idx.remove(best_global)

    return pd.DataFrame(history)


# ─────────────────────────────────────────────
# Leave-One-Out CV Evaluation
# ─────────────────────────────────────────────
def loocv_evaluation(X: np.ndarray, y: np.ndarray, names: list) -> pd.DataFrame:
    loo = LeaveOneOut()
    preds = []
    ranker = GPRanker()
    sc = StandardScaler()

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        X_tr_sc = sc.fit_transform(X_tr)
        X_te_sc = sc.transform(X_te)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=5, random_state=42)
        gp.fit(X_tr_sc, y_tr)
        mu, sigma = gp.predict(X_te_sc, return_std=True)
        preds.append({
            "MOF": names[test_idx[0]],
            "true_score": float(y[test_idx[0]]),
            "predicted_score": float(mu[0]),
            "uncertainty": float(sigma[0]),
        })

    df = pd.DataFrame(preds)
    rmse = np.sqrt(mean_squared_error(df["true_score"], df["predicted_score"]))
    print(f"  LOO-CV RMSE: {rmse:.4f}")
    return df, rmse


# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
def plot_rankings(rank_df: pd.DataFrame, loocv_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#04080f")

    # ── Panel A: Bar chart of predicted scores
    ax = axes[0]
    ax.set_facecolor("#0a1120")
    colors = ["#00d4ff" if i == 0 else "#1a2e4a" for i in range(len(rank_df))]
    bars = ax.barh(rank_df["MOF"][::-1], rank_df["predicted_score"][::-1],
                   xerr=rank_df["uncertainty"][::-1],
                   color=colors[::-1], edgecolor="#1a2e4a",
                   error_kw=dict(ecolor="#ff6b35", capsize=4))
    ax.set_xlabel("Predicted Binding Score", color="#cdd9e5")
    ax.set_title("ML Candidate Ranking (GP Surrogate)", color="#fff", fontsize=11)
    ax.tick_params(colors="#5a7a9a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["bottom","left"]:
        ax.spines[spine].set_color("#1a2e4a")
    ax.set_xlim(0, 1.15)
    ax.axvline(rank_df["predicted_score"].iloc[0], color="#00d4ff",
               linestyle="--", linewidth=1, alpha=0.5)

    # ── Panel B: LOO-CV parity plot
    ax2 = axes[1]
    ax2.set_facecolor("#0a1120")
    ax2.scatter(loocv_df["true_score"], loocv_df["predicted_score"],
                c="#00d4ff", s=80, zorder=5, edgecolors="#fff", linewidths=0.5)
    lims = [0, 1.05]
    ax2.plot(lims, lims, "--", color="#5a7a9a", linewidth=1)
    for _, row in loocv_df.iterrows():
        ax2.annotate(row["MOF"], (row["true_score"], row["predicted_score"]),
                     fontsize=7, color="#5a7a9a",
                     xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("True Score", color="#cdd9e5")
    ax2.set_ylabel("Predicted Score", color="#cdd9e5")
    ax2.set_title("LOO-CV Parity Plot", color="#fff", fontsize=11)
    ax2.tick_params(colors="#5a7a9a")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for spine in ["bottom","left"]:
        ax2.spines[spine].set_color("#1a2e4a")
    ax2.set_xlim(0, 1.05); ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig("stage2_ml_screening.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("  ✓ Saved: stage2_ml_screening.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 2 — ML-Guided Structure Screening")
    print("=" * 60)

    # Load features from Stage 1 (or reconstruct inline)
    try:
        df_feat = pd.read_csv("mof_features_normalized.csv", index_col="name")
    except FileNotFoundError:
        # Inline fallback: re-import from stage 1
        import sys; sys.path.insert(0, ".")
        from stage1_mof_synthesis import build_feature_matrix, normalize, MOF_LIBRARY
        df_feat, _, _ = normalize(build_feature_matrix(MOF_LIBRARY))

    names = list(df_feat.index)
    X = df_feat.values
    y = np.array([EXPERIMENTAL_SCORES[n] for n in names])

    # ── Full GP fit on all data
    print("\n[1] Fitting GP surrogate on all observed data...")
    ranker = GPRanker(kappa=1.96)
    ranker.fit(X, y)
    rank_df = ranker.rank(X, names)
    print("\nCandidate Rankings:")
    print(rank_df.to_string())

    # ── LOO-CV
    print("\n[2] Leave-One-Out Cross-Validation...")
    loocv_df, rmse = loocv_evaluation(X, y, names)
    print(loocv_df.to_string(index=False))

    # ── Active learning simulation
    print("\n[3] Active Learning Loop Simulation (n_initial=2, rounds=3)...")
    al_history = active_learning_simulation(X, y, names, n_initial=2, n_rounds=3)
    print(al_history.to_string(index=False))

    # ── Plot
    print("\n[4] Generating plots...")
    plot_rankings(rank_df, loocv_df)

    # Export
    rank_df.to_csv("stage2_rankings.csv")
    al_history.to_csv("stage2_al_history.csv", index=False)
    print("\n✓ Stage 2 complete. Top candidate:", rank_df["MOF"].iloc[0])
