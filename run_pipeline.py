"""
run_pipeline.py — Master Pipeline Runner
==========================================
Executes all 6 stages of the AI-Accelerated MOF-to-Device Pipeline
in sequence, collecting outputs and printing a final summary.

Usage:
    python run_pipeline.py

Requirements:
    pip install numpy pandas matplotlib scikit-learn scipy
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd


def section(title: str):
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def run_stage(stage_num: int, label: str, fn) -> bool:
    section(f"STAGE {stage_num} — {label}")
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✓ Completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"\n  ✗ Stage {stage_num} failed: {e}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════
# Stage 1 — MOF Synthesis & Featurization
# ══════════════════════════════════════════════════════════════
def stage1():
    from stage1_mof_synthesis import (
        MOF_LIBRARY, build_feature_matrix, normalize
    )
    df = build_feature_matrix(MOF_LIBRARY)
    df_norm, mins, maxs = normalize(df)
    df.to_csv("mof_features_raw.csv")
    df_norm.to_csv("mof_features_normalized.csv")
    print(f"  MOFs featurized: {len(MOF_LIBRARY)}")
    print(f"  Features per MOF: {df.shape[1]}")
    print(df.to_string())


# ══════════════════════════════════════════════════════════════
# Stage 2 — ML Screening
# ══════════════════════════════════════════════════════════════
def stage2():
    import warnings; warnings.filterwarnings("ignore")
    from stage2_ml_screening import (
        GPRanker, active_learning_simulation,
        loocv_evaluation, plot_rankings,
        EXPERIMENTAL_SCORES
    )
    from stage1_mof_synthesis import (
        build_feature_matrix, normalize, MOF_LIBRARY
    )

    df_feat, _, _ = normalize(build_feature_matrix(MOF_LIBRARY))
    names = list(df_feat.index)
    X = df_feat.values
    y = np.array([EXPERIMENTAL_SCORES[n] for n in names])

    ranker = GPRanker(kappa=1.96)
    ranker.fit(X, y)
    rank_df = ranker.rank(X, names)
    print("\n  Candidate Rankings:")
    print(rank_df.to_string())

    loocv_df, rmse = loocv_evaluation(X, y, names)
    print(f"\n  LOO-CV RMSE: {rmse:.4f}")

    al_history = active_learning_simulation(X, y, names, n_initial=2, n_rounds=3)
    print("\n  Active Learning History:")
    print(al_history.to_string(index=False))

    plot_rankings(rank_df, loocv_df)
    rank_df.to_csv("stage2_rankings.csv")
    al_history.to_csv("stage2_al_history.csv", index=False)

    top = rank_df["MOF"].iloc[0]
    print(f"\n  → Selected candidate: {top}")
    return top


# ══════════════════════════════════════════════════════════════
# Stage 3 — Molecular Docking
# ══════════════════════════════════════════════════════════════
def stage3():
    from stage3_molecular_docking import (
        ZN_GA_POSES, PBPA2_BINDING_SITES,
        md_trajectory, plot_docking_results
    )
    import warnings; warnings.filterwarnings("ignore")

    records = []
    for pose in ZN_GA_POSES:
        score = pose.vina_score()
        records.append({
            "pose":        pose.name,
            "site":        pose.site,
            "n_contacts":  pose.n_contacts,
            "n_hbonds":    pose.n_hbonds,
            "vina_score":  score,
        })
    scores_df = pd.DataFrame(records).sort_values("vina_score")
    top = scores_df.iloc[0]

    print(f"\n  Best pose : {top['pose']}")
    print(f"  Site      : {top['site']}")
    print(f"  Vina score: {top['vina_score']} kcal/mol")
    print(f"  Residues  : {', '.join(PBPA2_BINDING_SITES[top['site']]['residues'])}")
    print("\n  All poses:")
    print(scores_df.to_string(index=False))

    t, rmsd = md_trajectory(n_steps=4000)
    mean_rmsd = rmsd[len(rmsd)//4:].mean()
    print(f"\n  MD steady-state RMSD: {mean_rmsd:.3f} Å")

    plot_docking_results(scores_df, t, rmsd)
    scores_df.to_csv("stage3_docking_scores.csv", index=False)


# ══════════════════════════════════════════════════════════════
# Stage 4 — FDTD Simulation
# ══════════════════════════════════════════════════════════════
def stage4():
    import warnings; warnings.filterwarnings("ignore")
    from stage4_fdtd_simulation import (
        lspr_spectrum, lspr_peak,
        grid_search, LSPROptimizer, plot_results
    )

    radii = np.arange(20, 80, 10)
    gaps  = np.arange(5,  40, 5)
    wl_arr = np.linspace(400, 800, 400)
    TARGET_WL = 650.0

    spec_data = {}
    for r, g in [(30, 10), (50, 10), (50, 20), (70, 10)]:
        key = f"r={r}nm, gap={g}nm"
        spec_data[key] = (wl_arr, lspr_spectrum(r, g, wl_arr))

    grid_df, n_grid = grid_search(radii, gaps, target_wl=TARGET_WL)

    opt = LSPROptimizer(target_wl=TARGET_WL, kappa=2.0)
    al_history = opt.optimize(radii, gaps, n_initial=4, n_rounds=10)
    al_best = opt.best()
    n_al = len(al_history)

    print(f"\n  Grid search: {n_grid} evaluations")
    print(f"  AL search:   {n_al} evaluations  ({n_grid/n_al:.1f}× reduction)")
    print(f"  Optimal geometry: r={al_best['radius_nm']} nm, gap={al_best['gap_nm']} nm")
    print(f"  LSPR peak: {al_best['peak_wl_nm']:.1f} nm (target {TARGET_WL} nm)")

    plot_results(spec_data, al_history, grid_df, al_best, n_grid)
    grid_df.to_csv("stage4_grid_results.csv", index=False)
    al_history.to_csv("stage4_al_history.csv", index=False)
    pd.DataFrame([al_best]).to_csv("stage4_optimal_geometry.csv", index=False)


# ══════════════════════════════════════════════════════════════
# Stages 5 & 6 — Chip Fabrication + Validation
# ══════════════════════════════════════════════════════════════
def stage5_6():
    import warnings; warnings.filterwarnings("ignore")
    from scipy.optimize import curve_fit
    from stage5_6_chip_and_validation import (
        ChipDesign,
        generate_calibration_data,
        generate_lspr_data,
        simulate_mouse_model,
        langmuir_response, hill_response,
        lod_loq, compute_roc_metrics,
        plot_validation,
    )

    # Stage 5
    chip = ChipDesign()
    summary = chip.summary()
    print("\n  Chip Design:")
    print(summary.to_string(index=False))
    chip.to_gcode("stage5_chip.gcode")
    summary.to_csv("stage5_chip_design.csv", index=False)

    # Stage 6
    concentrations = np.array([0, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100], dtype=float)

    echem_df = generate_calibration_data(concentrations)
    popt_e, pcov_e = curve_fit(langmuir_response,
                                echem_df["concentration_ng_mL"],
                                echem_df["current_uA"],
                                p0=[30.0, 10.0], maxfev=5000)
    echem_fit = {"popt": popt_e, "pcov": pcov_e}

    lspr_df = generate_lspr_data(concentrations)
    popt_l, pcov_l = curve_fit(hill_response,
                                lspr_df["concentration_ng_mL"],
                                lspr_df["lspr_shift_nm"],
                                p0=[18.0, 6.0, 1.0], maxfev=5000)
    lspr_fit = {"popt": popt_l}

    lod_info = lod_loq(concentrations, echem_df["current_uA"].values)

    mouse_df = simulate_mouse_model()
    sensor_scores = langmuir_response(mouse_df["gpc3_ng_mL"].values, *popt_e)
    roc_info = compute_roc_metrics(mouse_df["label"].values, sensor_scores)

    print(f"\n  Electrochemical LOD : {lod_info['LOD_ng_mL']} ng/mL")
    print(f"  ROC AUC             : {roc_info['auc']:.4f}")
    print(f"  Sensitivity         : {roc_info['sensitivity']*100:.1f}%")
    print(f"  Specificity         : {roc_info['specificity']*100:.1f}%")

    plot_validation(echem_df, lspr_df, echem_fit, lspr_fit,
                    lod_info, roc_info, mouse_df)

    echem_df.to_csv("stage6_echem_calibration.csv", index=False)
    lspr_df.to_csv("stage6_lspr_calibration.csv", index=False)
    mouse_df.to_csv("stage6_mouse_model.csv", index=False)
    pd.DataFrame([{
        "AUC":                    roc_info["auc"],
        "sensitivity":            roc_info["sensitivity"],
        "specificity":            roc_info["specificity"],
        "PPV":                    roc_info["ppv"],
        "NPV":                    roc_info["npv"],
        "threshold_ng_mL":       roc_info["optimal_threshold_ng_mL"],
        "LOD_ng_mL":              lod_info["LOD_ng_mL"],
        "LOQ_ng_mL":              lod_info["LOQ_ng_mL"],
    }]).to_csv("stage6_performance_summary.csv", index=False)

    return roc_info, lod_info


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    print("\n" + "█" * 60)
    print("  AI-ACCELERATED MOF-TO-DEVICE PIPELINE")
    print("  Full Pipeline Runner")
    print("█" * 60)

    results = {}
    ok = [
        run_stage(1, "MOF Synthesis & Featurization",  stage1),
        run_stage(2, "ML-Guided Structure Screening",  stage2),
        run_stage(3, "Molecular Docking Validation",   stage3),
        run_stage(4, "FDTD Electromagnetic Simulation",stage4),
        run_stage(5, "Chip Fabrication + Validation",  stage5_6),
    ]

    total = time.time() - t_start

    section("PIPELINE COMPLETE — SUMMARY")
    passed = sum(ok)
    print(f"\n  Stages passed : {passed} / 5")
    print(f"  Total runtime : {total:.1f}s")
    print(f"\n  Output files generated:")
    outputs = [
        "mof_features_raw.csv",
        "mof_features_normalized.csv",
        "stage2_rankings.csv",
        "stage2_al_history.csv",
        "stage2_ml_screening.png",
        "stage3_docking_scores.csv",
        "stage3_md_trajectory.csv",
        "stage3_docking.png",
        "stage4_grid_results.csv",
        "stage4_al_history.csv",
        "stage4_optimal_geometry.csv",
        "stage4_fdtd.png",
        "stage5_chip_design.csv",
        "stage5_chip.gcode",
        "stage6_echem_calibration.csv",
        "stage6_lspr_calibration.csv",
        "stage6_mouse_model.csv",
        "stage6_performance_summary.csv",
        "stage6_validation.png",
    ]
    for f in outputs:
        print(f"    {f}")

    sys.exit(0 if all(ok) else 1)
