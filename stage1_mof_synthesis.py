"""
Stage 1 — MOF Synthesis & Featurization
========================================
Defines MOF candidate structures as feature vectors for downstream ML screening.
Features include: metal center electronegativity, pore size, surface area,
ligand donor count, UV absorption wavelength, and synthesis yield.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict


# ─────────────────────────────────────────────
# MOF Dataclass
# ─────────────────────────────────────────────
@dataclass
class MOFCandidate:
    name: str
    metal: str
    ligand: str
    pore_size_angstrom: float       # Å
    BET_surface_area_m2g: float     # m²/g
    metal_electronegativity: float  # Pauling scale
    donor_atom_count: int           # number of coordinating atoms per ligand
    uv_absorption_nm: float         # primary UV/vis absorption peak
    synthesis_yield_pct: float      # experimental yield %
    luminescent: bool = False
    notes: str = ""

    def feature_vector(self) -> np.ndarray:
        """Returns normalized numeric feature vector for ML input."""
        return np.array([
            self.pore_size_angstrom,
            self.BET_surface_area_m2g,
            self.metal_electronegativity,
            self.donor_atom_count,
            self.uv_absorption_nm,
            self.synthesis_yield_pct,
            float(self.luminescent),
        ])

    def feature_names(self) -> List[str]:
        return [
            "pore_size_A",
            "BET_m2g",
            "metal_EN",
            "donor_atoms",
            "uv_abs_nm",
            "yield_pct",
            "luminescent",
        ]


# ─────────────────────────────────────────────
# Candidate Library
# ─────────────────────────────────────────────
MOF_LIBRARY: List[MOFCandidate] = [
    MOFCandidate(
        name="Zn-GA MOF",
        metal="Zn",
        ligand="Gallic acid",
        pore_size_angstrom=8.4,
        BET_surface_area_m2g=312.0,
        metal_electronegativity=1.65,
        donor_atom_count=4,
        uv_absorption_nm=280.0,
        synthesis_yield_pct=78.0,
        luminescent=False,
        notes="Primary candidate; UV-active white-yellow solid",
    ),
    MOFCandidate(
        name="Eu-Based MOF",
        metal="Eu",
        ligand="Pyrophosphate + OPs",
        pore_size_angstrom=6.1,
        BET_surface_area_m2g=188.0,
        metal_electronegativity=1.20,
        donor_atom_count=8,
        uv_absorption_nm=394.0,
        synthesis_yield_pct=65.0,
        luminescent=True,
        notes="Strong red luminescence under UV; lanthanide-based",
    ),
    MOFCandidate(
        name="Zn-BDC MOF",
        metal="Zn",
        ligand="Benzene-1,4-dicarboxylic acid",
        pore_size_angstrom=10.2,
        BET_surface_area_m2g=1020.0,
        metal_electronegativity=1.65,
        donor_atom_count=4,
        uv_absorption_nm=255.0,
        synthesis_yield_pct=82.0,
        luminescent=False,
        notes="High surface area; MOF-5 analogue",
    ),
    MOFCandidate(
        name="Cu-TCPP MOF",
        metal="Cu",
        ligand="Tetrakis(4-carboxyphenyl)porphyrin",
        pore_size_angstrom=7.6,
        BET_surface_area_m2g=490.0,
        metal_electronegativity=1.90,
        donor_atom_count=8,
        uv_absorption_nm=418.0,
        synthesis_yield_pct=55.0,
        luminescent=False,
        notes="Porphyrin-based; Soret band absorption",
    ),
    MOFCandidate(
        name="Fe-MIL-101",
        metal="Fe",
        ligand="Terephthalic acid",
        pore_size_angstrom=29.0,
        BET_surface_area_m2g=3200.0,
        metal_electronegativity=1.83,
        donor_atom_count=6,
        uv_absorption_nm=300.0,
        synthesis_yield_pct=70.0,
        luminescent=False,
        notes="Mesoporous; excellent for large-molecule capture",
    ),
]


# ─────────────────────────────────────────────
# Featurize Library → DataFrame
# ─────────────────────────────────────────────
def build_feature_matrix(library: List[MOFCandidate]) -> pd.DataFrame:
    rows = []
    for mof in library:
        row = {"name": mof.name}
        for fname, fval in zip(mof.feature_names(), mof.feature_vector()):
            row[fname] = fval
        rows.append(row)
    df = pd.DataFrame(rows).set_index("name")
    return df


# ─────────────────────────────────────────────
# Normalizer (min-max per feature)
# ─────────────────────────────────────────────
def normalize(df: pd.DataFrame):
    mins = df.min()
    maxs = df.max()
    return (df - mins) / (maxs - mins + 1e-9), mins, maxs


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STAGE 1 — MOF Synthesis & Featurization")
    print("=" * 60)

    df = build_feature_matrix(MOF_LIBRARY)
    print("\nRaw Feature Matrix:")
    print(df.to_string())

    df_norm, mins, maxs = normalize(df)
    print("\nNormalized Feature Matrix:")
    print(df_norm.round(4).to_string())

    # Export for Stage 2
    df.to_csv("mof_features_raw.csv")
    df_norm.to_csv("mof_features_normalized.csv")
    print("\n✓ Saved: mof_features_raw.csv, mof_features_normalized.csv")
