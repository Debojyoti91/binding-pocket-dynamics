# 🧬 Binding Pocket Detection & Molecular Dynamics Analysis

This repository presents a complete pipeline for detecting and analyzing protein binding pockets using **Fpocket** and evaluating their stability and druggability through **Molecular Dynamics (MD) simulations** with **GROMACS** and **MDAnalysis**.

## 🔬 Task Overview

- **Target**: Protein of interest (e.g., PD-1)
- **Goal**: Detect potential binding pockets and analyze their dynamic behavior under simulated physiological conditions.

## 🧪 Workflow Summary

1. **Binding Pocket Detection (Fpocket)**
   - Identification of potential binding pockets on the protein surface
   - Characterization of pockets by volume, polarity, and hydrophobicity
   - Selection of the most promising pocket for further MD analysis

2. **Molecular Dynamics Simulation (GROMACS)**
   - System preparation: protonation, solvation, neutralization
   - 100 ns MD simulation of the protein system
   - Analysis:
     - Pocket stability
     - RMSD, RMSF (for pocket residues)
     - SASA
     - Changes in pocket shape, size, and accessibility over time

## 📁 Contents

- `binding_pocket_fpocket.py` – Fpocket analysis and visualization
- `gromacs_md_simulation_mdanalysis.py` – MD simulation analysis using MDAnalysis
- `combined_fpocket_gromacs_analysis.py` – Unified pipeline
