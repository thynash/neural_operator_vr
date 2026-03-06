# Neural Operator Variance Reduction - Publication Package

## Overview
Complete publication package for "Variance Reduction Methods for Training Neural Operators on Chaotic Dynamical Systems"

## Structure

```
FINAL_PUBLICATION/
├── plots/                    # All generated plots
│   ├── 1_lorenz_system/     # Lorenz attractor visualizations
│   ├── 2_logistic_map/      # Logistic map bifurcations
│   ├── 3_operator_analysis/ # FNO/DeepONet architectures
│   ├── 4_optimizer_results/ # Performance comparisons
│   └── 5_combined_figures/  # Main paper figures
│
├── paper/                    # Paper materials
│   ├── PAPER_OUTLINE.md     # Complete paper structure
│   └── figures/             # Figures for paper (PDF)
│
├── presentation/             # Presentation materials
│   ├── PPT_OUTLINE.md       # Slide-by-slide guide
│   └── figures/             # Figures for slides (PNG)
│
└── data/                     # Experimental data
    ├── comprehensive_results.csv
    ├── COMPREHENSIVE_SUMMARY.md
    └── [analysis plots]
```

## Quick Start

### For Paper Writing
1. Open `paper/PAPER_OUTLINE.md` for structure
2. Use figures from `paper/figures/` (PDF format)
3. Reference data from `data/comprehensive_results.csv`

### For Presentation
1. Open `presentation/PPT_OUTLINE.md` for slide structure
2. Use figures from `presentation/figures/` (PNG format)
3. Follow timing and delivery tips in outline

### For Exploring Plots
Browse `plots/` subdirectories:
- Each plot available in PNG (high-res) and PDF (vector)
- Organized by category for easy selection

## Key Results

### Lorenz System with FNO

| Optimizer | Loss      | Time    | Variance  | Spectral |
|-----------|-----------|---------|-----------|----------|
| Adam      | 3.86e-06  | 58.9s   | 3.08e-05  | 0.993    |
| SGD       | 2.07e-07  | 428.4s  | 1.80e-06  | 0.996    |
| SVRG      | 2.39e-06  | 1370.4s | 2.44e-07  | 0.930    |

### Key Findings
- **Best Accuracy**: SGD (2.07×10⁻⁷)
- **Fastest**: Adam (58.9s)
- **Lowest Variance**: SVRG (99.2% reduction)
- **Most Stable**: SVRG (spectral radius 0.930)

## Figures Guide

### Main Paper Figures
1. **fig1_systems_overview.pdf** - Lorenz + Logistic systems
2. **fig2_main_results.pdf** - 4-panel optimizer comparison

### Supplementary Figures
- Lorenz 3D attractor
- Logistic bifurcation diagram
- Operator architectures
- Trade-off analyses

## Citation

```bibtex
@article{neural_operator_vr_2024,
  title={Variance Reduction Methods for Training Neural Operators on Chaotic Dynamical Systems},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Source Code

Full implementation available in parent directory:
- `models/` - FNO implementation
- `optimizers/` - Adam, SGD, SVRG
- `datasets/` - Lorenz, Logistic generators
- `experiments/` - Training scripts

## Contact

For questions: your.email@institution.edu

---

**Status**: Ready for submission ✓
