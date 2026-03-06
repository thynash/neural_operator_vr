#!/usr/bin/env python3
"""
Clean up repository and organize for publication
"""
import shutil
from pathlib import Path
import os

print("="*70)
print("REPOSITORY CLEANUP AND ORGANIZATION")
print("="*70)

# Files/dirs to remove (temporary, redundant, or unnecessary)
to_remove = [
    # Temporary demo files
    "burgers_comprehensive_demo.py",
    "quick_burgers_demo.py",
    "generate_lorenz_plots.py",
    "generate_lorenz_variety_plots.py",
    "generate_results_variety_plots.py",
    
    # Old organization attempts
    "organized_project",
    "publication_materials",
    "lorenz_plots",
    "results_plots",
    
    # Redundant guides
    "QUICK_START_GUIDE.md",
    
    # Old generation scripts (keep only the final one)
    "generate_publication_package.py",
    "reorganize_project.py",
    "generate_presentation_materials.py",
    "generate_accurate_comparison.py",
    
    # Unnecessary READMEs in subdirectories
    "analysis/README.md",
    "visualization/README.md",
    
    # HTML coverage (can regenerate)
    "htmlcov",
    
    # Visualization output (old)
    "visualization_output",
]

print("\n[1/3] Removing unnecessary files...")
removed_count = 0
for item in to_remove:
    path = Path(item)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
            print(f"  ✓ Removed directory: {item}")
        else:
            path.unlink()
            print(f"  ✓ Removed file: {item}")
        removed_count += 1

print(f"\n  Total removed: {removed_count} items")


# Create clean structure
print("\n[2/3] Creating clean directory structure...")

clean_structure = {
    'FINAL_PUBLICATION': {
        'plots': None,
        'paper': None,
        'presentation': None,
        'data': None
    }
}

base = Path("FINAL_PUBLICATION")
base.mkdir(exist_ok=True)

# Move plots
if Path("PUBLICATION_PLOTS").exists():
    if (base / "plots").exists():
        shutil.rmtree(base / "plots")
    shutil.move("PUBLICATION_PLOTS", base / "plots")
    print("  ✓ Moved plots to FINAL_PUBLICATION/plots/")

# Create paper directory
paper_dir = base / "paper"
paper_dir.mkdir(exist_ok=True)

# Move paper outline
if Path("PAPER_OUTLINE.md").exists():
    shutil.copy2("PAPER_OUTLINE.md", paper_dir / "PAPER_OUTLINE.md")
    print("  ✓ Copied paper outline")

# Copy relevant plots to paper
if (base / "plots").exists():
    (paper_dir / "figures").mkdir(exist_ok=True)
    # Copy combined figures (main paper figures)
    combined_src = base / "plots" / "5_combined_figures"
    if combined_src.exists():
        for f in combined_src.glob("*.pdf"):
            shutil.copy2(f, paper_dir / "figures" / f.name)
        print(f"  ✓ Copied {len(list((paper_dir / 'figures').glob('*.pdf')))} figures to paper/")

# Create presentation directory
pres_dir = base / "presentation"
pres_dir.mkdir(exist_ok=True)

# Move PPT outline
if Path("PPT_OUTLINE.md").exists():
    shutil.copy2("PPT_OUTLINE.md", pres_dir / "PPT_OUTLINE.md")
    print("  ✓ Copied presentation outline")

# Copy all plots to presentation (they'll choose which to use)
if (base / "plots").exists():
    (pres_dir / "figures").mkdir(exist_ok=True)
    for subdir in (base / "plots").iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.png"):  # PNG for presentations
                shutil.copy2(f, pres_dir / "figures" / f.name)
    print(f"  ✓ Copied {len(list((pres_dir / 'figures').glob('*.png')))} figures to presentation/")

# Create data directory
data_dir = base / "data"
data_dir.mkdir(exist_ok=True)

# Copy experimental results
if Path("comprehensive_analysis").exists():
    for f in Path("comprehensive_analysis").glob("*"):
        if f.is_file():
            shutil.copy2(f, data_dir / f.name)
    print(f"  ✓ Copied experimental data")

print("\n[3/3] Creating master README...")

readme_content = """# Neural Operator Variance Reduction - Publication Package

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
"""

with open(base / "README.md", 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("  ✓ Created master README")

print("\n" + "="*70)
print("CLEANUP COMPLETE!")
print("="*70)
print(f"\nFinal structure created in: FINAL_PUBLICATION/")
print("\nContents:")
print(f"  📊 Plots: {len(list((base / 'plots').rglob('*.png')))} PNG + {len(list((base / 'plots').rglob('*.pdf')))} PDF")
print(f"  📄 Paper: outline + {len(list((paper_dir / 'figures').glob('*')))} figures")
print(f"  📊 Presentation: outline + {len(list((pres_dir / 'figures').glob('*')))} figures")
print(f"  📈 Data: {len(list(data_dir.glob('*')))} files")
print("\nNext steps:")
print("  1. Review FINAL_PUBLICATION/README.md")
print("  2. Start writing paper using paper/PAPER_OUTLINE.md")
print("  3. Create slides using presentation/PPT_OUTLINE.md")
print("="*70)
