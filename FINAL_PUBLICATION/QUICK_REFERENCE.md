# Quick Reference Guide

## 🎯 I Want To...

### Write the Paper
→ Open `paper/PAPER_OUTLINE.md`  
→ Use figures from `paper/figures/` (PDF format)  
→ Reference data from `data/comprehensive_results.csv`

### Create Presentation Slides
→ Open `presentation/PPT_OUTLINE.md`  
→ Use figures from `presentation/figures/` (PNG format)  
→ Follow slide-by-slide guide with timing

### Find a Specific Plot

**Lorenz System**:
- 3D attractor → `plots/1_lorenz_system/lorenz_3d_attractor.png`
- Time series → `plots/1_lorenz_system/lorenz_timeseries.png`
- Projections → `plots/1_lorenz_system/lorenz_projections.png`

**Logistic Map**:
- Bifurcation → `plots/2_logistic_map/logistic_bifurcation.png`
- Time series → `plots/2_logistic_map/logistic_timeseries.png`
- Cobweb → `plots/2_logistic_map/logistic_cobweb.png`

**Operators**:
- Architectures → `plots/3_operator_analysis/operator_architectures.png`
- Fourier layer → `plots/3_operator_analysis/fourier_layer_detail.png`

**Results**:
- Main comparison → `plots/4_optimizer_results/optimizer_comparison_main.png`
- Trade-offs → `plots/4_optimizer_results/optimizer_tradeoffs.png`

**Combined (Main Figures)**:
- Fig 1 (Systems) → `plots/5_combined_figures/fig1_systems_overview.png`
- Fig 2 (Results) → `plots/5_combined_figures/fig2_main_results.png`

### Get Experimental Numbers

**All Results**: `data/comprehensive_results.csv`

**Quick Reference**:
- Adam: Loss=3.86e-06, Time=58.9s, Variance=3.08e-05
- SGD: Loss=2.07e-07, Time=428.4s, Variance=1.80e-06
- SVRG: Loss=2.39e-06, Time=1370.4s, Variance=2.44e-07

### Cite This Work

```bibtex
@article{neural_operator_vr_2024,
  title={Variance Reduction Methods for Training Neural Operators on Chaotic Dynamical Systems},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 📋 Checklists

### Paper Submission Checklist
- [ ] Abstract written (250 words)
- [ ] All sections complete (1-6)
- [ ] Figures included and referenced
- [ ] Tables formatted properly
- [ ] References complete (50-60)
- [ ] Proofread 3+ times
- [ ] Colleague feedback received
- [ ] LaTeX compiles without errors
- [ ] Supplementary materials prepared
- [ ] Submitted before deadline

### Presentation Checklist
- [ ] All 20-25 slides created
- [ ] Figures inserted and visible
- [ ] Timing practiced (18 min)
- [ ] Speaker notes reviewed
- [ ] Q&A answers prepared
- [ ] Backup slides ready
- [ ] Clicker/pointer tested
- [ ] Practiced in front of audience

---

## 🔢 Key Numbers to Remember

### Performance
- **Best Accuracy**: SGD at 2.07×10⁻⁷ loss
- **Fastest**: Adam at 58.9 seconds
- **Lowest Variance**: SVRG at 2.44×10⁻⁷
- **Most Stable**: SVRG with spectral radius 0.930

### Comparisons
- Adam is **7.3× faster** than SGD
- SGD is **10× more accurate** than Adam
- SVRG reduces variance by **99.2%** vs Adam
- SVRG is **23× slower** than Adam

---

## 📞 Quick Contacts

**For Questions About**:
- Plots: See `generate_complete_plot_suite.py`
- Paper: See `PAPER_OUTLINE.md`
- Presentation: See `PPT_OUTLINE.md`
- Code: See parent directory source files

---

## ⚡ Emergency Commands

### Regenerate All Plots
```bash
python generate_complete_plot_suite.py
```

### View Results
```bash
cat data/comprehensive_results.csv
```

### Check File Structure
```bash
tree FINAL_PUBLICATION/
```

---

**Last Updated**: Today  
**Status**: ✅ Ready for Publication
