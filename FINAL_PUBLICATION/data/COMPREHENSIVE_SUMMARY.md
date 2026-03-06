# Comprehensive Analysis Summary

Generated: 2026-03-06 10:32:58.561689

## 1. Final Performance

| Optimizer | Val Loss | Iterations | Epochs | Time (min) |
|-----------|----------|------------|--------|------------|
| Adam | 3.857339e-06 | 15100 | 1 | 1.0 |
| SGD | 2.066017e-07 | 145300 | 5 | 7.1 |
| SVRG | 2.394572e-06 | 125000 | 5 | 22.8 |

## 2. Gradient Variance Analysis

| Optimizer | Mean Gradient Variance |
|-----------|------------------------|
| Adam | 3.076953e-05 |
| SGD | 1.802388e-06 |
| SVRG | 2.444414e-07 |

### Variance Reduction:

- **SVRG vs SGD**: 86.44% reduction
- **SVRG vs Adam**: 99.21% reduction

## 3. Spectral Analysis

| Optimizer | Spectral Radius | Stability |
|-----------|-----------------|------------|
| Adam | 0.993443 | ✓ Stable |
| SGD | 0.995899 | ✓ Stable |
| SVRG | 0.929604 | ✓ Stable |

**Note**: Spectral radius < 1.0 indicates stable operator dynamics.

## 4. Abstract Claims Validation

- ✓ SVRG achieves lower loss than Adam
- ✓ SVRG reduces gradient variance vs SGD
- ✗ SVRG NOT more accurate than SGD
- ✓ SVRG operator is spectrally stable

## Generated Files

- `gradient_variance_analysis.pdf/png` - Gradient variance plots
- `spectral_analysis.pdf/png` - Spectral properties plots
- `convergence_efficiency.pdf/png` - Convergence analysis
- `comprehensive_results.csv` - Full results table (CSV)
- `comprehensive_results.txt` - Formatted text table
- `comprehensive_results.tex` - LaTeX table
