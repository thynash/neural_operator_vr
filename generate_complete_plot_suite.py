#!/usr/bin/env python3
"""
Complete Plot Suite Generator
Generates all plots needed for publication: Lorenz, Logistic, Operators, Results
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path
import seaborn as sns

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# Create output structure
base = Path("PUBLICATION_PLOTS")
dirs = {
    'lorenz': base / "1_lorenz_system",
    'logistic': base / "2_logistic_map", 
    'operators': base / "3_operator_analysis",
    'results': base / "4_optimizer_results",
    'combined': base / "5_combined_figures"
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

print("="*70)
print("GENERATING COMPLETE PUBLICATION PLOT SUITE")
print("="*70)

# Load experimental results
df = pd.read_csv("comprehensive_analysis/comprehensive_results.csv")
df = df.rename(columns={
    'Final Val Loss': 'Loss',
    'Training Time (min)': 'Time_Min',
    'Total Epochs': 'Epochs',
    'Mean Grad Variance': 'Variance',
    'Spectral Radius': 'Spectral'
})
df['Time'] = df['Time_Min'] * 60
colors_opt = {'Adam': '#FF6B6B', 'SGD': '#4ECDC4', 'SVRG': '#45B7D1'}

print(f"\nLoaded {len(df)} experimental results")


# ============================================================================
# PART 1: LORENZ SYSTEM PLOTS
# ============================================================================
print("\n[1/5] Generating Lorenz system plots...")

def lorenz(x, y, z, s=10, r=28, b=2.667):
    return s*(y-x), r*x-y-x*z, x*y-b*z

dt, steps = 0.01, 10000
xs, ys, zs = np.zeros(steps), np.zeros(steps), np.zeros(steps)
xs[0], ys[0], zs[0] = 0., 1., 1.05

for i in range(steps-1):
    dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
    xs[i+1], ys[i+1], zs[i+1] = xs[i]+dx*dt, ys[i]+dy*dt, zs[i]+dz*dt

# Lorenz 1: 3D Attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs, lw=0.5, color='steelblue', alpha=0.8)
ax.set_xlabel('X', fontsize=12); ax.set_ylabel('Y', fontsize=12); ax.set_zlabel('Z', fontsize=12)
ax.set_title('Lorenz Attractor - 3D Phase Space', fontsize=14, fontweight='bold')
plt.savefig(dirs['lorenz'] / "lorenz_3d_attractor.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['lorenz'] / "lorenz_3d_attractor.pdf", bbox_inches='tight')
plt.close()

# Lorenz 2: Time Series
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
t = np.arange(steps) * dt
for ax, data, label, color in zip(axes, [xs, ys, zs], ['X', 'Y', 'Z'], ['C0', 'C1', 'C2']):
    ax.plot(t[:2000], data[:2000], lw=1.5, color=color)
    ax.set_ylabel(label, fontsize=12)
    ax.grid(True, alpha=0.3)
axes[0].set_title('Lorenz System Time Series', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Time', fontsize=12)
plt.tight_layout()
plt.savefig(dirs['lorenz'] / "lorenz_timeseries.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['lorenz'] / "lorenz_timeseries.pdf", bbox_inches='tight')
plt.close()

# Lorenz 3: Phase Projections
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(xs, ys, lw=0.3, alpha=0.7, color='steelblue')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_title('XY Projection')
axes[1].plot(xs, zs, lw=0.3, alpha=0.7, color='orange')
axes[1].set_xlabel('X'); axes[1].set_ylabel('Z'); axes[1].set_title('XZ Projection')
axes[2].plot(ys, zs, lw=0.3, alpha=0.7, color='green')
axes[2].set_xlabel('Y'); axes[2].set_ylabel('Z'); axes[2].set_title('YZ Projection')
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dirs['lorenz'] / "lorenz_projections.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['lorenz'] / "lorenz_projections.pdf", bbox_inches='tight')
plt.close()

print("  ✓ Generated 3 Lorenz plots")


# ============================================================================
# PART 2: LOGISTIC MAP PLOTS
# ============================================================================
print("\n[2/5] Generating Logistic map plots...")

# Logistic 1: Bifurcation Diagram
fig, ax = plt.subplots(figsize=(12, 8))
r_vals = np.linspace(2.5, 4.0, 2000)
iterations = 1000
last = 100
x = 1e-5 * np.ones(len(r_vals))

for i in range(iterations):
    x = r_vals * x * (1 - x)
    if i >= (iterations - last):
        ax.plot(r_vals, x, ',k', alpha=0.25, markersize=0.5)

ax.set_xlim(2.5, 4)
ax.set_xlabel('r (Growth Rate)', fontsize=12)
ax.set_ylabel('x', fontsize=12)
ax.set_title('Logistic Map Bifurcation Diagram', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.savefig(dirs['logistic'] / "logistic_bifurcation.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['logistic'] / "logistic_bifurcation.pdf", bbox_inches='tight')
plt.close()

# Logistic 2: Time Series for Different r
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
r_values = [2.8, 3.2, 3.5, 3.9]
for ax, r in zip(axes.flat, r_values):
    x_series = [0.1]
    for _ in range(100):
        x_series.append(r * x_series[-1] * (1 - x_series[-1]))
    ax.plot(x_series, lw=1.5, marker='o', markersize=3)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('x', fontsize=11)
    ax.set_title(f'r = {r}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dirs['logistic'] / "logistic_timeseries.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['logistic'] / "logistic_timeseries.pdf", bbox_inches='tight')
plt.close()

# Logistic 3: Cobweb Diagram
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, r in zip(axes, [3.2, 3.9]):
    x_vals = np.linspace(0, 1, 1000)
    ax.plot(x_vals, r * x_vals * (1 - x_vals), 'b-', lw=2, label='f(x)')
    ax.plot(x_vals, x_vals, 'r--', lw=1.5, label='y=x')
    
    x = 0.1
    for i in range(30):
        x_next = r * x * (1 - x)
        ax.plot([x, x], [x, x_next], 'g-', lw=0.8, alpha=0.6)
        ax.plot([x, x_next], [x_next, x_next], 'g-', lw=0.8, alpha=0.6)
        x = x_next
    
    ax.set_xlabel('x_n', fontsize=11)
    ax.set_ylabel('x_{n+1}', fontsize=11)
    ax.set_title(f'Cobweb Plot (r={r})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(dirs['logistic'] / "logistic_cobweb.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['logistic'] / "logistic_cobweb.pdf", bbox_inches='tight')
plt.close()

print("  ✓ Generated 3 Logistic plots")


# ============================================================================
# PART 3: OPERATOR ANALYSIS PLOTS
# ============================================================================
print("\n[3/5] Generating operator analysis plots...")

# Operator 1: Architecture Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# FNO Architecture
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.text(5, 9.5, 'Fourier Neural Operator', ha='center', fontsize=14, fontweight='bold')
layers = [('Input u(x)', 8.5), ('Lift Layer', 7.5), ('Fourier Layer 1', 6.5),
          ('Fourier Layer 2', 5.5), ('Fourier Layer 3', 4.5), ('Fourier Layer 4', 3.5),
          ('Project Layer', 2.5), ('Output û(x)', 1.5)]
colors_fno = ['lightblue', 'lightgreen', 'lightyellow', 'lightyellow', 
              'lightyellow', 'lightyellow', 'lightcoral', 'lightblue']
for (name, y), color in zip(layers, colors_fno):
    rect = plt.Rectangle((1, y-0.3), 8, 0.6, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    if y > 1.5:
        ax.arrow(5, y-0.3, 0, -0.3, head_width=0.3, head_length=0.1, fc='black', ec='black', lw=1.5)

# DeepONet Architecture
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.text(5, 9.5, 'Deep Operator Network', ha='center', fontsize=14, fontweight='bold')
# Branch net
for i, y in enumerate([8, 7, 6, 5]):
    rect = plt.Rectangle((0.5, y-0.3), 3, 0.6, facecolor='lightblue', edgecolor='black', lw=1.5)
    ax.add_patch(rect)
    ax.text(2, y, 'Branch' if i==0 else f'Dense {i}', ha='center', va='center', fontsize=9)
# Trunk net
for i, y in enumerate([8, 7, 6, 5]):
    rect = plt.Rectangle((6.5, y-0.3), 3, 0.6, facecolor='lightgreen', edgecolor='black', lw=1.5)
    ax.add_patch(rect)
    ax.text(8, y, 'Trunk' if i==0 else f'Dense {i}', ha='center', va='center', fontsize=9)
# Dot product
circle = plt.Circle((5, 3.5), 0.5, facecolor='lightyellow', edgecolor='black', lw=2)
ax.add_patch(circle)
ax.text(5, 3.5, '⊙', ha='center', va='center', fontsize=16, fontweight='bold')
ax.arrow(2, 4.7, 2.5, -0.9, head_width=0.2, head_length=0.15, fc='black', ec='black')
ax.arrow(8, 4.7, -2.5, -0.9, head_width=0.2, head_length=0.15, fc='black', ec='black')
# Output
rect = plt.Rectangle((3.5, 2-0.3), 3, 0.6, facecolor='lightcoral', edgecolor='black', lw=2)
ax.add_patch(rect)
ax.text(5, 2, 'Output û(y)', ha='center', va='center', fontsize=10, fontweight='bold')
ax.arrow(5, 3, 0, -0.6, head_width=0.2, head_length=0.15, fc='black', ec='black')

plt.tight_layout()
plt.savefig(dirs['operators'] / "operator_architectures.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['operators'] / "operator_architectures.pdf", bbox_inches='tight')
plt.close()

# Operator 2: Fourier Layer Detail
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.text(5, 9.5, 'Fourier Layer Computation Flow', ha='center', fontsize=14, fontweight='bold')
components = [
    ('Input: v(x) ∈ ℝⁿ', 8.5, 'lightblue', 2.5),
    ('FFT: v(x) → v̂(k)', 7.5, 'lightyellow', 2.5),
    ('Spectral Convolution: R·v̂(k)', 6.5, 'lightcoral', 3),
    ('IFFT: v̂(k) → v\'(x)', 5.5, 'lightyellow', 2.5),
    ('Linear Transform: W·v(x)', 4.5, 'lightgreen', 2.5),
    ('Add & Activate: σ(v\' + Wv)', 3.5, 'lightblue', 3),
    ('Output: v_{out}(x)', 2.5, 'lightcoral', 2)
]
for name, y, color, width in components:
    rect = plt.Rectangle((5-width/2, y-0.3), width, 0.6, 
                         facecolor=color, edgecolor='black', lw=2)
    ax.add_patch(rect)
    ax.text(5, y, name, ha='center', va='center', fontsize=10, fontweight='bold')
    if y > 2.5:
        ax.arrow(5, y-0.3, 0, -0.3, head_width=0.25, head_length=0.1, fc='black', ec='black', lw=2)

plt.savefig(dirs['operators'] / "fourier_layer_detail.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['operators'] / "fourier_layer_detail.pdf", bbox_inches='tight')
plt.close()

print("  ✓ Generated 2 Operator plots")


# ============================================================================
# PART 4: OPTIMIZER RESULTS PLOTS
# ============================================================================
print("\n[4/5] Generating optimizer results plots...")

# Results 1: Main Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
metrics = [
    ('Loss', 'Final Loss', 'lower is better'),
    ('Time', 'Training Time (s)', 'lower is better'),
    ('Epochs', 'Epochs to Convergence', 'lower is better'),
    ('Variance', 'Gradient Variance', 'lower is better'),
    ('Spectral', 'Spectral Radius', 'closer to 1 is stable'),
]

for ax, (col, title, note) in zip(axes.flat[:5], metrics):
    bars = ax.bar(df['Optimizer'], df[col], 
                  color=[colors_opt[opt] for opt in df['Optimizer']],
                  edgecolor='black', linewidth=2, alpha=0.8)
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'{title}\n({note})', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels
    for bar, val in zip(bars, df[col]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2e}' if val < 0.01 else f'{val:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

# Summary table in last subplot
ax = axes.flat[5]
ax.axis('off')
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Optimizer'],
        f"{row['Loss']:.2e}",
        f"{row['Time']:.1f}s",
        f"{row['Variance']:.2e}",
        f"{row['Spectral']:.3f}"
    ])
table = ax.table(cellText=table_data,
                colLabels=['Optimizer', 'Loss', 'Time', 'Variance', 'Spectral'],
                cellLoc='center', loc='center',
                colWidths=[0.15, 0.2, 0.15, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(df)+1):
    for j in range(5):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(['#f1f1f2', '#ffffff'][i % 2])

plt.suptitle('Optimizer Performance Comparison on Lorenz System', 
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(dirs['results'] / "optimizer_comparison_main.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['results'] / "optimizer_comparison_main.pdf", bbox_inches='tight')
plt.close()

# Results 2: Scatter Trade-offs
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
scatter_configs = [
    ('Time', 'Loss', 'Speed vs Accuracy'),
    ('Variance', 'Loss', 'Variance Reduction vs Accuracy'),
    ('Spectral', 'Loss', 'Stability vs Accuracy')
]

for ax, (x_col, y_col, title) in zip(axes, scatter_configs):
    for _, row in df.iterrows():
        ax.scatter(row[x_col], row[y_col], s=300, 
                  color=colors_opt[row['Optimizer']], 
                  label=row['Optimizer'], alpha=0.7,
                  edgecolor='black', linewidth=2)
        ax.text(row[x_col], row[y_col], row['Optimizer'], 
               ha='center', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if ax == axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

plt.tight_layout()
plt.savefig(dirs['results'] / "optimizer_tradeoffs.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['results'] / "optimizer_tradeoffs.pdf", bbox_inches='tight')
plt.close()

print("  ✓ Generated 2 Results plots")


# ============================================================================
# PART 5: COMBINED FIGURES FOR PAPER
# ============================================================================
print("\n[5/5] Generating combined figures...")

# Combined 1: System Overview (Lorenz + Logistic)
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Lorenz 3D
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot(xs[::10], ys[::10], zs[::10], lw=0.5, color='steelblue', alpha=0.8)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.set_title('(a) Lorenz Attractor', fontsize=12, fontweight='bold')

# Lorenz Time Series
ax2 = fig.add_subplot(gs[0, 1:])
ax2.plot(t[:1000], xs[:1000], lw=1, label='X', alpha=0.8)
ax2.plot(t[:1000], ys[:1000], lw=1, label='Y', alpha=0.8)
ax2.plot(t[:1000], zs[:1000], lw=1, label='Z', alpha=0.8)
ax2.set_xlabel('Time'); ax2.set_ylabel('State')
ax2.set_title('(b) Lorenz Time Series', fontsize=12, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

# Logistic Bifurcation
ax3 = fig.add_subplot(gs[1, :2])
r_vals = np.linspace(2.5, 4.0, 1500)
x = 1e-5 * np.ones(len(r_vals))
for i in range(1000):
    x = r_vals * x * (1 - x)
    if i >= 900:
        ax3.plot(r_vals, x, ',k', alpha=0.2, markersize=0.5)
ax3.set_xlabel('r'); ax3.set_ylabel('x')
ax3.set_title('(c) Logistic Map Bifurcation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Logistic Time Series
ax4 = fig.add_subplot(gs[1, 2])
x_series = [0.1]
for _ in range(100):
    x_series.append(3.9 * x_series[-1] * (1 - x_series[-1]))
ax4.plot(x_series, lw=1.5, marker='o', markersize=3, color='darkred')
ax4.set_xlabel('Iteration'); ax4.set_ylabel('x')
ax4.set_title('(d) Chaotic Dynamics (r=3.9)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Dynamical Systems for Neural Operator Learning', 
            fontsize=16, fontweight='bold')
plt.savefig(dirs['combined'] / "fig1_systems_overview.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['combined'] / "fig1_systems_overview.pdf", bbox_inches='tight')
plt.close()

# Combined 2: Main Results Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Loss comparison
ax = axes[0, 0]
bars = ax.bar(df['Optimizer'], df['Loss'], 
             color=[colors_opt[opt] for opt in df['Optimizer']],
             edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
ax.set_title('(a) Accuracy Comparison', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, df['Loss']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
           f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Time comparison
ax = axes[0, 1]
bars = ax.bar(df['Optimizer'], df['Time'],
             color=[colors_opt[opt] for opt in df['Optimizer']],
             edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('Training Time (s)', fontsize=12, fontweight='bold')
ax.set_title('(b) Computational Cost', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, df['Time']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
           f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Variance comparison
ax = axes[1, 0]
bars = ax.bar(df['Optimizer'], df['Variance'],
             color=[colors_opt[opt] for opt in df['Optimizer']],
             edgecolor='black', linewidth=2, alpha=0.8)
ax.set_ylabel('Gradient Variance', fontsize=12, fontweight='bold')
ax.set_title('(c) Variance Reduction', fontsize=13, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, df['Variance']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
           f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Efficiency scatter
ax = axes[1, 1]
for _, row in df.iterrows():
    ax.scatter(row['Time'], row['Loss'], s=400,
              color=colors_opt[row['Optimizer']], alpha=0.7,
              edgecolor='black', linewidth=2, label=row['Optimizer'])
    ax.text(row['Time'], row['Loss'], row['Optimizer'],
           ha='center', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Training Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
ax.set_title('(d) Efficiency Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)

plt.suptitle('Optimizer Performance on Lorenz System with FNO', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(dirs['combined'] / "fig2_main_results.png", dpi=300, bbox_inches='tight')
plt.savefig(dirs['combined'] / "fig2_main_results.pdf", bbox_inches='tight')
plt.close()

print("  ✓ Generated 2 Combined figures")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PLOT GENERATION COMPLETE!")
print("="*70)
print(f"\nGenerated plots in: {base}/")
print("\nBreakdown:")
print(f"  1. Lorenz System:      3 plots  ({dirs['lorenz'].name})")
print(f"  2. Logistic Map:       3 plots  ({dirs['logistic'].name})")
print(f"  3. Operator Analysis:  2 plots  ({dirs['operators'].name})")
print(f"  4. Optimizer Results:  2 plots  ({dirs['results'].name})")
print(f"  5. Combined Figures:   2 plots  ({dirs['combined'].name})")
print(f"\nTotal: 12 publication-ready plots (PNG + PDF)")
print("="*70)
