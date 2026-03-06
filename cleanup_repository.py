#!/usr/bin/env python3
"""
Clean up repository - remove unnecessary files
"""
import os
import shutil
from pathlib import Path

print("=" * 70)
print("CLEANING UP REPOSITORY")
print("=" * 70)

# Files and directories to remove
to_remove = [
    # Temporary/demo files
    "burgers_comprehensive_demo.py",
    "quick_burgers_demo.py",
    "burgers_output",
    
    # Old generation scripts (keep only the final one)
    "generate_accurate_comparison.py",
    "generate_all_results.py",
    "generate_comprehensive_analysis.py",
    "generate_final_report.py",
    "generate_presentation_materials.py",
    "compare_results.py",
    "check_and_generate_all.py",
    
    # Old reports (consolidated into organized_project)
    "FINAL_REPORT.md",
    "FINAL_RESULTS_SUMMARY.md",
    "experiment_suite_report.json",
    
    # Temporary output directories
    "visualization_output",
    "publication_materials",  # Moved to organized_project
    
    # HTML coverage (not needed for publication)
    "htmlcov",
    ".coverage",
    
    # Cache directories
    ".pytest_cache",
    "__pycache__",
]

# Additional patterns to search for
patterns_to_remove = [
    "**/__pycache__",
    "**/*.pyc",
    "**/.DS_Store",
]

removed_count = 0

print("\n[1/2] Removing specific files and directories...")
for item in to_remove:
    path = Path(item)
    if path.exists():
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  ✓ Removed directory: {item}")
            else:
                path.unlink()
                print(f"  ✓ Removed file: {item}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {item}: {e}")

print("\n[2/2] Removing cache and temporary files...")
for pattern in patterns_to_remove:
    for path in Path(".").glob(pattern):
        try:
            if path.is_dir():
                shutil.rmtree(path)
                print(f"  ✓ Removed: {path}")
            else:
                path.unlink()
                print(f"  ✓ Removed: {path}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {path}: {e}")

print("\n" + "=" * 70)
print(f"CLEANUP COMPLETE! Removed {removed_count} items")
print("=" * 70)
print("\nRepository is now clean and organized!")
print("\nRemaining structure:")
print("  ✓ Source code (models/, datasets/, optimizers/, etc.)")
print("  ✓ Tests (tests/)")
print("  ✓ Documentation (docs/)")
print("  ✓ Examples (examples/)")
print("  ✓ Organized results (organized_project/)")
print("  ✓ Lorenz plots (lorenz_plots/)")
print("  ✓ Comprehensive analysis (comprehensive_analysis/)")
print("=" * 70)
