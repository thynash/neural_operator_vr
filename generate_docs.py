#!/usr/bin/env python3
"""
API Documentation Generator

Generates comprehensive API documentation from docstrings.
Uses pdoc3 for automatic documentation generation.
"""

import sys
import subprocess
from pathlib import Path


def generate_documentation():
    """Generate API documentation using pdoc3."""
    print("="*80)
    print("API DOCUMENTATION GENERATION")
    print("="*80)
    
    # Check if pdoc3 is installed
    try:
        import pdoc
        print("\n[OK] pdoc3 is installed")
    except ImportError:
        print("\n[WARNING] pdoc3 not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pdoc3"], check=True)
        print("[OK] pdoc3 installed")
    
    # Modules to document
    modules = [
        "datasets",
        "models",
        "optimizers",
        "training",
        "analysis",
        "visualization",
        "experiments",
        "utils",
    ]
    
    print(f"\nGenerating documentation for {len(modules)} modules...")
    
    # Generate HTML documentation
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    try:
        cmd = [
            sys.executable, "-m", "pdoc",
            "--html",
            "--output-dir", str(output_dir),
            "--force",
        ] + modules
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n[SUCCESS] Documentation generated successfully!")
            print(f"Location: {output_dir.absolute()}")
            print("\nGenerated documentation for:")
            for module in modules:
                module_doc = output_dir / f"{module}.html"
                if module_doc.exists():
                    print(f"  - {module}")
            
            # Create index
            create_index(output_dir, modules)
            
            return True
        else:
            print("\n[FAILED] Documentation generation failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        return False


def create_index(output_dir: Path, modules: list):
    """Create an index.html file."""
    index_path = output_dir / "index.html"
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Neural Operator Variance Reduction Framework - API Documentation</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .module-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .module-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .module-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .module-card a {
            text-decoration: none;
            color: #4CAF50;
            font-weight: bold;
            font-size: 18px;
        }
        .module-card p {
            color: #666;
            margin-top: 10px;
            font-size: 14px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #888;
        }
    </style>
</head>
<body>
    <h1>Neural Operator Variance Reduction Framework</h1>
    <p>Research-grade experimental framework for comparing SVRG, SGD, and Adam optimizers 
    when training neural operators (DeepONet and FNO) on dynamical systems.</p>
    
    <h2>API Documentation</h2>
    <div class="module-list">
"""
    
    module_descriptions = {
        "datasets": "Dynamical system data generators (Logistic Map, Lorenz, Burgers)",
        "models": "Neural operator architectures (DeepONet, FNO)",
        "optimizers": "Optimization algorithms with variance tracking (SGD, Adam, SVRG)",
        "training": "Training loop and checkpoint management",
        "analysis": "Metrics computation and spectral analysis",
        "visualization": "Publication-quality plotting utilities",
        "experiments": "Configuration management and experiment runners",
        "utils": "Logging, seeding, device management utilities",
    }
    
    for module in modules:
        desc = module_descriptions.get(module, "")
        html_content += f"""
        <div class="module-card">
            <a href="{module}/index.html">{module}</a>
            <p>{desc}</p>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="footer">
        <p>Generated with pdoc3 | Neural Operator Variance Reduction Framework</p>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n[OK] Index page created: {index_path}")


def main():
    """Main entry point."""
    success = generate_documentation()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
