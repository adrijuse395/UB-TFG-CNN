"""
run_plots.py

Executes all central plot scripts from scripts/plots/ using the specified run
directory as the data source and target. Images are generated inside a 'plots/'
subdirectory within the run folder. No symlinks or script files are created
inside the run directory, keeping it perfectly clean!

Usage:
    # Run all plots for a specific run directory:
    python run_plots.py runs/run_20260523_151331

    # Run plots for multiple run directories:
    python run_plots.py runs/run_20260523_151331 runs/run_another
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts" / "plots"


def run_plots_for_dir(run_dir: Path, python: str = sys.executable) -> None:
    """Run all plot_*.py scripts from SCRIPTS_DIR setting RUN_DIR as environment variable."""
    if not run_dir.is_dir():
        print(f"[Error] Run directory not found: {run_dir}")
        sys.exit(1)

    scripts = sorted(SCRIPTS_DIR.glob("plot_*.py"))
    if not scripts:
        print(f"[Error] No plot_*.py scripts found in {SCRIPTS_DIR}")
        sys.exit(1)

    print(f"\n[*] Generating plots for run directory: {run_dir}")
    print(f"[*] Outputs will be saved to: {run_dir / 'plots'}")
    
    # Set the RUN_DIR environment variable for subprocesses
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir.resolve())

    for script in scripts:
        print(f"  Running: {script.name} ...")
        # Run the script directly from SCRIPTS_DIR, passing the custom environment
        result = subprocess.run([python, str(script)], env=env)
        if result.returncode != 0:
            print(f"  [Warning] {script.name} exited with code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run centralized plot scripts for one or more run directories."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="RUN_DIR",
        help="Path(s) to run directories (e.g. runs/run_20260523_151331)",
    )
    # Kept for compatibility but made dummy/deprecated
    parser.add_argument(
        "--run",
        action="store_true",
        help="Deprecated: plots are now run by default without symlinks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated: symlinks are no longer used.",
    )
    args = parser.parse_args()

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        run_plots_for_dir(run_dir)

    print("\n[Done]")


if __name__ == "__main__":
    main()
