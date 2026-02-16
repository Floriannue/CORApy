"""
Run Python and MATLAB jetEngine tracking in sequence (or in parallel if MATLAB run separately).

Usage:
  python run_both_tracking.py              # Run Python only, then print MATLAB instructions
  python run_both_tracking.py --matlab     # Run Python then MATLAB (if matlab on PATH)
  python run_both_tracking.py --compare    # Only run comparison (requires existing logs)

Same benchmark: jetEngine, tFinal=8, R0=zonotope([1;1],0.1*eye(2)), alg=lin-adaptive,
                trackUpstream=True, trackOptimaldeltat=True.
"""
import sys
import os
import subprocess
import argparse

# Project root (where track_upstream_*.py/m and logs are)
ROOT = os.path.dirname(os.path.abspath(__file__))


def run_python():
    """Run Python tracking; saves upstream_python_log.pkl."""
    print("=" * 60)
    print("Running Python tracking (jetEngine, tFinal=8)...")
    print("=" * 60)
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "track_upstream_python.py")],
        cwd=ROOT,
        env=env,
        timeout=300,
    )
    if r.returncode != 0:
        print("[ERROR] Python tracking failed with exit code", r.returncode)
        return False
    if not os.path.isfile(os.path.join(ROOT, "upstream_python_log.pkl")):
        print("[ERROR] upstream_python_log.pkl was not created")
        return False
    print("[OK] Python done. upstream_python_log.pkl written.\n")
    return True


def run_matlab():
    """Run MATLAB tracking; saves upstream_matlab_log.mat. Requires matlab on PATH."""
    m_script = os.path.join(ROOT, "track_upstream_matlab.m")
    if not os.path.isfile(m_script):
        print("[ERROR] track_upstream_matlab.m not found")
        return False
    print("=" * 60)
    print("Running MATLAB tracking (jetEngine, tFinal=8)...")
    print("=" * 60)
    # Run from ROOT so cora_matlab and paths work; use -batch for no GUI
    # Use -r to run script (script must be on path); cd in MATLAB to ROOT first
    run_cmd = f"cd('{ROOT}'); run('track_upstream_matlab');"
    if os.name == "nt":
        run_cmd = run_cmd.replace("\\", "\\\\")
    cmd = ["matlab", "-batch", run_cmd]
    try:
        r = subprocess.run(cmd, cwd=ROOT, timeout=600)
        if r.returncode != 0:
            print("[WARNING] MATLAB exited with code", r.returncode)
        if not os.path.isfile(os.path.join(ROOT, "upstream_matlab_log.mat")):
            print("[ERROR] upstream_matlab_log.mat was not created")
            return False
        print("[OK] MATLAB done. upstream_matlab_log.mat written.\n")
        return True
    except FileNotFoundError:
        print("[INFO] 'matlab' not found on PATH. Run MATLAB manually:")
        print("  In MATLAB: cd to project root, then run('track_upstream_matlab')")
        print("  Or from shell: matlab -batch \"cd('<ROOT>'); run('track_upstream_matlab');\"")
        return False


def main():
    ap = argparse.ArgumentParser(description="Run Python/MATLAB tracking and compare.")
    ap.add_argument("--matlab", action="store_true", help="Also run MATLAB (if matlab on PATH)")
    ap.add_argument("--compare", action="store_true", help="Only run comparison script")
    args = ap.parse_args()

    if args.compare:
        # Run comparison only
        compare_script = os.path.join(ROOT, "compare_intermediate_step2.py")
        if not os.path.isfile(compare_script):
            print("[ERROR] compare_intermediate_step2.py not found")
            sys.exit(1)
        subprocess.run([sys.executable, compare_script], cwd=ROOT)
        return

    # Run Python
    if not run_python():
        sys.exit(1)

    # Optionally run MATLAB
    if args.matlab:
        run_matlab()
    else:
        print("To run MATLAB in parallel (or after):")
        print("  python run_both_tracking.py --matlab")
        print("  Or in MATLAB: run('track_upstream_matlab')")
        print()

    # Run comparison if both logs exist
    py_log = os.path.join(ROOT, "upstream_python_log.pkl")
    ml_log = os.path.join(ROOT, "upstream_matlab_log.mat")
    if os.path.isfile(py_log) and os.path.isfile(ml_log):
        print("Running comparison...")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "compare_intermediate_step2.py")],
            cwd=ROOT,
        )
    else:
        print("Skipping comparison (need both upstream_python_log.pkl and upstream_matlab_log.mat).")


if __name__ == "__main__":
    main()
