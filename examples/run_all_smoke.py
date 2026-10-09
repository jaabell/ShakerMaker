# Run every example .py as a smoke test and report PASS / SKIP / FAIL.
#
#   python run_all_smoke.py                 # examples, without 12_validation
#   python run_all_smoke.py --full          # also 12_validation (LOH.1, LOH.3)
#   python run_all_smoke.py --pytest        # also the unit tests in tests/
#   python run_all_smoke.py --mpi 4         # also the MPI contracts in tests/*/mpi_*_contract.py
#   python run_all_smoke.py --timeout 1800  # per-script limit in seconds (default 1800)
#
# Skipped always: legacy_examples/, notebooks, generated "_*" folders, and
# 14_SFSI/ (production launch scripts for a cluster, hours long, no checks).
# --mpi needs `mpirun` on PATH; use one node's worth of ranks or fewer.

import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SKIP_DIRS = {"legacy_examples", "notebooks", "14_SFSI"}
SLOW_DIRS = {"12_validation"}


def collect(full):
    scripts = []
    for root, dirs, files in os.walk(HERE):
        # skip helper/legacy/notebook dirs and generated output dirs ("_*")
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("_")]
        rel = os.path.relpath(root, HERE)
        top = rel.split(os.sep)[0]
        if not full and top in SLOW_DIRS:
            continue
        for f in sorted(files):
            if f.endswith(".py") and f != "run_all_smoke.py":
                scripts.append(os.path.join(root, f))
    return sorted(scripts)


def arg_value(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


def run(cmd, cwd, env, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env,
                           timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return None, out + f"\n[timeout after {timeout} s]"


def main():
    full = "--full" in sys.argv
    timeout = float(arg_value("--timeout", 1800))
    nmpi = int(arg_value("--mpi", 0))
    # Make the repo root importable so examples run from a source checkout
    # without `pip install` (shakermaker resolves to the local package tree).
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}

    def report(name, status, out=""):
        counts[status] += 1
        print(f"{name:60s} {status}", flush=True)
        if status == "FAIL":
            print(out[-2000:], flush=True)

    for s in collect(full):
        rel = os.path.relpath(s, HERE)
        code, out = run([sys.executable, s], os.path.dirname(s), env, timeout)
        if code == 0:
            report(rel, "PASS")
        elif code is not None and "SKIP" in out:
            report(rel, "SKIP")
        else:
            report(rel, "FAIL", out)

    if "--pytest" in sys.argv:
        code, out = run([sys.executable, "-m", "pytest", "-q", "tests"], REPO_ROOT, env, timeout)
        report("pytest tests/", "PASS" if code == 0 else "FAIL", out)
        if code == 0:
            print(out.strip().splitlines()[-1], flush=True)

    if nmpi:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        contracts = sorted(glob.glob(os.path.join(REPO_ROOT, "tests", "*", "mpi_*_contract.py")))
        for c in contracts:
            name = os.path.relpath(c, REPO_ROOT)
            if mpirun is None:
                report(name, "SKIP")
                continue
            module = os.path.splitext(name)[0].replace(os.sep, ".")
            code, out = run([mpirun, "-n", str(nmpi), sys.executable, "-m", module],
                            REPO_ROOT, env, timeout)
            ok = code == 0 and "_PASS" in out
            report(f"{name} (mpirun -n {nmpi})", "PASS" if ok else "FAIL", out)

    print(f"\n{counts['PASS']} passed, {counts['SKIP']} skipped, {counts['FAIL']} failed")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
