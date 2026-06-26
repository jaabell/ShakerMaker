# Run every example .py as a smoke test and report PASS / SKIP / FAIL.
# Skips legacy_examples/ and 12_validation/ unless --full is passed.
# 2026-06-06

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_DIRS = {"legacy_examples", "notebooks"}
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


def main():
    full = "--full" in sys.argv
    scripts = collect(full)
    # Make the repo root importable so examples run from a source checkout
    # without `pip install` (shakermaker resolves to the local package tree).
    repo_root = os.path.dirname(HERE)
    env = dict(os.environ)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    npass = nskip = nfail = 0
    for s in scripts:
        rel = os.path.relpath(s, HERE)
        r = subprocess.run([sys.executable, s], cwd=os.path.dirname(s),
                           capture_output=True, text=True, env=env)
        out = (r.stdout + r.stderr)
        if r.returncode == 0:
            print(f"{rel:55s} PASS"); npass += 1
        elif "SKIP" in out:
            print(f"{rel:55s} SKIP"); nskip += 1
        else:
            print(f"{rel:55s} FAIL"); nfail += 1
            print(out[-1500:])
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
