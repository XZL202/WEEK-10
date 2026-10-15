#!/usr/bin/env python3
"""Package the notebook and actual Lab 10 artefacts at the submission ZIP root."""

from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    validator = ROOT / "check_submission.py"
    subprocess.run([sys.executable, str(validator)], cwd=ROOT, check=True)
    required = [
        "lab10_drl_highway.ipynb", "artifacts/dqn_highway.zip",
        "artifacts/results.json", "artifacts/training_curve.png",
    ]
    extra = [
        "artifacts/rollout.gif", "artifacts/rollout.mp4",
        "artifacts/monitor.csv", "artifacts/evaluation_episodes.json", "README.md",
        "check_submission.py", "run_lab10.py", "build_submission.py",
        "verify_saved_policy.py",
    ]
    files = required + [name for name in extra if (ROOT / name).is_file()]
    archive = ROOT / "Lab10_submission.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as submission:
        for name in files:
            submission.write(ROOT / name, arcname=name)
    subprocess.run(
        [sys.executable, str(validator), "--submission", str(archive)],
        cwd=ROOT, check=True,
    )
    print(f"Created {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
