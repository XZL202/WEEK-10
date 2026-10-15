"""Check the Lab 10 submission without importing Torch or running training.

Usage:
    python check_submission.py
    python check_submission.py --root path/to/repository
    python check_submission.py --submission path/to/submission.zip

The ZIP form checks the exact files that will be uploaded, without extracting it.
Pillow is required to decode PNG/GIF files. MP4 files need OpenCV for decoding.
This is a local preflight check, not the hidden Gradescope evaluator: it does not
load/evaluate the policy or prove that plotted signals are correct.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tempfile
import zipfile


NOTEBOOK = "lab10_drl_highway.ipynb"
ARTIFACTS = "artifacts/"
START = "# === AUTOGRADER_EXTRACT_START: diagnostics ==="
END = "# === AUTOGRADER_EXTRACT_END: diagnostics ==="
SUMMARY_KEYS = {
    "n_episodes", "mean_reward", "collision_rate", "success_rate",
    "mean_survival_steps",
}
META_KEYS = {"algo", "env_id", "total_timesteps", "seed"}


class CheckFailed(ValueError):
    """An actionable validation failure."""


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


class Submission:
    def __init__(self, root, archive=None):
        self.root = Path(root)
        self.archive = zipfile.ZipFile(archive) if archive else None
        if self.archive:
            names = self.archive.namelist()
            require(len(names) == len(set(names)), "submission ZIP contains duplicate paths")
            for name in names:
                parts = PurePosixPath(name).parts
                require("\\" not in name and not name.startswith("/")
                        and ".." not in parts and not re.match(r"^[A-Za-z]:", name),
                        f"unsafe or nonportable ZIP entry: {name!r}")
            self.names = set(names)
        else:
            require(self.root.is_dir(), f"directory does not exist: {self.root}")
            self.names = None

    def close(self):
        if self.archive:
            self.archive.close()

    def exists(self, path):
        if self.archive:
            return path in self.names and not self.archive.getinfo(path).is_dir()
        return (self.root / path).is_file()

    def read(self, path):
        require(self.exists(path), f"missing {path}")
        data = self.archive.read(path) if self.archive else (self.root / path).read_bytes()
        require(bool(data), f"empty file: {path}")
        require(not data.startswith(b"version https://git-lfs.github.com/spec/v1"),
                f"{path} is a Git LFS pointer, not the actual file")
        return data


def check_required_files(source):
    required = [NOTEBOOK, ARTIFACTS + "dqn_highway.zip",
                ARTIFACTS + "results.json", ARTIFACTS + "training_curve.png"]
    missing = [path for path in required if not source.exists(path)]
    if not any(source.exists(ARTIFACTS + "rollout." + ext) for ext in ("gif", "mp4")):
        missing.append(ARTIFACTS + "rollout.gif or artifacts/rollout.mp4")
    hint = ""
    if source.archive and NOTEBOOK not in source.names:
        nested = [name for name in source.names if name.endswith("/" + NOTEBOOK)]
        if nested:
            hint = ("; the notebook is inside a wrapper directory. Put the notebook and "
                    "artifacts/ directly at the ZIP root")
    require(not missing, "missing required files: " + ", ".join(missing) + hint)
    return "notebook, model, results, curve, and rollout present"


def check_diagnostics(source):
    notebook = json.loads(source.read(NOTEBOOK).decode("utf-8-sig"))
    require(isinstance(notebook, dict) and isinstance(notebook.get("cells"), list),
            "notebook has no valid cells array")
    marked = []
    start_count = end_count = 0
    for index, cell in enumerate(notebook["cells"]):
        code = cell.get("source", "")
        code = "".join(code) if isinstance(code, list) else code
        require(isinstance(code, str), f"invalid source in cell {index}")
        starts, ends = code.count(START), code.count(END)
        start_count += starts
        end_count += ends
        if starts or ends:
            require(cell.get("cell_type") == "code", "diagnostics markers must be in a code cell")
            marked.append((index, code))
    require(start_count == end_count == 1 and len(marked) == 1,
            "expected exactly one START and one END diagnostics marker, in the same code cell")
    index, code = marked[0]
    begin, finish = code.index(START) + len(START), code.index(END)
    require(begin < finish, "diagnostics END marker appears before START")
    extracted = code[begin:finish]
    namespace = {}
    exec(compile(extracted, f"{NOTEBOOK}:cell-{index}:diagnostics", "exec"), namespace)
    for name in ("collision_rate", "success_rate", "mean_survival_steps", "summarize_rollout"):
        require(callable(namespace.get(name)), f"extracted diagnostics is missing callable {name}")
    fixtures = [
        ([], (0.0, 0.0, 0.0, 0.0)),
        ([{"reward": 10.0, "length": 30, "crashed": False},
          {"reward": -5.0, "length": 12, "crashed": True},
          {"reward": 8.0, "length": 30, "crashed": False},
          {"reward": -2.0, "length": 5, "crashed": True}], (0.5, 0.5, 19.25, 2.75)),
        ([{"reward": 9.0, "length": 30, "crashed": False},
          {"reward": 11.0, "length": 25, "crashed": False}], (0.0, 1.0, 27.5, 10.0)),
        ([{"reward": -1.0, "length": 3, "crashed": True},
          {"reward": -1.0, "length": 7, "crashed": True}], (1.0, 0.0, 5.0, -1.0)),
    ]
    for episodes, expected in fixtures:
        for name, value in zip(("collision_rate", "success_rate", "mean_survival_steps"), expected):
            actual = namespace[name](episodes)
            require(number(actual) and math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9),
                    f"{name} failed a fixture: expected {value}, got {actual!r}")
        summary = namespace["summarize_rollout"](episodes)
        require(isinstance(summary, dict) and SUMMARY_KEYS <= summary.keys(),
                "summarize_rollout returned an invalid summary")
        wanted = dict(zip(("collision_rate", "success_rate", "mean_survival_steps", "mean_reward"), expected))
        wanted["n_episodes"] = len(episodes)
        for key, value in wanted.items():
            require(number(summary[key]) and math.isclose(summary[key], value, abs_tol=1e-9),
                    f"summarize_rollout fixture failed for {key}")
    return f"unique diagnostics cell {index}; extraction executes independently; four fixtures pass"


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def check_results(source):
    payload = json.loads(source.read(ARTIFACTS + "results.json").decode("utf-8-sig"))
    require(isinstance(payload, dict), "results.json must contain a JSON object")
    missing = (SUMMARY_KEYS | META_KEYS) - payload.keys()
    require(not missing, f"results.json missing keys: {sorted(missing)}")
    require(payload["algo"] == "dqn", "results.json algo must be 'dqn'")
    require(payload["env_id"] == "highway-fast-v0", "results.json env_id must be 'highway-fast-v0'")
    require(integer(payload["total_timesteps"]) and 0 < payload["total_timesteps"] <= 20_000,
            "results.json total_timesteps must be an integer in 1..20000")
    require(integer(payload["seed"]) and 0 <= payload["seed"] < 2**32,
            "results.json seed must be a nonnegative integer below 2**32")
    require(integer(payload["n_episodes"]) and payload["n_episodes"] > 0,
            "results.json n_episodes must be a positive integer")
    for key in SUMMARY_KEYS - {"n_episodes"}:
        require(number(payload[key]), f"results.json {key} must be a finite number")
    for key in ("collision_rate", "success_rate"):
        require(0 <= payload[key] <= 1, f"results.json {key} must be in [0, 1]")
    require(math.isclose(payload["collision_rate"] + payload["success_rate"], 1.0,
                         rel_tol=1e-6, abs_tol=1e-6), "collision_rate + success_rate must equal 1")
    require(0 < payload["mean_survival_steps"] <= 30,
            "mean_survival_steps must be in (0, 30] for the mandatory environment")
    require(0 <= payload["mean_reward"] <= payload["mean_survival_steps"] + 1e-6,
            "mean_reward is inconsistent with the mandatory normalized rewards and survival steps")
    source.results = payload
    return (f"schema and ranges valid; {payload['n_episodes']} evaluation episodes; "
            f"{payload['total_timesteps']} training steps")


def check_model(source):
    with zipfile.ZipFile(io.BytesIO(source.read(ARTIFACTS + "dqn_highway.zip"))) as model:
        names = model.namelist()
        require(len(names) == len(set(names)), "model ZIP contains duplicate entry names")
        for name in ("data", "policy.pth"):
            require(name in names and model.getinfo(name).file_size > 0,
                    f"model ZIP missing nonempty SB3 entry {name!r}")
        require(model.testzip() is None, "model ZIP has a CRC failure")
        data = json.loads(model.read("data").decode("utf-8"))
        require(isinstance(data, dict), "model ZIP SB3 data must be a JSON object")
        require(integer(data.get("num_timesteps")) and 0 < data["num_timesteps"] <= 20_000,
                "saved model must report num_timesteps in 1..20000")
        if hasattr(source, "results"):
            require(data["num_timesteps"] == source.results["total_timesteps"],
                    "results total_timesteps differs from the saved model's num_timesteps")
            if "seed" in data:
                require(data["seed"] == source.results["seed"],
                        "results seed differs from the saved model's seed")
    return "SB3 ZIP structure and CRC valid; training steps agree with results"


def check_curve(source):
    from PIL import Image, ImageStat

    data = source.read(ARTIFACTS + "training_curve.png")
    require(data.startswith(b"\x89PNG\r\n\x1a\n"), "training_curve.png has no PNG signature")
    with Image.open(io.BytesIO(data)) as picture:
        require(picture.format == "PNG", "training_curve.png is not a PNG image")
        require(picture.width >= 200 and picture.height >= 100, "training_curve.png is too small to be a usable plot")
        picture.load()
        require(max(ImageStat.Stat(picture.convert("RGB")).stddev) > 1.0,
                "training_curve.png appears blank")
        size = picture.size
    return f"PNG decodes, is nonblank, and measures {size[0]} x {size[1]} pixels"


class FrameContentCheck:
    """Reject empty rendering and static clips, including MP4 codec noise."""

    def __init__(self):
        self.first = None
        self.nonblank = False
        self.changed = False

    def add(self, picture):
        from PIL import ImageChops, ImageStat

        rgb = picture.convert("RGB")
        self.nonblank |= max(ImageStat.Stat(rgb).stddev) > 1.0
        sample = rgb.resize((96, 96))
        if self.first is None:
            self.first = sample.copy()
        else:
            difference = ImageStat.Stat(ImageChops.difference(sample, self.first))
            # Comparing downsampled frames tolerates small encoding noise while
            # requiring a visible change somewhere in the rollout.
            self.changed |= sum(difference.mean) / 3 > 0.5

    def validate(self, label):
        require(self.nonblank, f"{label} contains only blank/solid-colour frames")
        require(self.changed, f"{label} is static: no two visibly distinct decoded frames")


def check_video(source):
    errors = []
    if source.exists(ARTIFACTS + "rollout.gif"):
        try:
            from PIL import Image

            data = source.read(ARTIFACTS + "rollout.gif")
            require(data[:6] in (b"GIF87a", b"GIF89a"), "rollout.gif has no GIF signature")
            with Image.open(io.BytesIO(data)) as animation:
                require(animation.format == "GIF", "rollout.gif is not a GIF")
                require(animation.width > 0 and animation.height > 0, "rollout.gif has invalid dimensions")
                count = animation.n_frames
                require(count >= 2, "rollout.gif must contain multiple frames")
                duration = 0
                content = FrameContentCheck()
                for frame in range(count):
                    animation.seek(frame)
                    animation.load()
                    content.add(animation)
                    duration += animation.info.get("duration", 0)
                require(duration > 0, "rollout.gif has no positive playback duration")
                content.validate("rollout.gif")
            return (f"GIF decodes all {count} frames with nonblank, changing content; "
                    f"duration {duration / 1000:.1f}s")
        except Exception as exc:
            errors.append(f"GIF: {exc}")
    if source.exists(ARTIFACTS + "rollout.mp4"):
        try:
            data = source.read(ARTIFACTS + "rollout.mp4")
            require(len(data) > 32 and data[4:8] == b"ftyp", "rollout.mp4 has no MP4 ftyp box")
            box_size = struct.unpack(">I", data[:4])[0]
            require(8 <= box_size <= len(data), "rollout.mp4 has an invalid ftyp box")
            try:
                import cv2
                from PIL import Image
            except ImportError as exc:
                raise CheckFailed("OpenCV is required to decode MP4; supply a valid GIF or install opencv-python") from exc
            with tempfile.TemporaryDirectory(prefix="lab10-video-check-") as temporary:
                path = Path(temporary) / "rollout.mp4"
                path.write_bytes(data)
                capture = cv2.VideoCapture(str(path))
                count = 0
                content = FrameContentCheck()
                try:
                    require(capture.isOpened(), "OpenCV cannot open rollout.mp4")
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        require(frame is not None and frame.size > 0, "MP4 contains an empty frame")
                        content.add(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                        count += 1
                    require(count >= 2, "rollout.mp4 must decode at least two frames")
                    content.validate("rollout.mp4")
                finally:
                    capture.release()
            return f"MP4 decodes {count} frames with nonblank, changing content"
        except Exception as exc:
            errors.append(f"MP4: {exc}")
    require(False, "; ".join(errors) or "missing artifacts/rollout.gif or artifacts/rollout.mp4")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="repository directory (default: directory containing this script)")
    parser.add_argument("--submission", type=Path,
                        help="validate this submission ZIP instead of local files; notebook/artifacts must be at ZIP root")
    args = parser.parse_args(argv)
    try:
        source = Submission(args.root, args.submission)
    except Exception as exc:
        print(f"[FAIL] Cannot open submission: {exc}", file=sys.stderr)
        return 1
    checks = [("Required files", check_required_files), ("Diagnostics", check_diagnostics),
              ("Results", check_results), ("Saved model", check_model),
              ("Training curve", check_curve), ("Rollout video", check_video)]
    failures = 0
    try:
        for label, check in checks:
            try:
                detail = check(source)
                print(f"[PASS] {label}: {detail}")
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
    finally:
        source.close()
    if failures:
        print(f"\nSubmission is incomplete or invalid ({failures} failed checks).")
        return 1
    print("\nALL LOCAL SUBMISSION CHECKS PASSED")
    print("Manually confirm the plot shows reward plus a secondary signal. "
          "Gradescope independently evaluates the saved policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
