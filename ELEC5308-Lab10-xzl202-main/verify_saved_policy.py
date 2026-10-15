#!/usr/bin/env python3
"""Reload the submitted model and reproduce its reported evaluation metrics."""

import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".lab10_deps"))
os.environ.update({"MPLBACKEND": "Agg", "SDL_VIDEODRIVER": "dummy",
                   "SDL_AUDIODRIVER": "dummy", "PYGAME_HIDE_SUPPORT_PROMPT": "1"})

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import DQN
import torch


def main():
    torch.set_num_threads(1)
    notebook = json.loads((ROOT / "lab10_drl_highway.ipynb").read_text(encoding="utf-8"))
    namespace = {"gym": gym, "np": np}
    # Reuse the submitted implementation and mandatory configuration exactly.
    for index in (6, 12, 16):
        exec(compile("".join(notebook["cells"][index]["source"]),
                     f"lab10_drl_highway.ipynb:cell_{index + 1}", "exec"), namespace)
    expected = json.loads((ROOT / "artifacts/results.json").read_text())
    model = DQN.load(ROOT / "artifacts/dqn_highway.zip", device="cpu")
    assert model.num_timesteps == expected["total_timesteps"]
    environment = namespace["make_env"]()
    records = []
    try:
        for seed in range(expected["n_episodes"]):
            record = namespace["rollout"](model, environment, n_episodes=1, seeds=[seed])[0]
            records.append({"seed": seed, **record})
            if (seed + 1) % 20 == 0:
                print(f"Saved-model evaluation: {seed + 1}/{expected['n_episodes']} episodes", flush=True)
    finally:
        environment.close()
    actual = namespace["summarize_rollout"](records)
    for name, value in actual.items():
        assert math.isclose(value, expected[name], rel_tol=1e-9, abs_tol=1e-6), (
            name, value, expected[name]
        )
    (ROOT / "artifacts/evaluation_episodes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print("Saved model reproduces every reported metric:", json.dumps(actual), flush=True)


if __name__ == "__main__":
    main()
