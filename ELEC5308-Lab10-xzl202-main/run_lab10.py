#!/usr/bin/env python3
"""Execute Lab 10 in a clean kernel and save actual outputs and artefacts."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

os.environ.pop("SDL_VIDEODRIVER", None)  # highway-env disables drawing for "dummy".
os.environ.update({
    "PYTHONUTF8": "1", "MPLBACKEND": "Agg",
    "SDL_AUDIODRIVER": "dummy",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
})
for variable, folder in {
    "JUPYTER_RUNTIME_DIR": "jupyter_runtime",
    "JUPYTER_CONFIG_DIR": "jupyter_config",
    "IPYTHONDIR": "ipython",
    "MPLCONFIGDIR": "matplotlib_config",
}.items():
    directory = ROOT / ".verification" / folder
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)

from nbclient import NotebookClient
import nbformat


class StreamingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["header"]["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return result


def main():
    notebook_path = ROOT / "lab10_drl_highway.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []

    def announce(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"\nExecuting notebook cell {cell_index + 1}...", flush=True)

    client = StreamingClient(
        notebook, timeout=3600, startup_timeout=60, allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}}, on_cell_start=announce,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error"
                   for cell in code_cells for output in cell.outputs)
    nbformat.validate(notebook)
    print(f"\nCompleted all {len(code_cells)} code cells; actual outputs saved.", flush=True)


if __name__ == "__main__":
    main()
