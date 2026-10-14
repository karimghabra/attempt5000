"""The notebook's code for the tests: its definition cells, run by id into one namespace.

Section 29's fresh-kernel cell runs the cells that only define functions and constants, by the ids it lists in
`DEFINITION_CELLS`, from the saved notebook file; this does the same, with the list read from that cell, so that a
test sees what a fresh kernel would. Nothing of the burst is loaded. The notebook needs what its setup cell imports
(opencv, psutil, IPython); without them the tests that use it are skipped.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

NOTEBOOK = Path(os.environ.get("VASCNET_NOTEBOOK",
                               Path(__file__).resolve().parents[1] / "notebooks" / "development copy 2.ipynb"))


@functools.lru_cache(maxsize=None)
def definitions(notebook: str = str(NOTEBOOK)) -> dict:
    """Run the notebook's definition cells, in the order DEFINITION_CELLS gives, and return their namespace."""
    import matplotlib
    matplotlib.use("Agg")
    with open(notebook, encoding="utf-8") as file:
        sources = {cell.get("id"): "".join(cell["source"]) for cell in json.load(file)["cells"]}
    fresh = next(source for source in sources.values() if "DEFINITION_CELLS = (" in source)
    names = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(fresh))
                 if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "DEFINITION_CELLS")
    space = {"__name__": "notebook"}
    here = os.getcwd()
    os.chdir(Path(notebook).parent)  # as the notebook runs: from inside notebooks/
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for cell_id in names:
                exec(compile(sources[cell_id], f"<cell {cell_id}>", "exec"), space)
    finally:
        os.chdir(here)
    return space


def _feed(digest, value):
    if isinstance(value, np.ndarray):
        digest.update(f"array:{value.dtype}:{value.shape}:".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=str):
            digest.update(f"{key}=".encode())
            _feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(f"{type(value).__name__}:{len(value)}:".encode())
        for item in value:
            _feed(digest, item)
    elif isinstance(value, (bool, np.bool_)):
        digest.update(f"bool:{bool(value)};".encode())
    elif isinstance(value, (int, np.integer)):
        digest.update(f"int:{int(value)};".encode())
    elif isinstance(value, (float, np.floating)):
        digest.update(f"float:{float(value).hex()};".encode())
    elif value is None or isinstance(value, str):
        digest.update(f"{value!r};".encode())
    else:
        raise TypeError(f"no digest for {type(value)}")


def burst_digest(burst: dict, keys) -> dict:
    """One SHA-256 per key of a synthetic burst, over the bytes of every array and the exact value of every number:
    two bursts with the same digests are the same to the bit."""
    out = {}
    for key in keys:
        digest = hashlib.sha256()
        _feed(digest, burst[key])
        out[key] = digest.hexdigest()
    return out
