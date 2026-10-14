"""Execute a notebook from top to bottom in one kernel, like

    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 NOTEBOOK

but with a log that is written as the run goes: every code cell's number, id, first line, printed text and
wall-clock time. nbconvert writes nothing until the last cell is done, so a failure hours in loses every printed
result; here they are on disk the moment a cell ends. The executed notebook is written in place at the end, and
also when a cell fails (with what has run so far), before the error is raised again.

    python scripts/run_notebook.py "notebooks/development copy 2.ipynb" reports/logs/reproduction.log

The kernel's working directory is the notebook's folder, as when it is run from inside notebooks/.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def cell_text(cell):
    """Return what a code cell printed: its streams, its text results and the head of an error."""
    parts = []
    for output in cell.get("outputs", []):
        kind = output.get("output_type")
        if kind == "stream":
            parts.append(output.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            text = output.get("data", {}).get("text/plain")
            if text and not text.startswith("<"):  # not the repr of a picture
                parts.append(text + "\n")
        elif kind == "error":
            parts.append(f"{output.get('ename')}: {output.get('evalue')}\n")
    return "".join(parts)


def main(notebook_path, log_path):
    notebook_path, log_path = Path(notebook_path).resolve(), Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = sum(cell.cell_type == "code" for cell in notebook.cells)
    started, cell_started = time.time(), {}

    def note(text):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(text)

    def on_cell_execute(cell, cell_index):
        cell_started[cell_index] = time.time()
        first = cell.source.strip().splitlines()[0] if cell.source.strip() else ""
        note(f"\n===== cell {cell_index} id={cell.get('id')} | started at {time.strftime('%H:%M:%S')} | {first[:100]}\n")

    def on_cell_executed(cell, cell_index, execute_reply):
        took = time.time() - cell_started.get(cell_index, time.time())
        note(cell_text(cell))
        note(f"----- cell {cell_index} took {took:.1f} s | {(time.time() - started) / 60:.1f} min since the start\n")

    if sys.platform == "win32":  # keep the machine from sleeping while this process lives; undone when it exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

    note(f"Run of {notebook_path.name} started {time.strftime('%Y-%m-%d %H:%M:%S')}: {code_cells} code cells\n")
    client = NotebookClient(notebook, timeout=None, kernel_name=notebook.metadata.get("kernelspec", {}).get("name", "python3"),
                            resources={"metadata": {"path": str(notebook_path.parent)}},
                            on_cell_execute=on_cell_execute, on_cell_executed=on_cell_executed)
    try:
        client.execute()
    except CellExecutionError as error:
        note(f"\nFAILED after {(time.time() - started) / 60:.1f} min\n{error}\n")
        raise
    finally:
        nbformat.write(notebook, notebook_path)
    note(f"\nDONE: {code_cells} code cells in {(time.time() - started) / 60:.1f} min "
         f"({time.time() - started:.0f} s), finished {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
