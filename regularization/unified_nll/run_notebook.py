"""Execute the experiment notebook with the invoking Python environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class SavingNotebookClient(NotebookClient):
    async def async_execute_cell(self, cell, cell_index, execution_count=None, store_history=True):
        result = await super().async_execute_cell(cell, cell_index, execution_count, store_history)
        nbformat.write(self.nb, self.output_path)
        print(f"Saved completed notebook cell {cell_index + 1}/{len(self.nb.cells)}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", type=Path,
                        default=Path(__file__).with_name("tune_nll_cv_rho.ipynb"))
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    module_dir = Path(__file__).resolve().parent
    repository_root = module_dir.parent.parent
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(module_dir / ".jupyter-runtime"))
    os.environ.setdefault("MPLCONFIGDIR", str(module_dir / ".mplconfig"))
    notebook = nbformat.read(notebook_path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = SavingNotebookClient(notebook, km=manager, timeout=7200,
                                  resources={"metadata": {"path": str(repository_root)}})
    client.output_path = notebook_path
    try:
        client.execute()
    finally:
        nbformat.write(notebook, notebook_path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print(f"Executed notebook: {notebook_path}", flush=True)


if __name__ == "__main__":
    main()
