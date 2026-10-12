"""Retry transient Windows cache locks without changing numerical fit sources."""
import json
from pathlib import Path
import time


def write_json(path,value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2),encoding="utf-8")
    for attempt in range(200):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 199:
                raise
            time.sleep(.05)


def install_safe_cache_writer():
    """Install only an I/O helper; model, sampler and checkpoint code stay intact.

    Kept outside the numerical fingerprint, like the notebook executor and
    progress reader. Serializes exactly the same JSON as the original writer.
    """
    from . import nll_cv_search
    nll_cv_search._json = write_json
