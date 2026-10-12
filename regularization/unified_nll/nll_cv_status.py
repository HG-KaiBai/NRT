"""Read-only progress for both independent NLL studies and matched controls."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import time


if os.name == "nt":
    import ctypes
    import msvcrt
    kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    create_file = kernel.CreateFileW
    create_file.argtypes = [ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,
                           ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
    create_file.restype = ctypes.c_void_p


def read_shared(path):
    """Allow atomic cache replacement while a Windows progress reader is open."""
    if os.name != "nt":
        return path.read_text(encoding="utf-8")
    # FILE_SHARE_DELETE is essential: ordinary Python readers can temporarily
    # deny os.replace() on Windows, interrupting an otherwise completed fit.
    handle = create_file(str(path.resolve()),0x80000000,0x1|0x2|0x4,None,3,0x80,None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle,os.O_RDONLY)
    with os.fdopen(descriptor,"r",encoding="utf-8") as stream:
        return stream.read()


def status(output_dir):
    output = Path(output_dir)
    manifest = json.loads(read_shared(output/"config.json"))
    files = list((output/"fold_cache").glob("*.json"))
    latest = max(files,key=lambda p:p.stat().st_mtime) if files else None
    fits = sum(len(json.loads(read_shared(p))["folds"]) for p in files)
    def rows(filename):
        path = output/filename
        if not path.exists():
            return []
        return list(csv.DictReader(io.StringIO(read_shared(path))))
    selected,runs = rows("selected_parameters.csv"),rows("runs.csv")
    search = manifest["search"]
    return dict(cv_fold_fits=fits,expected_cv_fold_fits=manifest["settings"]*(2*search["trials"]+1)*search["folds"],
        settings_selected=len({r["setting"] for r in selected}),expected_settings=manifest["settings"],
        final_fits=len(runs),expected_final_fits=manifest["settings"]*len(manifest["methods"])*len(manifest["experiment"]["seeds"]),
        active=({k:runs[-1][k] for k in ("setting","method","data_seed")} if runs else latest.stem if latest else None),
        seconds_since_last_cache=round(time.time()-latest.stat().st_mtime,1) if latest else None,
        audit_passed=(json.loads(read_shared(output/"audit.json")).get("passed",False) if (output/"audit.json").exists() else False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir",type=Path)
    args = parser.parse_args()
    print(json.dumps(status(args.output_dir),ensure_ascii=False))
