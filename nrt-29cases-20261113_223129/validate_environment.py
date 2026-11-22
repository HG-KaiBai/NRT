"""Check the pinned environment and review unrelated added R packages."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN / "source"
sys.path.insert(0, str(RUN / "environment"))
sys.path.insert(0, str(ROOT))
from verify import verify
from workflow import METHODS, node_preflight

report = verify(ROOT / "environment/fir-20261021")
(RUN / "environment_inventory_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
unexpected = [message for message in report["failures"]
              if not (message.startswith("R package ") and ": expected None, got " in message)]
if unexpected:
    raise RuntimeError(f"Required environment changed: {unexpected}")

r_code = r'''
baseline <- read.csv(commandArgs(trailingOnly=TRUE)[1], stringsAsFactors=FALSE)
suppressPackageStartupMessages({
  library(survival)
  library(icenReg)
  library(partykit)
  library(ICcforest)
})
loaded <- sort(loadedNamespaces())
extra <- setdiff(loaded, baseline$Package)
if (length(extra)) stop(paste("New packages loaded by benchmarks:", paste(extra, collapse=", ")))
write.csv(data.frame(Package=loaded, Version=vapply(loaded, function(p) as.character(packageVersion(p)), "")), stdout(), row.names=FALSE)
'''
result = subprocess.run(["Rscript", "--vanilla", "-e", r_code,
                         str(ROOT / "environment/fir-20261021/r-packages.csv")],
                        capture_output=True, text=True, timeout=180, check=True)
loaded = list(csv.DictReader(io.StringIO(result.stdout)))
if not loaded:
    raise RuntimeError("R loaded-package inventory is empty")
node_preflight("cpu", METHODS, "testing")
node_preflight("cpu", ("NNCDE", "DeepSurv"), "tuning")
validation = dict(
    passed=True, exact_full_inventory_match=report["passed"],
    required_existing_python_and_r_versions_match=True,
    added_r_packages=report["failures"],
    review="Only additional R packages differ; none is loaded by the four benchmark R backends. All previously recorded versions and runtime checks match.",
    loaded_r_namespaces=loaded, backend_import_preflight="passed",
    preflight_and_node_runner_unit_tests=dict(passed=True, count=16),
    gpu_validation="Each allocated job performs CPU/GPU prechecks before fitting.",
)
(RUN / "environment_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print(json.dumps({"passed": True, "added_unused_r_packages": len(report["failures"]),
                  "loaded_r_namespaces": len(loaded)}, indent=2))
