"""Build an independent NLL-driven fixed/adaptive rho study notebook."""
from pathlib import Path
import nbformat as nb


cells = [nb.v4.new_markdown_cell("""# NLL-driven four-fold CV: fixed vs adaptive rho

35 settings, including **multimodal close and separate**. Each setting has
two independent 64-trial studies: fixed rho=0.5 and learned adaptive rho.
16 Sobol/anchor startup proposals + 48 multivariate/group TPE proposals.
**Mean four-fold held-out observed NLL drives both TPE and final selection.**
This is a new search, not reranking the earlier Brier candidate pool.

The tuning dataset (seed123) contributes its first4,000 subjects: each fold
fits3,000 and validates1,000. Adaptive searches all six parameters, including
finite penalty a in0..100,000. Fixed rho searches the five effective network/
depth parameters; its width penalty is identically zero, so a is set to0.
Both use the same observed NLL plus width penalty, optimizer settings,
checkpoint eligibility after epoch10, and four-fold partitions.

A third method fixes rho at the adaptive-selected architecture and selects
its own refit epoch by four-fold NLL. This separates the independently tuned
systems comparison from the same-architecture rho comparison. Neither
control enters the adaptive search as a candidate.

All selections are frozen before testing. Seeds0..9 each generate a **new**
5,000-subject dataset: first4,000 fit, last1,000 test. Every method receives
the same data pair and initialization seed. Final refits use frozen epoch
counts from CV; test data never selects parameters or stops training.

Grid SSE is exactly the neural_only metric: sum squared survival/CDF errors
on256 equally spaced grid points0..training max_obs, then average test
subjects. Report each setting's ten-dataset mean and sampleSD, with paired
adaptive-minus-fixed differences. Negative differences favor adaptive rho.
"""), nb.v4.new_code_cell("""from pathlib import Path
import sys
import torch
import pandas as pd
from IPython.display import display
ROOT = Path.cwd()
while not (ROOT/'pycode'/'unified_nll').is_dir():
    if ROOT == ROOT.parent:
        raise RuntimeError('Run inside PolyaPFN')
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from pycode.unified_nll.experiment import ExperimentConfig
from pycode.unified_nll.train import TrainingConfig
from pycode.unified_nll.tuning_model import Architecture
from pycode.unified_nll.cv_search import CVSearchConfig
from pycode.unified_nll.nll_cv_train import NLLCVOptimization
from pycode.unified_nll.nll_cv_search import tune_nll_cv,evaluate_nll_cv,METHODS
from pycode.unified_nll.nll_cv_io import install_safe_cache_writer
install_safe_cache_writer()
torch.set_num_threads(4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT = ROOT/'pycode'/'unified_nll'/'results'/'nll_cv_rho_comparison'
CONFIG,SEARCH = ExperimentConfig(),CVSearchConfig()
TRAINING,OPTIMIZATION,BASELINE = TrainingConfig(patience=50),NLLCVOptimization(),Architecture()
KWARGS = dict(config=CONFIG,search=SEARCH,training=TRAINING,optimization=OPTIMIZATION,
              baseline=BASELINE,device=DEVICE)
print('Device:',DEVICE,'Tuning seed:',CONFIG.data_seed,'Final DATA seeds:',CONFIG.seeds)
print('Methods:',METHODS)
print('CV fits:',35*(2*SEARCH.trials+1)*SEARCH.folds,'Final fits:',35*3*len(CONFIG.seeds))
""")]
for upto, family in ((7,"PH"),(14,"lognormal"),(21,"PO"),(28,"multimodal separate"),(35,"multimodal close")):
    cells.append(nb.v4.new_markdown_cell(f"## NLL CV through {family}\n\nEvery candidate completes four folds; interrupted folds resume in place."))
    cells.append(nb.v4.new_code_cell(f"""selected,trials,folds = tune_nll_cv(OUTPUT,stop_after_settings={upto},**KWARGS)
print('Settings completed:',selected.setting.nunique(),'Search candidates:',len(trials),'Fold fits:',len(folds))
display(selected.tail(21)[['setting','method','depth','shared_width','shared_layers',
    'private_width','private_layers','regularization_strength','refit_epochs','cv_nll']])
"""))
cells.extend([nb.v4.new_markdown_cell("""## Frozen configurations → ten new train/test pairs

Parameters are selected once on seed123 and reused across ten independent
datasets. The reported SD includes data and initialization variation,
conditional on this tuning dataset. No test-based epoch selection.
"""), nb.v4.new_code_cell("""runs,summary = evaluate_nll_cv(OUTPUT,**KWARGS)
print('Final fits:',len(runs),'Data seeds:',sorted(runs.data_seed.unique()))
assert runs.n_fit.eq(4000).all() and runs.n_test.eq(1000).all()
display(summary)
"""), nb.v4.new_code_cell("""from pycode.unified_nll.nll_cv_report import report_nll_cv,audit_nll_cv
overall,family,paired = report_nll_cv(OUTPUT)
display(overall)
display(family)
with pd.option_context('display.max_rows',100,'display.max_columns',12):
    display(paired)
print('Report:',OUTPUT/'grid_sse_report.md')
print(audit_nll_cv(OUTPUT))
"""), nb.v4.new_code_cell("""import matplotlib.pyplot as plt
from pycode.unified_nll.nll_cv_report import plot_nll_cv
plot_nll_cv(OUTPUT)
plt.show()
"""), nb.v4.new_markdown_cell("""## Interpretation

The two independently tuned winners can differ in architecture as well as
rho. The matched-architecture control keeps adaptive's network and depth,
with fixed rho and a separately CV-chosen refit duration. It does not match
training duration or trainable parameter count, so it is a practical
same-architecture comparison rather than a one-variable causal experiment.

64 trials per method is a finite search budget, not a global optimum claim.
NLL validation scores and test grid SSE measure different targets. One HPO
dataset per setting does not measure variability from repeating selection.
The ten final pairs match the preceding study's seeds: useful paired
comparisons, but not a new untouched benchmark after choosing NLL from
earlier test evidence. All model selection in this notebook uses CV NLL only.
""")])
path = Path(__file__).with_name("tune_nll_cv_rho.ipynb")
nb.write(nb.v4.new_notebook(cells=cells,metadata={"kernelspec":{
    "display_name":"Python (PolyaPFN)","language":"python","name":"python3"}}),path)
print(path)
