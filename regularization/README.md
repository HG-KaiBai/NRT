# Regularized NLL：tuning 与 simulation testing

本目录把 regular NLL（KL 正则化的 observed-data NLL）整理为两个独立阶段：
先用现有 tuning dataset 做 CV、选参、冻结训练轮数；随后在现有 simulation
dataset 协议下重新拟合和评估。默认保留原代码的三种比较方法。

| 方法名 | rho 与正则化 | 参数选择 |
| --- | --- | --- |
| `fixed_nll_cv` | 固定 rho=0.5，a=0 | 独立搜索架构 |
| `adaptive_nll_cv` | 学习 rho，KL 强度 a≥0 | 独立搜索架构与 a |
| `fixed_at_adaptive_architecture` | 固定 rho=0.5，a=0 | 使用 adaptive 选出的架构，另做 CV 确定自身训练轮数 |

`unified_nll/`、`kl_single_case/` 和已有结果保留。新入口不依赖这些旧入口中
缺失的 `pycode`、`finite_support_kl`、`neural_only` 包。
网络结构复用 `unified_nll/tuning_model.py`；训练循环复制自
`unified_nll/nll_cv_train.py`，并接到本地原时间有限支持模型。

## 与现有数据完全一致

新流程直接导入 NRT 根目录的 `workflow.generate()`、`workflow.horizon()` 和
`workflow.cv_splits()`；实际生成器仍是 `tuning/testdata2.py`，没有另一份可漂移的实现。

| 项目 | 与现有流程一致的设置 |
| --- | --- |
| 数据情形 | 5 个 setting × 每个 7 个 observation/q 组合，共 35 个 case |
| setting | ph、lognormal、po、multimodal_separate、multimodal_closer |
| observation | exact；right/current/interval 各有 q=0.2、0.5 |
| exact case ID | 保留 `_exact_q0.2`，只生成一次，q 不参与生成 |
| 每个 dataset | 默认 n=5000，6 个协变量 |
| tuning seed | `base_seed + case.index * 10000`，base_seed 默认 123 |
| simulation seed | `base_seed + 1000000 + case.index * 10000 + replication` |
| replication | 从 1 开始，默认 1..500，与现有 testing 编号一致 |
| tau | tuning 使用同一个 horizon 校准函数；testing 使用 CSV 保存的 tau，不重新校准 |
| train/test | 使用 `seed+1` 的独立 CPU Generator 随机排列，前 80% / 后 20% |
| tuning CV | 在同一 80% train pool 中用原 `cv_splits`，默认 seed=321、4 折 |
| mu / r | 默认 0 / None；自定义值在 tuning 保存后原样传给 testing |

例如 `ph_right_q0.2` 的 tuning seed=20123，simulation replication 1 的
seed=1020124。`case.index` 保留现有实现中去掉冗余 exact 情形之前的索引，
不能改成新的连续 0..34 编号来生成数据；array task 编号只用来选 case。

在同一 Python/PyTorch 环境、相同 n、base seed、mu、r、tau、case 和
replication 下，两套入口的 X、L、R、T、生成分布参数和 train/test 索引逐项一致。
不同方法也共享相同数据。每个新结果保存 `dataset_sha256`，覆盖这八个 tensor。
测试直接调用现有 tuning 入口捕获生成结果，并用现有 testing 的 `make_job`
核对 simulation seed 和 dataset，覆盖全部 35 个 case，样本量均为 5000。

`tuning/output/cases/<case_id>.json` 使用与原 tuning 相同的字段，包括 seed、tau、
mu、r、样本数量及整体/train/test 的实际右删失率。
这里的数据一致性不表示不同训练方法会得到同样的模型或 SSE。

## 模型、损失与支持域

使用原时间 t 上的有限支持 `[0,U]`：

```text
U = 1.1 * max(80% train pool 的有限 L/R 端点)，在 CV 搜索前固定
K = 2**depth
R(x) = -2/(depth*K) * sum_k log(width_k(x)/(U/K))
lambda_n = a/sqrt(n_fit)
objective = mean(observed-data NLL) + lambda_n * mean(R(x))
```

精确事件用密度，删失数据用区间概率；右删失的 R=inf 直接进入同一个似然。
geometry 和叶宽使用 float64，网络和 log mass 使用 float32。
叶宽下限为 `U * 0.0008/K`；rho=`0.1+0.8*sigmoid(raw)`。
每个 case 的全部 CV fold、候选和方法共用同一个 U，最终拟合也使用这个 U。
这是在外层训练池上固定支持域的协议：内层 validation 的观测端点参与 U 的确定，
但不按候选或验证分数调整 U；外层 20% test 和真实事件时间不参与。
模拟 testing 仍只用每次模拟的 80% train pool 确定 U，不复用 tuning dataset 的数值上界。
直接调用 `fit_raw_time_kl` 时，默认从本次拟合子集确定 U；CV 通过 `upper_bound` 显式传入
预先固定的支持上界。请求文件记录 `support.policy=tuning_train_pool` 和 `upper_bound`，
逐折 `fit_info.max_observation` 仍记录该折实际训练端点的最大值。

似然保留严格计算，不把零概率裁剪为 epsilon。直接拟合时如果 validation observation
在给定支持域下概率为零，仍会报错；正常 tuning 的共同支持域覆盖全部 CV 观测。
测试出现零似然时，结果保存
`test_nll=null, test_nll_infinite=true`，SSE 仍可正常计算；不把无穷 NLL 写成非标准 JSON。

**修复的逐折支持域问题：** 旧规则在每折训练子集上单独确定 U。
在完全复用现有数据、4 折及 fold seed=321 时，
`ph_exact_q0.2` 第 3 折有 2 个零似然验证观测，`lognormal_exact_q0.2` 第 4 折有 1 个，
`lognormal_right_q0.2` 第 2 折有 1 个。共同支持域消除了这三个 case 的 CV 越界，
数据生成、80/20 split 和 CV 索引均保留原样。

原 NRT `tuning/` 的 `NRT_NLL` 同样使用逐折有限支持，但 CV 选择指标是 Brier，
并且 `tuning/tree.py` 会把似然裁剪到至少 `1e-8` 后取 log；因此零似然样本的 NLL
被替换为约 18.42，不会以无穷 NLL 报错。这里继续使用严格 validation NLL 选参。

## 第一部分：tuning

1. 按原流程生成 dataset 和随机 80/20 split，从前 80% 的观测端点固定共同支持域，
   后 20% 不参与选参或支持域确定。
2. fixed 和 adaptive 每个 case 各自运行 200 个候选：20 个 Sobol/anchor startup，随后 180 个 TPE 候选。
   adaptive 包含 a=0 和强正则 anchor，a 用 `log10(1+a)` 搜索。
3. 每个候选在 train pool 中完成相同的 4 折 CV；checkpoint 与选参指标均为
   **未加正则的 validation NLL**，不使用 Brier 或真实生存曲线。
4. 选择加权平均 validation NLL 最小的候选，相同分数按 trial 顺序决定。
   最终训练轮数为各折最佳轮数的中位数向上取整。
5. matched control 复用 adaptive 架构，但用固定 rho 自身的 CV 决定训练轮数。
6. 冻结当前 case 请求的全部配置后，在完整 train pool 上按固定轮数重新拟合，
   只在这时计算 tuning held-out test SSE/NLL。
7. 每种方法的全部请求 case 完成后，自动导出参数 CSV 和 tuning summary。

| 搜索/训练项 | 默认值 |
| --- | --- |
| depth | 3..8 |
| shared layers | 0..3；0 时 shared width 无效，归一为 baseline width |
| private layers | 1..4，含 feature projection，不含输出 head |
| shared/private width | 16、32、64、128 |
| a（adaptive） | 0..100000 |
| baseline | depth=3，shared layers=0，shared/private width=32，private layers=3，a=500 |
| HPO search / fit seed | 871 / 101 |
| optimizer | Adam；mass/shared LR=1e-3，split LR=1e-4 |
| warmup | 前 10 epoch 固定 rho=0.5；随后重置 split Adam 状态并解冻 |
| max epochs / patience | 200 / 50 epoch |
| validation_every / min_delta | 5 / 1e-4 |
| batch size / gradient clip | 256 / 1.0 |
| CUDA graphs | 默认开启，可用 `--no-cuda-graphs` 关闭 |

完整默认 tuning 共 `35 * (2*200*4 + 4) = 56140` 次 CV 拟合和 105 次最终拟合。
可用 `--search-space` 的 JSON 改 `search` 和 `baseline` 字段；baseline 必须在搜索范围内。
修改模型/搜索/训练/生成设置后使用新输出目录，旧缓存会被指纹检查拒绝。
本次支持域修复也改变源码及请求指纹；已有 tuning 输出需保留在原目录，修复后另选目录重跑。

## 第二部分：simulation testing

1. 默认从 `regularization/tuning/output/hyper_para/<method>.csv` 读取配置；
   `--hyper-dir` 可指定其他目录，无需复制 CSV。
2. 恢复参数、training_config、optimization、refit_epochs 和数据设置。
   检查方法之间的 tau/mu/r/n/base seed 一致，并检查数值源码指纹。
3. 每个 case、replication 按现有 testing 协议生成新数据，随机 80/20 split。
4. 使用冻结的参数和轮数从头训练；没有新 CV、选参或 test early stopping。
   模型初始化 seed=`data_seed+1`，不同方法使用同一初始化 seed。
5. 每个拟合保存一个 resumable JSON；成功且请求相同的结果复用，失败记录重试。
6. worker 结束时检查共享 JSON，完整 case 自动生成 SSE 表。默认全部三方法完成
   后发布；单方法运行可通过 `--summary-methods` 指定等待集合。

默认模拟预算为 `35 * 3 * 500 = 52500` 次最终拟合。
SSE 与现有评估相同：256 点均匀网格 `[0,训练集最大有限端点]`，
`mean_subjects(sum_grid((S_hat-S_true)**2))`，不除以网格点数、不乘积分步长。
summary 的 SD 使用样本标准差（n−1）。数据真实分布仅用于 SSE 评估。

## 运行

2026-10-11 实测依赖与 CPU 流程通过，Optuna 4.5.0 从项目 `.deps` 加载。
完整版本、加载路径和 GPU 验证范围见 [环境检查记录](ENVIRONMENT_CHECK.md)。

以下命令都在 **NRT 根目录**执行。默认使用项目已有 Python 环境。
补充依赖装入本目录 `.deps`，新入口会自动追加该路径，不覆盖已有 numpy：

```bash
cd /home/y679zhao/projects/def-anthree/y679zhao/NRT
PY=/home/y679zhao/envs/pt/bin/python3.14
"$PY" -m pip install --no-index --target regularization/.deps -r regularization/requirements.txt
"$PY" -m unittest regularization.test_workflow -v
```

`--no-index` 使用本服务器配置的 Compute Canada wheelhouse；其他服务器可去掉
该选项，或先在已激活环境中安装 requirements。

只检查任务配置，不运行训练：

```bash
"$PY" -m regularization.tuning.main --dry-run
"$PY" -m regularization.testing.simulation run --dry-run
```

提交完整 tuning；等其全部成功后，再提交 testing：

下面是完整任务的提交方式；需先处理上述三个有限支持 case，再运行完整流程。
其他 case 可按 `--cases` 子集独立调参与模拟。

```bash
sbatch regularization/tuning/tune_nll.sh
# 等 tuning array 全部成功后：
sbatch regularization/testing/run_nll.sh
```

也可以显式绑定成功依赖（把 job ID 换成实际 tuning array ID）：

```bash
sbatch --dependency=afterok:TUNING_JOB_ID regularization/testing/run_nll.sh
```

tuning 默认 array=0..34，一个 GPU 负责一个 case 内的三个方法。
testing 默认 array=0..2，一个 GPU 负责一个方法的全部 case 和 replication。
更改 `CASES` 时，相应缩小 tuning 的 array 范围；其 task-id 是所选 case 列表中的索引。
launcher 复用原 `runtime.sh` 的环境检查、设备 preflight 和启动失败 requeue。
`PY`、`OUTDIR`、`HYPER_DIR`、`CASES`、`N_SAMPLES`、`BASE_SEED`、`DEVICE` 可覆盖默认值。
自定义生成参数用 tuning 的 `--mu` / `--r` 传入，testing 自动读取 CSV 保存值。

单 case 示例（仍用完整默认搜索）：

```bash
"$PY" -m regularization.tuning.main --cases ph_right_q0.2 --device cuda:0
"$PY" -m regularization.testing.simulation run --cases ph_right_q0.2 --device cuda:0
```

只运行 adaptive：

```bash
METHODS=adaptive_nll_cv CASES=ph_right_q0.2 \
  sbatch --array=0 regularization/tuning/tune_nll.sh
SUMMARY_METHODS=adaptive_nll_cv CASES=ph_right_q0.2 \
  sbatch --array=1 regularization/testing/run_nll.sh
```

CPU 小规模验证，使用新输出目录，后续 testing 必须保持 n 和 base seed 一致：

```bash
"$PY" -m regularization.tuning.main --cases po_interval_q0.2 \
  --output-dir /tmp/nll_tuning_smoke --n-samples 128 --device cpu --threads 1 \
  --trials 2 --initial-trials 2 --folds 2 --epochs 3 --warmup-epochs 1 \
  --validation-every 1 --no-cuda-graphs
"$PY" -m regularization.testing.simulation run --cases po_interval_q0.2 \
  --hyper-dir /tmp/nll_tuning_smoke/hyper_para --output-dir /tmp/nll_testing_smoke \
  --n-samples 128 --replications 2 --device cpu --threads 1
```

手动重建 summary，无需重新拟合：

```bash
"$PY" -m regularization.tuning.main --summarize
"$PY" -m regularization.testing.simulation summarize
# 不完整的模拟结果仅在显式请求下发布，缺失/失败单元格留空：
"$PY" -m regularization.testing.simulation summarize --allow-incomplete
```

子集或自定义输出目录要传入与原运行一致的 `--cases`、`--methods`、
`--replications`、`--output-dir`。`--replication` 或
`--start-replication/--end-replication` 可分批执行，但汇总目标仍为 1..replications。

## 输出

```text
regularization/
├── model.py, train.py, workflow.py
├── tuning/
│   ├── main.py, tune_nll.sh
│   └── output/
│       ├── cases/<case_id>.json
│       ├── studies/<case_id>/request.json, folds.json
│       ├── studies/<case_id>/<method>/study.sqlite3, cv_results.csv, fold_cache/
│       ├── <method>/<case_id>/selection.json, result.json, history.json
│       ├── hyper_para/<method>.csv
│       └── <method>_summary.csv
└── testing/
    ├── simulation.py, run_nll.sh
    └── output/
        ├── <method>/<case_id>/replication_0001.json ... replication_0500.json
        ├── tables/<case_id>.csv
        └── censoring_summary.csv
```

参数 CSV 仍包含原流程的 `model_name,data_types,parameters,test_average_grid_sse,tau`，
并额外保存训练设置、refit 轮数、数据配置及源码/tuning 指纹。
它只由新 regularization testing 读取，不覆盖原 NRT 的参数 CSV。
case SSE 表为 500 行 replication 加 `average`、`standard_deviation` 两行。
`censoring_summary.csv` 按 dataset 去重，同一 case/replication 在三个方法中只计一次。

调参与每次模拟使用写锁，结果 JSON、CSV 均原子替换。中断后用相同命令继续：
tuning 恢复 SQLite 与逐折缓存，testing 复用成功 JSON。修改协议应换新输出目录；
自动汇总在缺失或失败时保留原有完整表。

## 验证范围

`regularization/test_workflow.py` 检查：35 个 case 的原/新 tuning 数据逐项一致、
35 个 case 在 simulation replication 1 和 7 的数据逐项一致、自定义 mu/r 与 RNG、
四类观测似然、有限支持和 KL 缩放、固定 rho、固定轮数、无穷测试 NLL 的 JSON 表达，
全部 35 case / 140 默认 CV fold 的共同支持域、训练池孤立极端值下三个方法的真实选参，
外层 test / 真实事件时间不参与支持域确定，
以及真实的小规模 tuning → CSV → simulation → 自动汇总、恢复和缺失结果处理。
该验证不等同于完整的 35 case × 500 次 GPU 实验。

本次本地验证：上述 9 项测试全部通过，包含真实 CPU 小规模训练与全部 35 case
的默认样本量数据一致性检查；两个 Slurm 脚本通过 `bash -n`。
另外对原先报错的三个 case 使用默认 n=5000、4 折，在小架构、2 trial、3 epoch 下
完成了 60 次 CV 拟合、9 次 tuning refit 和 9 次 simulation 拟合，全部成功，参数 CSV
及三张 case 汇总表正常生成。
尚未提交完整 tuning 或 500 次 GPU simulation 作业。
