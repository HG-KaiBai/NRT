# Regularized NLL 环境检查

检查日期：2026-10-11。检查节点：`login1`。

当前环境满足 regularization 新入口的依赖和 CPU 运行需求。
Optuna 4.5.0 已在项目 `.deps` 中可用，代码会自动加载，无需再次安装。
GPU/CUDA 实际训练尚未验证，因为当前登录节点没有可用 GPU。

## 实际使用的环境

运行解释器：`/home/y679zhao/envs/pt/bin/python3.14`，与 `runtime.sh` 默认值一致。

| 组件 | 实测版本 | 加载位置/状态 |
| --- | --- | --- |
| Python | 3.14.2 | `/home/y679zhao/envs/pt` |
| PyTorch | 2.11.0 | 主环境；CUDA build 12.9 |
| NumPy | 2.4.2 | 主环境，Optuna 加载前后版本和路径一致 |
| SciPy | 1.18.1 | 主环境，Sobol API 可用 |
| pandas | 3.0.0 | 主环境，launcher 环境检查所需 |
| Optuna | 4.5.0 | `regularization/.deps/optuna` |
| Alembic | 1.19.1+computecanada | 项目 `.deps` |
| SQLAlchemy | 2.0.54+computecanada | 项目 `.deps` |
| colorlog | 6.10.1+computecanada | 项目 `.deps` |
| tqdm | 4.67.3+computecanada | 使用主环境版本，满足依赖 |
| PyYAML | 6.0.3+computecanada | 使用主环境版本，满足依赖 |

Optuna wheel 的发行版元数据为 `4.5.0+computecanada`，
`optuna.__version__` 为 `4.5.0`。
实测 `packaging.specifiers.SpecifierSet('==4.5.0').contains(...)` 为 True，
满足 [requirements.txt](requirements.txt) 的精确版本要求。

## 加载方式

主环境本身没有安装 Optuna；直接在新 Python 进程中执行 `import optuna`
会找不到模块。新入口通过 `regularization.tuning.main.optuna_module()`
将 `regularization/.deps` 追加到搜索路径后加载 Optuna。

因此运行 tuning 的正常入口可以使用 Optuna。追加路径没有覆盖主环境中已经
加载的 torch、numpy、scipy、pandas，四者的版本和路径均已核对。
项目 `.deps` 内另有 NumPy wheel，但实际使用的仍是主环境 NumPy 2.4.2。

请使用 `python3.14` 的完整路径。服务器默认 `python` 指向另一个 Python；
`/home/y679zhao/envs/pt/bin/python` 文件当前也没有执行权限。
现有 launcher 已使用可执行的 `python3.14`，无需修改。

## 验证结果

- 主环境 `python3.14 -m pip check`：`No broken requirements found.`
- 加载项目 `.deps` 后，Optuna、Alembic、SQLAlchemy、colorlog、tqdm、PyYAML
  的当前生效运行依赖版本检查通过。
- fixed 和 adaptive 分别执行真实 Optuna 搜索：20 个 startup + 1 个 TPE trial。
  验证了 `multivariate=True`、`group=True`、`n_ei_candidates=48`、
  `ask/tell`、trial user attributes、SQLite 保存及重新加载。
  此项使用轻量目标函数检查搜索器与存储，不代表完成 200 个模型候选训练。
- `python3.14 -m unittest regularization.test_workflow -v`：7 项全部通过，
  包括 35 个 case 的数据一致性及真实小规模 CPU tuning → 参数 CSV → simulation → 汇总。
- `runtime.sh`、tuning 和 testing Slurm 脚本通过 `bash -n`；新入口通过 Python 编译检查。
- 默认搜索设置核对为 200 个候选、20 个 startup，fixed/adaptive startup 各生成 20 个配置。

## GPU 检查范围

当前节点实测 `torch.cuda.is_available() == False`、设备数量为 0，
`nvidia-smi` 无法连接 NVIDIA driver。PyTorch 带 CUDA 12.9 支持且提供
`torch.cuda.CUDAGraph` API，但登录节点上的结果不能验证 GPU 节点驱动、
CUDA graph capture 或 fused Adam 的实际执行。

GPU 节点需用同一个解释器验证。实际训练入口已有设备计算 preflight；
本次没有提交 GPU 作业、完整 HPO 或 500 次模拟。
README 中的有限支持零似然问题属于模型/数据范围问题，与环境依赖无关。

## 可复查命令

在 NRT 根目录执行：

```bash
PY=/home/y679zhao/envs/pt/bin/python3.14
"$PY" -m pip check
"$PY" -c 'from regularization.tuning.main import optuna_module; o=optuna_module(); print(o.__version__); print(o.__file__)'
"$PY" -m unittest regularization.test_workflow -v
```

迁移到新环境时，原 `environment/` 快照未包含这次新增的 Optuna。
在恢复原 NRT 环境后，还需按 regularization 的 requirements 安装新增依赖：

```bash
"$PY" -m pip install --no-index --target regularization/.deps -r regularization/requirements.txt
```

该命令适用于本服务器已配置的 Compute Canada wheelhouse；其他服务器需要相应
的包源。本次检查使用已经安装的 `.deps`，没有升级或修改主 Python 环境。
