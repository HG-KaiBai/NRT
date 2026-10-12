# 服务器上的原时间有限支持 KL 调参

打开 `tune_kl_server.ipynb`，先编辑配置 cell，再顺序执行。
当前入口使用 `pycode.unified_nll.raw_time.search` 和 `raw_time.report`。
固定 rho、adaptive KL、匹配 adaptive 架构的固定 rho 三组，均使用原时间有限支持。
历史 `nll_cv_search` / `nll_cv_train` 及完成结果保留，供旧无限支持实验复现。

## 树支持域

每次拟合仅从当前拟合子集的原始 L/R 有限端点计算：

```text
max_observation = max(finite training L/R endpoints)
upper_bound = 1.1 * max_observation
tree support = [0, upper_bound]
R = -2/(depth*L) * sum(log(width/(upper_bound/L)))
lambda_n = a/sqrt(n_fit), L=2**depth
```

没有非线性时间变换，叶内原时间密度均匀。
KL 保留叶平均1/L、深度2/depth；不额外乘 upper_bound。
叶宽下限为 `upper_bound * 0.0008/L`。
CV 每折上界只用该折训练部分；最终拟合上界只用前80%完整拟合池。
验证/测试和潜在真实事件时间不会决定支持域。
验证观测若在训练支持域下不可能，直接报错，不扩大上界或裁剪事件。
最终测试的零似然诚实记录为无穷 NLL，SSE 仍照原网格计算。

## 环境和执行

```bash
cd /your/path/PolyaPFN
python -m pip install numpy pandas scipy matplotlib nbformat nbclient ipykernel
python -m pip install -r pycode/unified_nll/requirements-tuning.txt
python -m unittest pycode.kl_single_case.test_case_study pycode.unified_nll.raw_time.test_raw_time -v
python -m pycode.unified_nll.run_notebook pycode/unified_nll/tuning/tune_kl_server.ipynb
```

默认新目录 `results/kl_cv_server_raw_time_v1`，不复用旧归一化缓存。
配置、源码、设备指纹一致时，重新执行 notebook 恢复缓存；写锁拒绝并发写入。
先由 `verify_generator()` 核对 frozen simulator 与 neural_only.testdata2 的实际逻辑一致。
数据生成仍使用 CPU seed 和原始观测，不改变删失机制、划分或时间数据。

## 配置 cell

| 配置 | 默认值 |
| --- | --- |
| CONFIG.scenarios / censor_types / censor_qs | 全部35 setting |
| CONFIG.n_samples / grid_size | 5000 / 256 |
| CONFIG.data_seed / seeds | 123 / 0..9 |
| BASELINE.x_dim | 6 |
| SEARCH.trials / initial_trials / folds | 64 / 16 / 4 |
| SEARCH.fold_seed / search_seed / search_fit_seed | 20261010 / 871 / 101 |
| depth / shared_layers / private_layers | 3..8 / 0..3 / 1..4 |
| shared_width / private_width | 16,32,64,128 |
| SEARCH.strength_max | 100000，用 log10(1+a) 搜索 |
| TRAINING.lr / OPTIMIZATION.split_lr_ratio | 1e-3 / 0.1 |
| epochs / patience / validation_every / min_delta | 200 / 50 / 5 / 1e-4 |
| warmup_epochs / batch_size / cuda_graphs | 10 / 256 / True |

`train_fraction=0.6` 与 `validation_fraction=0.2` 合成前80%拟合池，
四折只在池内划分，后20%只用于冻结后的最终测试。
每个搜索/架构范围必须仍包含 BASELINE，协议变化必须换输出目录。
实际方法列表由 `raw_time.search.METHODS` 决定。
预算仍为18060 CV fits、1050 final fits；修改 notebook 不会自动启动完整搜索。
checkpoint / TPE / 选参均用未加正则验证 NLL。
所有选择全局冻结后，最终轮数取各折最佳轮数中位数的上取整；不依据测试 SSE 停止。
SSE 保留 neural_only 定义：256点 `[0,训练最大有限端点]`，逐对象误差平方和再平均。
0..9 seed 已用于方法开发，新的确认性评估应预先指定新的数据 seed。
