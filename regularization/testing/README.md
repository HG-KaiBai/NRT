# Regularized NLL simulation testing

在 NRT 根目录执行 `python -m regularization.testing.simulation run`，或提交
`sbatch regularization/testing/run_nll.sh`。默认三个 GPU task 各负责一种方法，
每个 task 顺序运行 35 个 case × 500 次重复。

默认直接读取 `../tuning/output/hyper_para`；无需手动复制 CSV。
训练参数及 refit 轮数全部来自 tuning，testing 不再调参或选择停止轮数。
生成器、seed 公式和随机 train/test split 与现有 `testing/simulation.py` 相同。

每次拟合保存一个 JSON。全部请求的重复和方法成功后，自动发布每个 case 的
SSE 表，包括 `average` 和样本 `standard_deviation`。失败记录可重试。

完整命令和输出格式见 [流程说明](../README.md)。
