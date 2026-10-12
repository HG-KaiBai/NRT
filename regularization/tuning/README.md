# Regularized NLL tuning

在 NRT 根目录执行 `python -m regularization.tuning.main`，或提交
`sbatch regularization/tuning/tune_nll.sh`。默认每个 GPU array task 负责一个
case，共 35 个 task；task 内依次完成固定 rho、adaptive KL 和匹配架构对照。
固定 rho 和 adaptive KL 每个 case 各搜索 200 个候选（20 个 startup + 180 个 TPE）；
匹配架构对照只重新做 CV 确定训练轮数。

数据直接调用现有 `workflow.generate`，seed、tau、80/20 随机划分与
`tuning/main.py` 一致。每种方法的选择结果、完整训练设置、固定 refit 轮数和
tau 导出到 `output/hyper_para/<method>.csv`，作为 simulation 的直接输入。

完整算法、参数、恢复和小规模运行命令见 [流程说明](../README.md)。
