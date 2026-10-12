# Regularized NLL tuning

在 NRT 根目录执行 `python -m regularization.tuning.main`，或提交
`sbatch regularization/tuning/tune_nll.sh`。默认每个 GPU array task 负责一个
case，共 35 个 task；task 内依次完成固定 rho、adaptive KL 和匹配架构对照。
固定 rho 和 adaptive KL 每个 case 各搜索 200 个候选（20 个 startup + 180 个 TPE）；
匹配架构对照只重新做 CV 确定训练轮数。

数据直接调用现有 `workflow.generate`，seed、tau、80/20 随机划分与
`tuning/main.py` 一致。每种方法的选择结果、完整训练设置、固定 refit 轮数和
tau 导出到 `output/hyper_para/<method>.csv`，作为 simulation 的直接输入。

CV 各折和三个方法共用搜索前由 80% train pool 观测端点确定的支持域，
`U=1.1*max(有限 L/R)`；外层 test 和真实事件时间不参与。此次修复改变源码指纹，
不能继续复用旧的逐折支持域缓存。保留旧输出，在 NRT 根目录用新目录运行完整 array：

```bash
OUTDIR="$PWD/regularization/tuning/output/shared_support" \
  sbatch regularization/tuning/tune_nll.sh
# tuning 完成后，testing 读取新目录，并使用新的模拟输出目录：
HYPER_DIR="$PWD/regularization/tuning/output/shared_support/hyper_para" \
OUTDIR="$PWD/regularization/testing/output/shared_support" \
  sbatch regularization/testing/run_nll.sh
```

完整算法、参数、恢复和小规模运行命令见 [流程说明](../README.md)。
