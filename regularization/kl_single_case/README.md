# 原时间有限支持 KL single case

打开 `model_kl.ipynb` 顺序执行，只拟合一个 observed NLL + 叶宽 KL 模型。
保留 `model`、`history`、`results_df`、`estimated` 和 `grid_sse` 供交互检查。

## 支持域和损失

直接使用原时间 t，树覆盖 `[0, upper_bound]`：

```text
max_observation = max(训练集 L/R 的所有有限端点)
upper_bound = 1.1 * max_observation
R = -2/(depth * L) * sum(log(width / (upper_bound/L)))
loss = mean(observed NLL) + (a/sqrt(n_fit)) * mean(R)
L = 2**depth
```

叶内原时间密度均匀，不使用 t/(t+s) 或非线性 Jacobian。
叶宽下限为 `upper_bound * 0.0008/L`，保留 KL 的叶平均 `1/L` 和 `2/depth`。
上界不读取测试数据、潜在真实事件时间或其他折的验证数据。
精确事件落在支持域外、或删失区间没有支持域交集时，似然为零、NLL 为无穷大；
不裁剪事件、不加 epsilon、不根据测试数据扩大上界。

## 数据、选参和训练

直接导入 `pycode.neural_only.testdata2`。默认配置与 neural_only 单案例相同：
seed123、5000样本、X_DIM=6、PH/right/q0.5。每次重置 CPU seed，
前80%训练、后20%测试，不重排、不重新生成测试集。
SSE 仍用256点原时间网格 `[0,max_observation]`，先逐对象求误差平方和再平均。
支持域上界乘1.1；SSE 网格上界仍是训练观测最大值。

默认 `LOAD_SAVED_SELECTION=True`，读取
`../finite_support_kl/results/finite_support_normalization_comparison/selected_parameters.csv`
中 `raw_time_finite_kl` 已冻结的架构、a 和 epoch。
PH/right/q0.5 的 refit epoch 为35。旧无限支持选参不会被自动载入。
可给 `load_selected` 传入新的原时间 KL 调参结果路径。
改变 X_DIM 时，重新调参，或关闭 LOAD_SAVED_SELECTION 并事先指定架构与轮数。
训练沿用 NLL-only 四折选择后的固定轮数、Adam、warmup 和 split LR。
seed123 已用于 HPO，此 notebook 用于案例展示，不作为新的独立性能评估。

## 保存与运行

默认结果目录为 `results/ph_right_q0.5_seed123_raw_time_finite_v1/`，
与旧案例分开。配置含原时间支持域协议；model.pt 保存权重、架构、优化配置和上界。
另存 metrics/history/per_observation_sse CSV、原始数据、预测/真值和图片。
multimodal 密度图直接使用原时间叶密度。

```bash
python -m pycode.unified_nll.run_notebook pycode/kl_single_case/model_kl.ipynb
python -m unittest pycode.kl_single_case.test_case_study pycode.unified_nll.raw_time.test_raw_time -v
```

新调参入口见 `../unified_nll/tuning/tune_kl_server.ipynb`。
2026-10-11 按要求清理已保存的运行产物，仅保留 CSV 汇总结果。
两次单案例的指标合并在 `results/summary.csv`；模型、预测、原始数据、
逐对象误差、训练历史及图片均已删除，notebook 输出已清空。
默认选参 CSV 保留，重新执行 notebook 会重新训练并保存运行文件。
