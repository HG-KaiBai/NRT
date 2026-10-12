"""Reports and saved-artifact checks for independent train/test replications."""
import ast
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .cv_search import METHODS, SOURCE_FILES, observed_scale, refit_epoch_count
from .depth_experiment import _evaluation_data
from .diagnostic_experiment import expanded_settings
from .experiment import ExperimentConfig
from .model import ModelConfig
from .search_io import write_csv


LABELS = dict(adaptive_baseline="Adaptive baseline", brier_cv="Brier CV", nll_cv_control="NLL CV control")


def audit_cv(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output/"config.json").read_text(encoding="utf-8"))
    frozen = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update((Path(__file__).parent/name).read_bytes())
    if digest.hexdigest() != manifest["source_sha256"]:
        raise ValueError("CV numerical source changed")
    if (frozen["fingerprint"] != manifest["fingerprint"] or frozen["selected_sha256"]
            != hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()):
        raise ValueError("frozen parameter hash changed")
    runs, summary, selected, folds, trials = [pd.read_csv(output/name) for name in
        ("runs.csv", "summary.csv", "selected_parameters.csv", "cv_folds.csv", "trials.csv")]
    n_settings, n_trials, n_folds = frozen["settings"], manifest["search"]["trials"], manifest["search"]["folds"]
    seeds = set(manifest["experiment"]["seeds"])
    assert len(runs) == n_settings*len(METHODS)*len(seeds)
    assert len(selected) == n_settings*len(METHODS) and len(trials) == n_settings*n_trials
    assert len(folds) == n_settings*n_trials*n_folds
    assert not runs.duplicated(["setting", "method", "seed"]).any()
    assert not folds.duplicated(["setting", "trial", "fold"]).any()
    assert set(runs.method) == set(METHODS) and runs.learn_splits.all() and folds.learn_splits.all()
    assert runs.n_fit.eq(manifest["n_fit_pool"]).all()
    assert (runs.seed == runs.data_seed).all() and (runs.seed == runs.initialization_seed).all()
    assert not runs.data_seed.eq(manifest["experiment"]["data_seed"]).any()
    assert runs.parameters.eq(runs.trainable_parameters).all()
    assert np.isfinite(runs[["average_grid_sse", "test_nll"]]).all().all()
    fold_plan = json.loads((output/"folds.json").read_text(encoding="utf-8"))
    all_indices = sum(fold_plan["validation_indices"], [])
    assert sorted(all_indices) == list(range(manifest["n_fit_pool"]))
    assert folds.n_fit.add(folds.n_valid).eq(manifest["n_fit_pool"]).all()
    warmup = manifest["optimization"]["warmup_epochs"]
    assert folds.best_epoch.gt(warmup).all() and runs.best_epoch.gt(warmup).all()
    chosen = selected.set_index(["setting", "method"])
    trial_index = trials.set_index(["setting", "trial"])
    for (setting, trial), frame in folds.groupby(["setting", "trial"]):
        assert set(frame.fold) == set(range(n_folds))
        t = trial_index.loc[(setting, trial)]
        for field, actual in (("cv_brier", "validation_brier"), ("cv_nll", "validation_nll")):
            np.testing.assert_allclose(t[field], np.average(frame[actual], weights=frame.n_valid), rtol=1e-12)
        assert t.refit_epochs == refit_epoch_count(frame.best_epoch.tolist(), warmup)
        for row in frame.itertuples():
            history_path = output/"histories"/f"{setting}_trial{trial}_fold{row.fold}.json.gz"
            with gzip.open(history_path, "rt", encoding="utf-8") as handle:
                history = json.load(handle)
            assert len(history) == row.epochs_run
            eligible = [entry for entry in history if entry["split_active"] and "validation_nll" in entry]
            checkpoint = min(eligible, key=lambda entry: entry["validation_nll"])
            assert checkpoint["epoch"] == row.best_epoch
            np.testing.assert_allclose(checkpoint["validation_nll"], row.validation_nll, rtol=1e-12)
            assert all("validation_brier" not in entry for entry in history)
            np.testing.assert_allclose(row.lambda_n, row.regularization_strength/np.sqrt(row.n_fit), rtol=1e-12)
    for (setting, method), frame in runs.groupby(["setting", "method"]):
        assert set(frame.seed) == seeds
        selection = chosen.loc[(setting, method)]
        assert frame.trial.eq(selection.trial).all() and frame.epochs_run.eq(selection.refit_epochs).all()
        for key in manifest["baseline"]:
            np.testing.assert_allclose(frame[key], selection[key], rtol=1e-12)
        assert frame.epochs_run.eq(frame.best_epoch).all()
        np.testing.assert_allclose(frame.lambda_n, frame.regularization_strength/np.sqrt(frame.n_fit), rtol=1e-12)
        if method != "adaptive_baseline":
            criterion, tie = ("cv_brier", "cv_nll") if method == "brier_cv" else ("cv_nll", "cv_brier")
            expected = trials[trials.setting == setting].sort_values([criterion, tie, "trial"]).iloc[0]
            assert selection.trial == expected.trial
    for statistic in ("mean", "std"):
        expected = runs.groupby(["setting", "method"]).average_grid_sse.agg(statistic).sort_index()
        actual = summary.set_index(["setting", "method"])["average_grid_sse_"+statistic].sort_index()
        np.testing.assert_allclose(expected, actual, atol=1e-12)
    reference_curves = {}
    expected_data = {}
    config = ExperimentConfig(**manifest["experiment"])
    setting_map = {name: (setting, actual) for name, _, setting, actual in expanded_settings(config)}
    for row in runs.itertuples():
        with np.load(row.curves_file) as archive:
            grid, truth, survival = archive["grid"], archive["truth"], archive["survival"]
            errors = archive["subject_grid_sse"]
            assert errors.shape == (row.n_test,) and np.isfinite(errors).all()
            np.testing.assert_allclose(errors.mean(), row.average_grid_sse, atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(((truth-survival)**2).sum(-1), errors[:len(truth)], atol=2e-7, rtol=2e-5)
            assert truth.shape == survival.shape == (3, manifest["experiment"]["grid_size"])
            assert np.isfinite(survival).all() and survival.min() >= -1e-6 and survival.max() <= 1+1e-6
            assert np.diff(survival, axis=1).max() <= 1e-6
            key = row.setting, row.seed
            if key not in expected_data:
                setting, actual = setting_map[row.setting]
                data = _evaluation_data(setting, actual, ModelConfig(x_dim=manifest["baseline"]["x_dim"]), row.seed)
                pool = tuple(torch.cat((a, b)) for a, b in zip(data["train"], data["valid"]))
                expected_data[key] = (data["grid"].numpy(), data["truth"][:3].numpy(),
                                      observed_scale(pool), len(data["test"][0]))
            expected_grid, expected_truth, expected_scale, expected_n_test = expected_data[key]
            np.testing.assert_array_equal(grid, expected_grid)
            np.testing.assert_array_equal(truth, expected_truth)
            np.testing.assert_allclose(row.time_scale, expected_scale, rtol=1e-12)
            np.testing.assert_allclose(row.max_obs, expected_grid[-1], rtol=1e-12)
            assert row.n_test == expected_n_test
            if key in reference_curves:
                old_grid, old_truth = reference_curves[key]
                np.testing.assert_array_equal(grid, old_grid)
                np.testing.assert_array_equal(truth, old_truth)
            else:
                reference_curves[key] = (grid.copy(), truth.copy())
        with gzip.open(row.histories_file, "rt", encoding="utf-8") as handle:
            history = json.load(handle)
        assert len(history) == row.epochs_run
        assert all("validation_nll" not in entry for entry in history)
        assert history[-1]["split_active"]
    result = dict(settings=n_settings, cv_candidates=len(trials), cv_fold_fits=len(folds),
        final_train_test_pairs=n_settings*len(seeds), final_fits=len(runs),
        regenerated_train_test_pairs=len(expected_data),
        setting_method_groups=len(summary), data_seeds=sorted(seeds),
        cv_data_seed=manifest["experiment"]["data_seed"], all_rho_trainable=True,
        test_used_for_stopping=False, passed=True)
    (output/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def formatted_sse(output_dir):
    summary = pd.read_csv(Path(output_dir)/"summary.csv")
    summary["mean ± SD"] = summary.apply(lambda r:
        f"{r.average_grid_sse_mean:.6f} ± {r.average_grid_sse_std:.6f}", axis=1)
    return summary.pivot(index="setting", columns="method", values="mean ± SD").reindex(columns=METHODS)


def _markdown(frame):
    lines = ["| "+" | ".join(map(str, frame.columns))+" |",
             "| "+" | ".join("---" for _ in frame.columns)+" |"]
    return "\n".join(lines+["| "+" | ".join(map(str, row))+" |" for row in frame.itertuples(index=False, name=None)])


def report_cv(output_dir):
    output = Path(output_dir)
    audit = audit_cv(output)
    runs = pd.read_csv(output/"runs.csv")
    selected = pd.read_csv(output/"selected_parameters.csv")
    seed_averages = runs.groupby(["method", "seed"]).average_grid_sse.mean().reset_index()
    overall = seed_averages.groupby("method").average_grid_sse.agg(["mean", "std"]).reindex(METHODS).reset_index()
    family_seed = runs.groupby(["family", "method", "seed"]).average_grid_sse.mean().reset_index()
    family = family_seed.groupby(["family", "method"]).average_grid_sse.agg(["mean", "std"]).reset_index()
    write_csv(overall, output/"overall_grid_sse.csv")
    write_csv(family, output/"family_grid_sse.csv")
    table = formatted_sse(output)
    write_csv(table.reset_index(), output/"all_methods_grid_sse.csv")
    paired = runs.pivot(index=["setting", "seed"], columns="method", values="average_grid_sse")
    comparisons = []
    for reference, candidate in ((METHODS[0], METHODS[1]), (METHODS[0], METHODS[2]), (METHODS[1], METHODS[2])):
        differences = (paired[candidate]-paired[reference]).groupby("setting").mean()
        comparisons.append(dict(reference=reference, candidate=candidate,
            mean_difference=float(differences.mean()), better=int((differences < -1e-12).sum()),
            worse=int((differences > 1e-12).sum()), ties=int((differences.abs() <= 1e-12).sum())))
    comparisons = pd.DataFrame(comparisons)
    write_csv(comparisons, output/"paired_comparisons.csv")
    specification = json.loads((output/"config.json").read_text(encoding="utf-8"))
    search = specification["search"]
    strengths = selected[selected.method == "brier_cv"]
    rho_diagnostics = pd.read_csv(output/"summary.csv")
    rho_diagnostics = rho_diagnostics[rho_diagnostics.method == "brier_cv"][[
        "setting", "rho_mean_absolute_shift", "rho_root_mean_absolute_shift"]]
    penalty_rho = strengths[["setting", "regularization_strength"]].merge(rho_diagnostics, on="setting")
    write_csv(penalty_rho, output/"selected_penalty_rho.csv")
    above_old = int(strengths.regularization_strength.gt(1000+1e-8).sum())
    at_upper = int(np.isclose(strengths.regularization_strength, search["strength_max"], rtol=.001).sum())
    brier_comparison = comparisons[comparisons.candidate.eq("brier_cv")].iloc[0]
    nll_comparison = comparisons[comparisons.reference.eq("brier_cv")].iloc[0]
    overall_values = overall.set_index("method")
    improvement = 100*(1-overall_values.loc["brier_cv", "mean"]/overall_values.loc["adaptive_baseline", "mean"])
    nll_improvement = 100*(1-overall_values.loc["nll_cv_control", "mean"]/overall_values.loc["brier_cv", "mean"])
    lines = ["# 4-fold CV：10 组独立 train–test 数据的 grid SSE", "",
        f"Brier CV 的整体平均 SSE 比自适应基线低 {improvement:.2f}%，但只改善 {int(brier_comparison.better)}/35 个 setting，另 {int(brier_comparison.worse)} 个退化；整体收益主要来自两组 multimodal 数据。",
        f"NLL 对照在 {int(nll_comparison.better)} 个 setting 上优于 Brier CV，{int(nll_comparison.worse)} 个更差、{int(nll_comparison.ties)} 个相同，整体平均 SSE 再低 {nll_improvement:.2f}%。这些是测试结果描述，测试 SSE 没有被用于选参。", "",
        "每个 setting 先生成一个独立选参数据集（data seed=123），仅用前 4,000 个样本做固定的四折 CV。",
        f"每个 setting 搜索 {search['trials']} 个六参数组合，每个候选全部运行四折，每折 3,000 训练、1,000 验证。",
        "Sobol 初始化 + multivariate/group TPE；没有多 seed 确认阶段，也没有固定 rho 候选。",
        f"Penalty 系数 a ∈ [0, {search['strength_max']:g}]，实际训练系数 λ=a/√n_fit；所有 setting 使用相同的 NLL + 几何正则公式。",
        "统一优化：主体 LR=0.001，rho LR=0.0001，warmup=10，patience=50，上限 200 epoch。仅 rho 已开始学习后的 NLL checkpoint 可参与评分。",
        "Brier 仅在各折训练结束后用原 neural_only 公式计算，四折均值作为主选参标准；NLL 列是同一 Brier-TPE 候选池的重排序对照。",
        "各折共用选参训练池的 0..max_obs 网格；时间变换尺度仅从该折训练部分估计。",
        "",
        "所有 setting 的参数及训练轮数先冻结。最终数据 seed=0..9，各自重新生成 5,000 个样本，前 4,000 训练、后 1,000 测试。",
        "每次用完整 4,000 个训练样本拟合；epoch 数是所选候选四折最佳 epoch 的中位数向上取整，测试集不参与 early stopping。",
        "同一 setting/seed 下，各方法共用数据、初始化 seed 和网格；不同数据 seed 的 train–test 组合不同。选参数据 seed=123 不在最终十组数据中。",
        "每个 setting 只在 seed123 上选参一次，十组最终数据共用已冻结参数；这里的 SD 不包含重复选参数据集带来的额外波动。",
        "误差严格为 256 点上 squared survival error 的 **sum**，再对测试样本取 mean；CDF SSE 与此相同。先得到每次拟合的误差，再对十组结果取均值和样本标准差。",
        "因此这里的 SD 同时包含数据抽样和初始化变化，不能解读成同一数据集上的初始化 SD，也不能与此前同数据集实验直接作公平增益比较。",
        "CV 减少对单次验证划分的依赖；原 self-imputed Brier 的候选依赖目标未改变，这不构成统计一致性证明。", "",
        "## 所有 setting 等权平均（先在每个数据 seed 内平均，再计算十组 mean/SD）", "", _markdown(overall), "",
        "## 各数据分布", "", _markdown(family), "", "## 每个 setting、每个方法：10 组 mean ± SD", "",
        _markdown(table.reset_index()), "", "## 配对比较", "", _markdown(comparisons), "",
        "## Brier CV 选定的六个参数", "",
        f"{above_old}/{audit['settings']} 个 setting 选中了超过旧上限 1,000 的 a；{at_upper} 个接近新上限。", "",
        _markdown(strengths[["setting", "depth", "shared_width", "shared_layers", "private_width",
            "private_layers", "regularization_strength", "refit_epochs", "cv_brier"]]), "",
        "## 所选 penalty 与最终 rho 偏离 0.5 的程度", "",
        "以下分别报告全部节点的平均绝对偏移和根节点的平均绝对偏移，先平均训练样本/节点，再平均十组数据。深树的全部节点均值会被大量深层节点影响，不能单独代表上层切分是否接近固定。", "",
        _markdown(penalty_rho), "",
        "完整逐折分数见 cv_folds.csv；逐次测试误差见 runs.csv。审计重算所有均值/SD、逐样本 SSE、选参哈希、fold 完整性以及 full-pool 训练轮数。"]
    (output/"grid_sse_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return overall, family, comparisons


def plot_cv(output_dir):
    output = Path(output_dir)
    summary = pd.read_csv(output/"summary.csv")
    selected = pd.read_csv(output/"selected_parameters.csv")
    sse = summary.pivot(index="setting", columns="method", values="average_grid_sse_mean")
    ratios = sse[list(METHODS[1:])].div(sse[METHODS[0]], axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 13), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
    im = axes[0].imshow(np.log2(ratios.to_numpy()), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    axes[0].set_yticks(range(len(ratios)), ratios.index, fontsize=8)
    axes[0].set_xticks(range(2), [LABELS[m] for m in METHODS[1:]])
    axes[0].set_title("Grid SSE / adaptive baseline\nMean over 10 independent train/test pairs")
    figure.colorbar(im, ax=axes[0], label="log2 ratio (blue = lower error)", shrink=.6)
    brier = selected[selected.method == "brier_cv"].set_index("setting").loc[ratios.index]
    axes[1].scatter(brier.regularization_strength, np.arange(len(brier)), s=25)
    axes[1].axvline(1000, color="grey", linestyle="--", label="Previous upper bound")
    axes[1].set_xscale("symlog", linthresh=1)
    search = json.loads((output/"config.json").read_text(encoding="utf-8"))["search"]
    axes[1].set_xlim(0, search["strength_max"]*1.1)
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_yticks([])
    axes[1].set_xlabel("Selected finite penalty strength a\nDashed line: previous upper bound (1,000)")
    axes[1].set_title("Brier CV selected strength\nAll rho weights remain trainable")
    (output/"figures").mkdir(exist_ok=True)
    figure.savefig(output/"figures"/"cv_sse_and_penalty.png", dpi=160)
    return figure
