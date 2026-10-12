"""Ten-seed grid SSE reports and static scientific figures for diagnostics."""
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostic_experiment import METHODS
from .search_io import write_csv


LABELS = dict(common_baseline="Common baseline", brier_original="Brier / original",
    nll_original="NLL / original", brier_slowrho="Brier / slow rho",
    brier_warmup="Brier / +warmup", brier_stable="Brier / +patience",
    brier_fixed="Brier / fixed rho", nll_stable="NLL / stable")
CHINESE = dict(common_baseline="统一参数基线", brier_original="Brier选参·原训练",
    nll_original="NLL选参·原训练", brier_slowrho="Brier·降低rho学习率",
    brier_warmup="Brier·加预热", brier_stable="Brier·加长patience",
    brier_fixed="Brier·固定rho", nll_stable="NLL选参·组合优化")
COMPARISONS = (("brier_original", "brier_slowrho"), ("brier_slowrho", "brier_warmup"),
    ("brier_warmup", "brier_stable"), ("brier_stable", "brier_fixed"),
    ("brier_original", "nll_original"), ("brier_stable", "nll_stable"),
    ("common_baseline", "brier_stable"), ("common_baseline", "nll_stable"))


def formatted_sse(output_dir):
    summary = pd.read_csv(Path(output_dir)/"summary.csv")
    summary["SSE mean +/- SD"] = summary.apply(
        lambda r: f"{r.average_grid_sse_mean:.6f} ± {r.average_grid_sse_std:.6f}", axis=1)
    return summary.pivot(index="setting", columns="method", values="SSE mean +/- SD").reindex(columns=METHODS)


def _markdown(frame):
    columns = list(frame.columns)
    lines = ["| "+" | ".join(map(str, columns))+" |", "| "+" | ".join("---" for _ in columns)+" |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| "+" | ".join(str(x) for x in row)+" |")
    return "\n".join(lines)


def report_diagnostics(output_dir):
    output = Path(output_dir)
    runs = pd.read_csv(output/"runs.csv")
    summary = pd.read_csv(output/"summary.csv")
    expected = json.loads((output/"config.json").read_text(encoding="utf-8"))["experiment"]
    seeds = set(expected["seeds"])
    count = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))["settings"]
    if len(runs) != count*len(METHODS)*len(seeds) or runs.duplicated(["setting", "method", "seed"]).any():
        raise ValueError("incomplete or duplicate evaluations")
    for _, group in runs.groupby(["setting", "method"]):
        if set(group.seed) != seeds:
            raise ValueError("unexpected final seeds")
    if not np.isfinite(runs[["average_grid_sse", "validation_nll", "validation_brier", "test_nll", "test_brier"]]).all().all():
        raise ValueError("nonfinite final scores")
    for metric in ("mean", "std"):
        recomputed = runs.groupby(["setting", "method"]).average_grid_sse.agg(metric)
        actual = summary.set_index(["setting", "method"])[f"average_grid_sse_{metric}"]
        np.testing.assert_allclose(actual.sort_index(), recomputed.sort_index(), atol=1e-12)
    by_seed = runs.groupby(["method", "seed"]).average_grid_sse.mean().reset_index()
    overall = by_seed.groupby("method").average_grid_sse.agg(["mean", "std"]).reindex(METHODS).reset_index()
    overall["settings"] = count
    overall["n_seeds"] = len(seeds)
    write_csv(overall, output/"overall_grid_sse.csv")
    family_seed = runs.groupby(["family", "method", "seed"]).average_grid_sse.mean().reset_index()
    family = family_seed.groupby(["family", "method"]).average_grid_sse.agg(["mean", "std"]).reset_index()
    write_csv(family, output/"family_grid_sse.csv")
    pivot = runs.pivot(index=["setting", "seed"], columns="method", values="average_grid_sse")
    comparisons = []
    for reference, candidate in COMPARISONS:
        delta = pivot[candidate]-pivot[reference]
        setting_delta = delta.groupby("setting").mean()
        seed_delta = delta.groupby("seed").mean()
        comparisons.append(dict(reference=reference, candidate=candidate,
            mean_paired_difference=seed_delta.mean(), sd_paired_seed_averages=seed_delta.std(),
            settings_improved=int((setting_delta < -1e-12).sum()),
            settings_worsened=int((setting_delta > 1e-12).sum()),
            settings_tied=int((setting_delta.abs() <= 1e-12).sum())))
    comparisons = pd.DataFrame(comparisons)
    write_csv(comparisons, output/"paired_comparisons.csv")
    selection_comparisons = []
    for reference, candidate in (("brier_original", "nll_original"), ("brier_stable", "nll_stable"),
                                 ("common_baseline", "brier_original")):
        left = summary[summary.method == reference].set_index("setting")
        right = summary[summary.method == candidate].set_index("setting")
        for setting in left.index:
            selection_comparisons.append(dict(setting=setting, reference=reference, candidate=candidate,
                sse_reference=left.loc[setting, "average_grid_sse_mean"],
                sse_candidate=right.loc[setting, "average_grid_sse_mean"],
                brier_reference=left.loc[setting, "validation_brier_mean"],
                brier_candidate=right.loc[setting, "validation_brier_mean"],
                nll_reference=left.loc[setting, "validation_nll_mean"],
                nll_candidate=right.loc[setting, "validation_nll_mean"]))
    selection_comparisons = pd.DataFrame(selection_comparisons)
    selection_comparisons["lower_brier_higher_sse"] = (
        (selection_comparisons.brier_candidate < selection_comparisons.brier_reference)
        &(selection_comparisons.sse_candidate > selection_comparisons.sse_reference))
    write_csv(selection_comparisons, output/"score_alignment.csv")
    write_csv(formatted_sse(output).reset_index(), output/"all_methods_grid_sse.csv")
    counts = dict(settings=count, methods=len(METHODS), final_evaluations=len(runs),
        reused_evaluations=int(runs.imported_from.notna().sum()),
        new_final_fits=int(runs.imported_from.isna().sum()),
        extra_confirmation_fits=len(pd.read_csv(output/"extra_confirmation_runs.csv"))
            if (output/"extra_confirmation_runs.csv").exists() else 0)
    (output/"audit.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    lines = ["# 统一 NLL 的优化与选参对照：每个 setting 十个 seed 的 grid SSE", "",
        "本次包括 PH、lognormal、PO、multimodal separate 和 multimodal close，共 35 个 setting。",
        "close 对应原始模拟器的 `closer`（基准尺度 3、6）；separate 为 1、6；Weibull shape 都是 6、8，权重均为 0.5。", "",
        "主误差为每位测试样本在 0..max_obs 的 256 点上求 CDF 平方误差之和，再平均测试样本。",
        "每格报告十个独立拟合 seed（0..9）的误差 mean ± sample SD；不先平均预测、不除以网格点数、不积分。",
        "同一个 setting 的数据 seed=123、n=5000、60/20/20 划分和网格在全部方法之间一致。", "",
        "## 方法与控制变量", "",
        "全部方法训练同一个 observed NLL + a/sqrt(n_train) × partition penalty。Brier 完全保留 neural_only 的 self-imputed 公式，只在 NLL checkpoint 确定后评分。",
        "全部训练最多 200 epochs，mass/shared 学习率 0.001。Brier 仍是主选参方法；NLL 选参是明确标注的对照。", "",
        _markdown(pd.DataFrame([
            [CHINESE["common_baseline"], "(3,32,0,32,3,500)", "0.001", "0", "20"],
            [CHINESE["brier_original"], "每 setting Brier选参", "0.001", "0", "20"],
            [CHINESE["nll_original"], "每 setting NLL选参", "0.001", "0", "20"],
            [CHINESE["brier_slowrho"], "保持Brier所选六参数", "0.0001", "0", "20"],
            [CHINESE["brier_warmup"], "保持Brier所选六参数", "0.0001", "10", "20"],
            [CHINESE["brier_stable"], "保持Brier所选六参数", "0.0001", "10", "50"],
            [CHINESE["brier_fixed"], "保持Brier所选六参数", "固定0.5", "0", "50"],
            [CHINESE["nll_stable"], "每 setting NLL选参", "0.0001", "10", "50"],
        ], columns=["方法", "架构/正则强度", "rho学习率", "预热epochs", "patience"])), "",
        "预热期间 rho 严格等于 0.5，shared/mass 继续学习；结束时清零 split Adam moments/steps，随后联合训练。",
        "预热方法至少给联合学习阶段完整的 stopping patience；checkpoint 始终取最低 validation NLL，包括预热期间的 checkpoint。",
        "固定 rho 仍保留相同 mass/shared 架构；split 分支冻结。几何正则在均匀分区时为零，因此它是移除自由切分的整体对照。", "",
        "NLL 对照在同一 64 个 Brier-TPE 候选中，取 baseline 和 seed101 validation NLL 最好的三个候选，",
        "用 seed102..104 的平均 validation NLL 选择。Brier 对照沿用同样四候选/三确认 seed 的原选择。",
        "两组 shortlist 可以不同；候选池由 Brier 驱动，这不等价于重新进行 NLL-TPE 搜索。",
        "优化对照固定原选架构，因此它不是对新训练流程重新调参后的最优性能估计。", "",
        "## 全部 setting 的描述性总体误差", "",
        "先对每个 seed 的 35 个 setting 等权平均，再对十个 seed 求均值和 SD。不同分布的 max_obs 不同，此汇总仅描述当前 benchmark。", "",
        _markdown(pd.DataFrame({"方法": [CHINESE[x] for x in overall.method],
            "平均grid SSE ± SD": [f"{r['mean']:.6f} ± {r['std']:.6f}" for _, r in overall.iterrows()]})), "",
        "## 每个方法、每个 setting 的十 seed 误差", ""]
    formatted = formatted_sse(output)
    for label in runs.family.drop_duplicates():
        lines.extend([f"### {label}", ""])
        subset = formatted.loc[formatted.index.str.startswith(label+"_")].copy()
        for method_set in (("common_baseline", "brier_original", "nll_original", "nll_stable"),
                           ("brier_slowrho", "brier_warmup", "brier_stable", "brier_fixed")):
            table = subset[list(method_set)].rename(columns=CHINESE).reset_index()
            lines.extend([_markdown(table), ""])
    family_table = family.copy()
    family_table["平均grid SSE ± SD"] = family_table.apply(lambda r: f"{r['mean']:.6f} ± {r['std']:.6f}", axis=1)
    family_table = family_table.pivot(index="family", columns="method", values="平均grid SSE ± SD").reindex(columns=METHODS)
    lines.extend(["## 各分布族平均误差（close 与 separate 单列）", "",
        "每个 seed 先在该分布族的七个 setting 上等权平均，再汇总十个 seed。", ""])
    for method_set in (METHODS[:4], METHODS[4:]):
        lines.extend([_markdown(family_table[list(method_set)].rename(columns=CHINESE).reset_index()), ""])
    lines.extend(["## 配对对照", "", "差值为候选减参考，小于零表示候选 grid SSE 更低。统计以相同 setting、相同 seed 配对。", "",
        _markdown(pd.DataFrame({"参考": [CHINESE[x] for x in comparisons.reference],
            "候选": [CHINESE[x] for x in comparisons.candidate],
            "平均差值": [f"{x:.6f}" for x in comparisons.mean_paired_difference],
            "改善setting": comparisons.settings_improved, "变差setting": comparisons.settings_worsened,
        "相同setting": comparisons.settings_tied})), ""])
    lookup = overall.set_index("method")
    base_value = lookup.loc["brier_original", "mean"]
    stable_value = lookup.loc["brier_stable", "mean"]
    lines.extend(["## 本次诊断的直接结果", "",
        f"保持 Brier 所选架构，原训练的总体 grid SSE 为 {base_value:.6f}，降低 rho 学习率、预热并延长 patience 后为 {stable_value:.6f}；相对变化 {(stable_value/base_value-1)*100:+.2f}%。",
        "各单步改动的改善/变差 setting 数见上表，不能将组合效果归因于某一个改动。", ""])
    selected = pd.read_csv(output/"selected_parameters.csv")
    params = ["depth", "shared_width", "shared_layers", "private_width", "private_layers", "regularization_strength"]
    br = selected[selected.selector == "brier"].set_index("setting")[params].sort_index()
    nl = selected[selected.selector == "nll"].set_index("setting")[params].sort_index()
    n_different = int((~np.isclose(br, nl, rtol=1e-10, atol=1e-10)).any(axis=1).sum())
    lines.extend([f"同一候选池中，Brier 与 NLL 对照在 {n_different}/{count} 个 setting 选择了不同的六参数组合。", ""])
    for setting in ("ph_right_q0.2", "lognormal_exact_q0"):
        subset = summary[summary.setting == setting].set_index("method")
        if len(subset):
            lines.append(f"{setting}：原 Brier 所选模型 {subset.loc['brier_original','average_grid_sse_mean']:.6f} ± {subset.loc['brier_original','average_grid_sse_std']:.6f}，组合优化后 {subset.loc['brier_stable','average_grid_sse_mean']:.6f} ± {subset.loc['brier_stable','average_grid_sse_std']:.6f}。")
            lines.append("")
    lines.extend([
        "## 解释范围与复现", "",
        "已完成的 28 个 setting 之前看过测试结果，本次是在相同数据上的探索性机制诊断。新增 close 的 7 个 setting 首次评估。",
        "全部新选择在新测试评估前冻结，选择过程不读取 test SSE/true CDF；不能将这次结果解释为独立数据上的确认性验证。",
        "十个 seed 只反映同一数据集上的初始化/优化差异，不反映数据采样误差。更慢的 rho 学习率和预热没有消除硬边界的非光滑性，",
        "几何正则仍不控制 mass 复杂度；self-imputed Brier 的候选相关目标偏差也仍存在。不能据此声称统一流程已经实现统计一致性或所有 setting 都改善。", "",
        "`selected_parameters.csv` 保存全部六参数及选参分数；`runs.csv` 保存每个 seed 的 grid SSE、NLL、Brier、优化设置、训练历史和曲线路径。",
        "`summary.csv` 与 `all_methods_grid_sse.csv` 包含全部 35×8 个十-seed 汇总；`family_grid_sse.csv` 单独比较 close/separate。",
        "`config.json` 记录源码与输入结果哈希，`selection_complete.json` 冻结所有选择，`audit.json` 记录复用/新拟合数量。", ""])
    (output/"grid_sse_report.md").write_text("\n".join(lines), encoding="utf-8")
    return overall, family, comparisons


def plot_diagnostic_summary(output_dir):
    output = Path(output_dir)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(output/"summary.csv")
    pivot = summary.pivot(index="setting", columns="method", values="average_grid_sse_mean").reindex(columns=METHODS)
    fig, ax = plt.subplots(figsize=(12, 13), constrained_layout=True)
    ratios = pivot.divide(pivot.common_baseline, axis=0)
    shown = ax.imshow(np.log2(ratios.clip(lower=1e-3)), vmin=-2, vmax=2, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_title("Ten-seed mean grid SSE / common baseline (blue = lower)")
    for i, row in enumerate(ratios.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(np.log2(max(value, 1e-3))) > 1.2 else "black")
    fig.colorbar(shown, ax=ax, label="log2 SSE ratio", shrink=.6)
    fig.savefig(figures/"sse_ratios.png", dpi=150)
    return fig


def plot_mixture_cdfs(output_dir):
    output = Path(output_dir)
    runs = pd.read_csv(output/"runs.csv")
    methods = ("common_baseline", "brier_original", "brier_stable", "brier_fixed", "nll_stable")
    fig, axes = plt.subplots(2, 7, figsize=(23, 7), constrained_layout=True)
    for i, family in enumerate(("multimodal_close", "multimodal_separate")):
        setting_names = runs.loc[runs.family == family, "setting"].drop_duplicates().tolist()
        for j, name in enumerate(setting_names):
            ax = axes[i, j]
            for k, method in enumerate(methods):
                files = runs.loc[(runs.setting == name)&(runs.method == method), "curves_file"]
                arrays = [np.load(path) for path in files]
                grid = arrays[0]["grid"]
                curves = np.stack([1-a["survival"][0] for a in arrays])
                mean, std = curves.mean(0), curves.std(0, ddof=1)
                ax.plot(grid, mean, label=LABELS[method], color=f"C{k}", lw=1.3)
                ax.fill_between(grid, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1), color=f"C{k}", alpha=.09)
                if k == 0:
                    ax.plot(grid, 1-arrays[0]["truth"][0], color="black", ls="--", label="True CDF", lw=1.6)
                for array in arrays:
                    array.close()
            ax.set_title(name.replace(family+"_", ""), fontsize=10)
            ax.set_ylim(-.02, 1.02)
            ax.grid(alpha=.2)
            if j == 0:
                ax.set_ylabel(f"{family}\nCDF")
            if i == 1:
                ax.set_xlabel("time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6)
    fig.suptitle("First test subject: mean CDF across ten fits, shaded seed SD (illustration only)")
    fig.savefig(output/"figures"/"multimodal_close_separate_cdfs.png", dpi=150)
    return fig


def plot_problem_histories(output_dir):
    output = Path(output_dir)
    runs = pd.read_csv(output/"runs.csv")
    cases = (("ph_right_q0.2", 0), ("lognormal_exact_q0", 2), ("multimodal_close_current_q0.5", 0))
    methods = ("brier_original", "brier_slowrho", "brier_warmup", "brier_stable", "brier_fixed")
    fig, axes = plt.subplots(2, len(cases), figsize=(14, 7), constrained_layout=True)
    for j, (setting, seed) in enumerate(cases):
        for k, method in enumerate(methods):
            row = runs[(runs.setting == setting)&(runs.seed == seed)&(runs.method == method)].iloc[0]
            with gzip.open(row.histories_file, "rt", encoding="utf-8") as stream:
                history = pd.DataFrame(json.load(stream))
            axes[0, j].plot(history.epoch, history.get("train_nll"), color=f"C{k}", label=LABELS[method])
            valid = history.dropna(subset=["validation_nll"])
            axes[1, j].plot(valid.epoch, valid.validation_nll, color=f"C{k}")
            axes[1, j].scatter([row.best_epoch], [row.validation_nll], color=f"C{k}", s=16)
        axes[0, j].set_title(f"{setting}, seed {seed}", fontsize=10)
        axes[1, j].set_xlabel("epoch")
        for ax in axes[:, j]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Training observed NLL")
    axes[1, 0].set_ylabel("Validation observed NLL")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=5)
    fig.savefig(output/"figures"/"optimization_histories.png", dpi=150)
    return fig
