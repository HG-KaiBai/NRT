"""Tables and static scientific figures for the completed tuning notebook."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .experiment import ExperimentConfig,settings
from .search_io import write_csv as _write_csv


def report_tables(output_dir):
    output = Path(output_dir)
    runs = pd.read_csv(output/"final_runs.csv")
    summary = pd.read_csv(output/"final_summary.csv")
    best = pd.read_csv(output/"best_parameters.csv")
    means = summary.pivot(index="setting",columns="method",values="average_grid_sse_mean")
    means["sse_ratio"] = means.per_setting_tuned/means.common_baseline
    means["improvement_percent"] = 100*(1-means.sse_ratio)
    table = best.merge(means.reset_index(),on="setting")
    deviations = summary.pivot(index="setting",columns="method",values="average_grid_sse_std")
    deviations.columns = [f"{name}_std" for name in deviations.columns]
    table = table.merge(deviations.reset_index(),on="setting")
    _write_csv(table,output/"parameter_comparison.csv")
    error_columns = ["setting","method","n_runs","average_grid_sse_mean","average_grid_sse_std"]
    _write_csv(summary[error_columns],output/"grid_sse_by_setting.csv")
    historical = output.parent/"depth_regularization_sse"/"summary.csv"
    old_config = historical.with_name("config.json")
    matched_data = False
    if old_config.exists() and (output/"config.json").exists():
        previous = json.loads(old_config.read_text(encoding="utf-8"))["experiment"]
        current_config = json.loads((output/"config.json").read_text(encoding="utf-8"))["experiment"]
        matched_data = all(previous.get(k)==current_config.get(k) for k in (
            "n_samples","data_seed","train_fraction","validation_fraction","grid_size",
            "scenarios","censor_types","censor_qs","multimodal_case","resample_data","seeds"))
    if historical.exists() and matched_data:
        old = pd.read_csv(historical)
        old = old[["setting","method","depth","n_runs","average_grid_sse_mean","average_grid_sse_std"]]
        old["experiment"] = "common_strength_depth_comparison"
        depths = runs.groupby(["setting","method"]).depth.first().reset_index()
        current = summary[error_columns].merge(depths,on=["setting","method"])
        current["experiment"] = "per_setting_six_parameter_tuning"
        selected_path = historical.with_name("validation_selected_depth_summary.csv")
        pieces = [old,current]
        if selected_path.exists():
            selected = pd.read_csv(selected_path)[error_columns+["mean_depth"]]
            selected["depth"] = "selected_by_NLL"
            selected["experiment"] = "common_strength_NLL_depth_selection"
            pieces.append(selected)
        _write_csv(pd.concat(pieces,ignore_index=True),output/"all_methods_grid_sse.csv")
    # First average equally over settings within seed, then summarize seeds.
    by_seed = runs.groupby(["method","seed"]).average_grid_sse.mean().reset_index()
    overall = by_seed.groupby("method").agg(mean_grid_sse=("average_grid_sse","mean"),
        std_grid_sse=("average_grid_sse","std"),n_seeds=("seed","size")).reset_index()
    _write_csv(overall,output/"overall_grid_sse.csv")
    by_type_seed = runs.groupby(["censor_type","method","seed"]).average_grid_sse.mean().reset_index()
    by_type = by_type_seed.groupby(["censor_type","method"]).agg(
        mean_grid_sse=("average_grid_sse","mean"),std_grid_sse=("average_grid_sse","std")).reset_index()
    _write_csv(by_type,output/"by_censor_type_grid_sse.csv")
    write_sse_report(output,table,overall)
    return table,overall,by_type


def formatted_grid_sse(output_dir):
    """Primary table: all settings, both methods, ten-fit mean/SD."""
    summary = pd.read_csv(Path(output_dir)/"final_summary.csv")
    summary["mean +/- SD"] = [f"{m:.6f} +/- {s:.6f}" for m,s in
        zip(summary.average_grid_sse_mean,summary.average_grid_sse_std)]
    return summary.pivot(index="setting",columns="method",values="mean +/- SD")


def _markdown(frame):
    frame = frame.reset_index()
    lines = ["| "+" | ".join(map(str,frame.columns))+" |",
             "| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines += ["| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def write_sse_report(output,comparison,overall):
    """Full Chinese report with every setting/method's ten-seed grid SSE."""
    main = formatted_grid_sse(output).reindex(comparison.setting)
    main.columns = ["共同参数基准","逐 setting 调参"]
    paragraphs = [
        "以下数值均为 10 次独立初始化的平均 grid SSE ± seed 间样本标准差。"
        "每次先在 256 个时间点上求平方误差之和，再对 test sample 取平均；最后对 10 次 seed 取平均。",
        "本轮：共同参数基准与逐 setting 的六参数调参。Brier 只用于训练完成后的参数选择，"
        "梯度更新和 checkpoint 使用 NLL。",_markdown(main)]
    aggregate = overall.set_index("method")
    base,chosen = aggregate.loc["common_baseline"],aggregate.loc["per_setting_tuned"]
    paragraphs.insert(2,f"28 个 setting 等权平均的 grid SSE：共同参数基准 "
        f"{base.mean_grid_sse:.6f} ± {base.std_grid_sse:.6f}；逐 setting 调参 "
        f"{chosen.mean_grid_sse:.6f} ± {chosen.std_grid_sse:.6f}。这里的标准差先对每个 seed "
        "跨 setting 求平均，再在 10 个 seed 之间计算；各 setting 的标准差见下表。")
    wins = int((comparison.sse_ratio<1).sum())
    paragraphs.append(f"逐 setting 调参在 {wins}/{len(comparison)} 个 setting 上降低了平均 test grid SSE。"
        "这些结果来自每个 setting 的一个固定数据集；seed 标准差反映优化波动。")
    scores = pd.read_csv(output/"final_summary.csv").pivot(index="setting",columns="method",
                                                           values="validation_brier_mean")
    brier_ratio = scores.per_setting_tuned/scores.common_baseline
    mismatch = int(((brier_ratio<1-1e-8)&(comparison.set_index("setting").sse_ratio>1+1e-8)).sum())
    paragraphs.append(f"最终 10 个 seed 中，有 {mismatch} 个 setting 的平均验证 Brier 更低，"
        "但平均 test grid SSE 更高。原版 self-imputed Brier 的改善不能直接解释为真实 CDF 拟合的改善。")
    historical_path = output/"all_methods_grid_sse.csv"
    if historical_path.exists():
        all_errors = pd.read_csv(historical_path)
        historical = all_errors[all_errors.experiment.str.startswith("common_strength")].copy()
        historical["mean ± SD"] = [f"{m:.6f} ± {s:.6f}" for m,s in
            zip(historical.average_grid_sse_mean,historical.average_grid_sse_std)]
        names = dict(fixed_nll="固定 rho",learned_nll="learned rho，无正则",
                     partition_nll="learned rho，正则 a=500")
        paragraphs.append("此前完成的三种方法也按每个 setting 列在下方。历史实验与新共享网络实现"
                          "的初始化有所不同，因此分别列出原始结果。")
        for depth in ("3","6","8","selected_by_NLL"):
            part = historical[historical.depth.astype(str).str.replace(".0","",regex=False)==depth]
            if part.empty:
                continue
            title = "历史比较：验证 NLL 选择树深度。" if depth=="selected_by_NLL" else f"历史比较：树深度 {depth}。"
            frame = part.pivot(index="setting",columns="method",values="mean ± SD")
            frame = frame.reindex(comparison.setting).rename(columns=names)
            paragraphs.extend([title,_markdown(frame)])
    (output/"grid_sse_report.md").write_text("\n\n".join(paragraphs)+"\n",encoding="utf-8")


def plot_search(output_dir):
    output = Path(output_dir)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    trials = pd.read_csv(output/"trials.csv")
    fig,axes = plt.subplots(4,7,figsize=(21,11),layout="constrained")
    for ax,(name,frame) in zip(axes.flat,trials.groupby("setting",sort=False)):
        frame = frame.sort_values("trial")
        baseline = float(frame.iloc[0].validation_brier)
        ax.plot(frame.trial,frame.validation_brier/baseline,".",alpha=.35,color="gray")
        ax.plot(frame.trial,np.minimum.accumulate(frame.validation_brier)/baseline,color="C0")
        ax.axvline(15.5,color="gray",lw=.7,ls="--")
        ax.axhline(1,color="gray",lw=.6)
        ax.set_title(name,fontsize=9)
        ax.set_xlabel("trial")
        ax.set_ylabel("Brier / baseline")
    fig.suptitle("Validation Brier search: anchors + Sobol, then adaptive TPE (single search seed)")
    fig.savefig(figures/"search_traces.png",dpi=150)
    return fig


def plot_comparison(output_dir):
    output = Path(output_dir)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    table,overall,by_type = report_tables(output)
    summary = pd.read_csv(output/"final_summary.csv")
    fig,axes = plt.subplots(1,2,figsize=(16,11),layout="constrained")
    names = list(table.setting)
    pos = np.arange(len(names))
    for index,(method,label) in enumerate((("common_baseline","Common baseline"),("per_setting_tuned","Per-setting tuned"))):
        values = summary[summary.method==method].set_index("setting").loc[names]
        axes[0].errorbar(values.average_grid_sse_mean,pos+(index-.5)*.18,
            xerr=values.average_grid_sse_std,fmt="o",ms=4,capsize=2,label=label)
    axes[0].set_yticks(pos,names,fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Test grid SSE: mean +/- SD of 10 fits (log axis)")
    axes[0].legend()
    colors = np.where(table.sse_ratio<1,"C0","C3")
    axes[1].barh(pos,table.improvement_percent,color=colors)
    axes[1].set_yticks(pos,names,fontsize=8)
    axes[1].invert_yaxis()
    axes[1].axvline(0,color="gray",lw=1)
    axes[1].set_xlabel("Test SSE reduction vs common baseline (%)")
    fig.suptitle("Hyperparameters selected by observed validation Brier; test CDF used only here")
    fig.savefig(figures/"tuned_vs_baseline.png",dpi=150)
    return fig


def plot_cdf(output_dir,config=ExperimentConfig()):
    output = Path(output_dir)
    figures = output/"figures"
    figures.mkdir(exist_ok=True)
    fig,axes = plt.subplots(4,7,figsize=(21,11),layout="constrained")
    for ax,setting in zip(axes.flat,settings(config)):
        for index,(method,label) in enumerate((("common_baseline","Baseline"),("per_setting_tuned","Tuned"))):
            curves = [np.load(output/"curves"/f"{setting.name}_{method}_seed{seed}.npz") for seed in config.seeds]
            grid,truth = curves[0]["grid"],1-curves[0]["truth"][0]
            predicted = np.stack([1-c["survival"][0] for c in curves])
            mean,std = predicted.mean(0),predicted.std(0,ddof=1)
            ax.plot(grid,mean,label=label,color=f"C{index}")
            ax.fill_between(grid,np.maximum(0,mean-std),np.minimum(1,mean+std),color=f"C{index}",alpha=.15)
            for curve in curves:
                curve.close()
        ax.plot(grid,truth,"k--",label="True CDF")
        ax.set_title(setting.name,fontsize=9)
        ax.set_ylim(0,1)
        ax.set_xlabel("time")
        ax.set_ylabel("CDF")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("First test subject in each setting: mean CDF +/- seed SD (illustration only)")
    fig.savefig(figures/"cdf_all_settings.png",dpi=150)
    return fig


def plot_score_alignment(output_dir):
    """Compare validation selection score with the user's actual test metric."""
    output = Path(output_dir)
    summary = pd.read_csv(output/"final_summary.csv")
    brier = summary.pivot(index="setting",columns="method",values="validation_brier_mean")
    sse = summary.pivot(index="setting",columns="method",values="average_grid_sse_mean")
    table = summary[["setting","censor_type"]].drop_duplicates().set_index("setting")
    table["validation_brier_ratio"] = brier.per_setting_tuned/brier.common_baseline
    table["test_grid_sse_ratio"] = sse.per_setting_tuned/sse.common_baseline
    _write_csv(table.reset_index(),output/"score_alignment.csv")
    fig,ax = plt.subplots(figsize=(9,6),layout="constrained")
    for censor,frame in table.groupby("censor_type"):
        ax.scatter(frame.validation_brier_ratio,frame.test_grid_sse_ratio,label=censor,alpha=.8)
    ax.axvline(1,color="gray",ls="--")
    ax.axhline(1,color="gray",ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Validation Brier ratio: tuned / common baseline (10-seed means)")
    ax.set_ylabel("Test grid SSE ratio: tuned / common baseline (10-seed means)")
    ax.set_title("Lower self-imputed validation Brier can accompany higher test CDF error")
    ax.legend(title="Observation scheme")
    worst = table.test_grid_sse_ratio.nlargest(3).index
    for name in worst:
        row = table.loc[name]
        ax.annotate(name,(row.validation_brier_ratio,row.test_grid_sse_ratio),
                    xytext=(4,5),textcoords="offset points",fontsize=8)
    (output/"figures").mkdir(exist_ok=True)
    fig.savefig(output/"figures"/"brier_vs_sse.png",dpi=150)
    return fig
