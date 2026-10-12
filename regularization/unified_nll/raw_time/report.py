"""Paired grid SSE reports and complete audits of the NLL-driven rho study."""
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..cv_search import fold_indices, observed_pool, refit_epoch_count
from ..depth_experiment import _evaluation_data
from ..diagnostic_experiment import expanded_settings
from ..experiment import ExperimentConfig
from ..model import ModelConfig
from .search import METHODS, MATCHED, SEARCH_METHODS, SOURCE_FILES
from .model import observed_upper_bound, SUPPORT
from ..parameter_search import _optuna
from ..search_io import write_csv
from ..tuning_model import Architecture


def audit_nll_cv(output_dir):
    output = Path(output_dir)
    manifest = json.loads((output/"config.json").read_text(encoding="utf-8"))
    assert manifest["support"] == SUPPORT and manifest["time_coordinate"] == "raw"
    frozen = json.loads((output/"selection_complete.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for name in SOURCE_FILES:
        digest.update((Path(__file__).parent/name).read_bytes())
    assert digest.hexdigest() == manifest["source_sha256"], "numerical source changed"
    assert frozen["fingerprint"] == manifest["fingerprint"]
    assert frozen["selected_sha256"] == hashlib.sha256((output/"selected_parameters.csv").read_bytes()).hexdigest()
    runs, summary, selected, folds, trials = [pd.read_csv(output/name, float_precision="round_trip") for name in
        ("runs.csv","summary.csv","selected_parameters.csv","cv_folds.csv","trials.csv")]
    n_settings, n_trials, n_folds = manifest["settings"], manifest["search"]["trials"], manifest["search"]["folds"]
    seeds = set(manifest["experiment"]["seeds"])
    assert set(manifest["methods"]) == set(METHODS) and frozen["settings"] == n_settings
    assert len(selected) == n_settings*len(METHODS) and len(summary) == len(selected)
    assert len(trials) == n_settings*n_trials*len(SEARCH_METHODS)
    assert len(folds) == n_settings*(len(SEARCH_METHODS)*n_trials+1)*n_folds
    assert len(runs) == n_settings*len(METHODS)*len(seeds)
    assert not selected.duplicated(["setting","method"]).any()
    assert not trials.duplicated(["setting","method","trial"]).any()
    assert not folds.duplicated(["setting","method","trial","fold"]).any()
    assert not runs.duplicated(["setting","method","seed"]).any()
    np.testing.assert_allclose(trials.objective_value, trials.cv_nll, rtol=1e-12)
    plan = json.loads((output/"folds.json").read_text(encoding="utf-8"))
    assert sorted(sum(plan["validation_indices"],[])) == list(range(manifest["n_fit_pool"]))
    assert folds.n_fit.add(folds.n_valid).eq(manifest["n_fit_pool"]).all()
    splits = fold_indices(manifest["n_fit_pool"],n_folds,manifest["search"]["fold_seed"])
    assert plan["validation_indices"] == [v.tolist() for _,v in splits]
    assert folds.seed.eq(manifest["search"]["search_fit_seed"]).all()
    config = ExperimentConfig(**manifest["experiment"])
    setting_map = {name:(setting,actual) for name,_,setting,actual in expanded_settings(config)}
    cv_data = {}
    for setting_name,(setting,actual) in setting_map.items():
        pool = observed_pool(setting,actual,Architecture(**manifest["baseline"]))["pool"]
        cv_data[setting_name] = [(len(t),len(v),observed_upper_bound(tuple(x[t] for x in pool))) for t,v in splits]
    warmup = manifest["optimization"]["warmup_epochs"]
    assert folds.best_epoch.gt(warmup).all() and runs.best_epoch.gt(warmup).all()
    assert runs.n_fit.eq(manifest["n_fit_pool"]).all()
    assert runs.seed.eq(runs.data_seed).all() and runs.seed.eq(runs.initialization_seed).all()
    assert not runs.data_seed.eq(manifest["experiment"]["data_seed"]).any()
    for frame in (folds,runs):
        adaptive = frame.method.eq(SEARCH_METHODS[1])
        assert frame.learn_splits.eq(adaptive).all()
        assert frame.loc[adaptive,"parameters"].eq(frame.loc[adaptive,"trainable_parameters"]).all()
        assert frame.loc[~adaptive,"parameters"].gt(frame.loc[~adaptive,"trainable_parameters"]).all()
        assert frame.loc[~adaptive,"regularization_strength"].eq(0).all()
        assert frame.loc[~adaptive,"rho_mean_absolute_shift"].eq(0).all()
        assert frame.loc[~adaptive,"rho_root_mean_absolute_shift"].eq(0).all()
        np.testing.assert_allclose(frame.lambda_n,frame.regularization_strength/np.sqrt(frame.n_fit),rtol=1e-12)
    chosen = selected.set_index(["setting","method"])
    trial_index = trials.set_index(["setting","method","trial"])
    # Verify the actual Optuna feedback, not only exported CSV labels.
    optuna = _optuna()
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{output.resolve()/'nll_cv.sqlite3'}")
    try:
        for (setting,method),frame in trials.groupby(["setting","method"]):
            assert len(frame) == n_trials
            study = optuna.load_study(study_name=f"{setting}_{method}",storage=storage)
            completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
            assert len(completed) == n_trials
            for trial in completed:
                row = trial_index.loc[(setting,method,trial.number)]
                np.testing.assert_allclose(trial.value,row.cv_nll,rtol=1e-12)
                assert ("log10_one_plus_strength" in trial.params) == (method == SEARCH_METHODS[1])
            expected = frame.sort_values(["cv_nll","trial"]).iloc[0]
            selection = chosen.loc[(setting,method)]
            assert selection.trial == expected.trial
            np.testing.assert_allclose(selection.cv_nll,expected.cv_nll,rtol=1e-12)
    finally:
        storage.remove_session()
        storage.engine.dispose()
    for (setting,method,trial),frame in folds.groupby(["setting","method","trial"]):
        assert set(frame.fold) == set(range(n_folds))
        row = chosen.loc[(setting,method)] if method == MATCHED else trial_index.loc[(setting,method,trial)]
        np.testing.assert_allclose(row.cv_nll,np.average(frame.validation_nll,weights=frame.n_valid),rtol=1e-12)
        assert row.refit_epochs == refit_epoch_count(frame.best_epoch.tolist(),warmup)
        for fit in frame.itertuples():
            n_fit,n_valid,scale = cv_data[setting][fit.fold]
            assert fit.n_fit == n_fit and fit.n_valid == n_valid
            np.testing.assert_allclose(fit.upper_bound,scale,rtol=1e-12)
            path = output/"histories"/f"{setting}_{method}_trial{trial}_fold{fit.fold}.json.gz"
            with gzip.open(path,"rt",encoding="utf-8") as handle:
                history = json.load(handle)
            assert len(history) == fit.epochs_run
            eligible = [h for h in history if h["checkpoint_eligible"] and "validation_nll" in h]
            checkpoint = min(eligible,key=lambda h:h["validation_nll"])
            assert checkpoint["epoch"] == fit.best_epoch
            np.testing.assert_allclose(checkpoint["validation_nll"],fit.validation_nll,rtol=1e-12)
            assert all("brier" not in k for h in history for k in h)
            assert all(h["split_active"] == (fit.learn_splits and h["epoch"] > warmup) for h in history)
            if not fit.learn_splits:
                np.testing.assert_allclose([h["regularizer"] for h in history],0,atol=1e-10)
    architecture_keys = list(manifest["baseline"])
    for setting in selected.setting.unique():
        adaptive,matched = chosen.loc[(setting,SEARCH_METHODS[1])],chosen.loc[(setting,MATCHED)]
        assert matched.source_adaptive_trial == adaptive.trial
        for key in architecture_keys:
            if key != "regularization_strength":
                assert matched[key] == adaptive[key]
    for (setting,method),frame in runs.groupby(["setting","method"]):
        selection = chosen.loc[(setting,method)]
        assert set(frame.seed) == seeds and frame.trial.eq(selection.trial).all()
        assert frame.epochs_run.eq(selection.refit_epochs).all() and frame.best_epoch.eq(frame.epochs_run).all()
        for key in architecture_keys:
            np.testing.assert_allclose(frame[key],selection[key],rtol=1e-12)
    for statistic in ("mean","std"):
        expected = runs.groupby(["setting","method"]).average_grid_sse.agg(statistic).sort_index()
        actual = summary.set_index(["setting","method"])["average_grid_sse_"+statistic].sort_index()
        np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert np.isfinite(runs.average_grid_sse).all()
    assert not np.isnan(runs.test_nll).any() and not np.isneginf(runs.test_nll).any()
    for frame in (folds, runs):
        assert frame.support.eq(SUPPORT).all() and frame.time_coordinate.eq("raw").all()
        np.testing.assert_allclose(frame.upper_bound, 1.1*frame.max_observation, rtol=1e-12)
    regenerated = {}
    for row in runs.itertuples():
        key = row.setting,row.seed
        if key not in regenerated:
            setting,actual = setting_map[row.setting]
            data = _evaluation_data(setting,actual,ModelConfig(x_dim=manifest["baseline"]["x_dim"]),row.seed)
            pool = tuple(torch.cat((a,b)) for a,b in zip(data["train"],data["valid"]))
            regenerated[key] = (data["grid"].numpy(),data["truth"][:3].numpy(),observed_upper_bound(pool),len(data["test"][0]))
        grid,truth,scale,n_test = regenerated[key]
        assert row.n_test == n_test
        np.testing.assert_allclose(row.upper_bound,scale,rtol=1e-12)
        np.testing.assert_allclose(row.max_obs,grid[-1],rtol=1e-12)
        with np.load(row.curves_file) as archive:
            np.testing.assert_array_equal(archive["grid"],grid)
            np.testing.assert_array_equal(archive["truth"],truth)
            survival,errors = archive["survival"],archive["subject_grid_sse"]
            assert survival.shape == truth.shape == (3,config.grid_size)
            assert errors.shape == (n_test,) and np.isfinite(errors).all()
            np.testing.assert_allclose(errors.mean(),row.average_grid_sse,atol=2e-7,rtol=2e-6)
            np.testing.assert_allclose(((survival-truth)**2).sum(-1),errors[:3],atol=2e-7,rtol=2e-5)
            assert np.isfinite(survival).all() and survival.min() >= -1e-6 and survival.max() <= 1+1e-6
            assert np.diff(survival,axis=1).max() <= 1e-6
        with gzip.open(row.histories_file,"rt",encoding="utf-8") as handle:
            history = json.load(handle)
        assert len(history) == row.epochs_run and history[-1]["checkpoint_eligible"]
        assert all("validation_nll" not in h for h in history)
        assert all(h["split_active"] == (row.learn_splits and h["epoch"] > warmup) for h in history)
    result = dict(passed=True,settings=n_settings,independent_nll_studies=n_settings*2,
        search_candidates=len(trials),cv_fold_fits=len(folds),final_fits=len(runs),
        regenerated_train_test_pairs=len(regenerated),setting_method_groups=len(summary),
        data_seeds=sorted(seeds),cv_data_seed=config.data_seed,nll_drives_tpe_and_selection=True,
        fixed_rho_exactly_half=True,adaptive_rho_trainable=True,test_used_for_stopping=False,
        support=SUPPORT, train_only_upper_bounds_verified=True,
        infinite_test_nll_count=int(np.isposinf(runs.test_nll).sum()))
    (output/"audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def paired_sse(output_dir):
    output = Path(output_dir)
    runs,summary = pd.read_csv(output/"runs.csv"),pd.read_csv(output/"summary.csv")
    manifest = json.loads((output/"config.json").read_text(encoding="utf-8"))
    order = [name for name,_,_,_ in expanded_settings(ExperimentConfig(**manifest["experiment"]))]
    summary = summary.set_index(["setting","method"])
    paired = runs.pivot(index=["setting","seed"],columns="method",values="average_grid_sse")
    rows = []
    for setting in order:
        frame = paired.loc[setting]
        row = dict(setting=setting)
        for method in METHODS:
            group = summary.loc[(setting,method)]
            row[method] = f"{group.average_grid_sse_mean:.6f} ± {group.average_grid_sse_std:.6f}"
        for reference,label in ((SEARCH_METHODS[0],"tuned"),(MATCHED,"matched")):
            delta = frame[SEARCH_METHODS[1]]-frame[reference]
            row[f"adaptive_minus_{label}_fixed_mean"] = delta.mean()
            row[f"adaptive_minus_{label}_fixed_std"] = delta.std(ddof=1)
            row[f"adaptive_gain_vs_{label}_fixed_pct"] = 100*(1-frame[SEARCH_METHODS[1]].mean()/frame[reference].mean())
            row[f"adaptive_wins_vs_{label}_fixed_seeds"] = int((delta < -1e-12).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown(frame):
    return "\n".join(["| "+" | ".join(map(str,frame.columns))+" |",
        "| "+" | ".join("---" for _ in frame.columns)+" |"]+
        ["| "+" | ".join(map(str,row))+" |" for row in frame.itertuples(index=False,name=None)])


def report_nll_cv(output_dir):
    output = Path(output_dir)
    audit = audit_nll_cv(output)
    manifest = json.loads((output/"config.json").read_text(encoding="utf-8"))
    n_seeds = len(manifest["experiment"]["seeds"])
    runs = pd.read_csv(output/"runs.csv")
    selected = pd.read_csv(output/"selected_parameters.csv")
    trials = pd.read_csv(output/"trials.csv")
    architecture_keys = ["depth","shared_width","shared_layers","private_width",
                         "private_layers","regularization_strength"]
    coverage = pd.DataFrame([dict(setting=setting,method=method,trials=len(frame),
        unique_configurations=len(frame.drop_duplicates(architecture_keys)))
        for (setting,method),frame in trials.groupby(["setting","method"])])
    write_csv(coverage,output/"search_coverage.csv")
    coverage_summary = coverage.groupby("method").unique_configurations.agg(["mean","min","max"]).reset_index()
    overall = runs.groupby(["method","seed"]).average_grid_sse.mean().groupby("method").agg(
        ["mean","std"]).reindex(METHODS).reset_index()
    family = runs.groupby(["family","method","seed"]).average_grid_sse.mean().groupby(
        ["family","method"]).agg(["mean","std"]).reset_index()
    paired = paired_sse(output)
    write_csv(overall,output/"overall_grid_sse.csv")
    write_csv(family,output/"family_grid_sse.csv")
    write_csv(paired,output/"paired_grid_sse.csv")
    write_csv(paired[["setting",*METHODS]],output/"all_methods_grid_sse.csv")
    comparisons = []
    values = overall.set_index("method")
    for reference,label in ((SEARCH_METHODS[0],"tuned"),(MATCHED,"matched")):
        delta = paired[f"adaptive_minus_{label}_fixed_mean"]
        comparisons.append(dict(reference=reference,candidate=SEARCH_METHODS[1],
            mean_difference=delta.mean(),adaptive_better=int((delta < -1e-12).sum()),
            adaptive_worse=int((delta > 1e-12).sum()),ties=int((delta.abs() <= 1e-12).sum()),
            aggregate_gain_pct=100*(1-values.loc[SEARCH_METHODS[1],"mean"]/values.loc[reference,"mean"])))
    comparisons = pd.DataFrame(comparisons)
    write_csv(comparisons,output/"paired_comparisons.csv")
    compact = paired[["setting",*METHODS]].copy()
    for label in ("tuned","matched"):
        compact[f"Δ adaptive−{label} fixed"] = paired.apply(lambda r:
            f"{r[f'adaptive_minus_{label}_fixed_mean']:+.6f} ± {r[f'adaptive_minus_{label}_fixed_std']:.6f}",axis=1)
    diag = pd.read_csv(output/"summary.csv")
    adaptive = selected[selected.method == SEARCH_METHODS[1]].merge(
        diag[diag.method == SEARCH_METHODS[1]][["setting","rho_mean_absolute_shift","rho_root_mean_absolute_shift"]],on="setting")
    write_csv(adaptive,output/"selected_adaptive_penalty_rho.csv")
    lines = ["# 原时间有限支持 KL：固定 rho 与 adaptive rho", "",
        "使用原时间 t，树支持域为 [0,U]，U=1.1×当前拟合子集最大有限观测端点。叶内原时间密度均匀，不使用非线性变换。两个独立搜索仍由验证 NLL 驱动。", "",
        f"独立调参后，adaptive 的整体平均 grid SSE 为 {values.loc[SEARCH_METHODS[1],'mean']:.6f}，固定 rho 为 {values.loc[SEARCH_METHODS[0],'mean']:.6f}；adaptive 低 {comparisons.iloc[0].aggregate_gain_pct:.2f}%，在 {int(comparisons.iloc[0].adaptive_better)}/{audit['settings']} 个 setting 更好、{int(comparisons.iloc[0].adaptive_worse)} 个更差。", "",
        f"同架构固定对照的整体平均 SSE 为 {values.loc[MATCHED,'mean']:.6f}，adaptive 低 {comparisons.iloc[1].aggregate_gain_pct:.2f}%；在 {int(comparisons.iloc[1].adaptive_better)}/{audit['settings']} 个 setting 更好、{int(comparisons.iloc[1].adaptive_worse)} 个更差。同架构的两个模型各自使用 CV 确定训练轮数。", "",
        f"共 {audit['settings']} 个 setting，每个方法、每个 setting 都有 {n_seeds} 组 train–test 数据的 grid SSE。", "",
        "## 每个 setting 的平均 grid SSE 与配对差距", "",
        f"前三列为 {n_seeds} 次 mean ± 样本 SD。Δ 是同一数据 seed 下 adaptive SSE 减固定 SSE，再对 {n_seeds} 次差值取 mean ± SD；负值有利于 adaptive。", "",
        "fixed_nll_cv 与 adaptive_nll_cv 分别调参，比较的是两个调优系统；fixed_at_adaptive_architecture 使用 adaptive 选中的同一架构，rho 固定为 0.5，训练轮数仍按自己的 CV NLL 确定。", "",
        _markdown(compact), "", "## 全部 setting 等权平均", "",
        f"先在每个数据 seed 内平均所有 setting，再对 {n_seeds} 次取 mean/SD。", "",_markdown(overall),"",
        _markdown(comparisons),"", "## 各分布族", "",_markdown(family),"",
        "## 选参和最终评估协议", "",
        f"选参数据 seed={manifest['experiment']['data_seed']}，只使用前 {manifest['n_fit_pool']} 个观测样本。CV 折数={manifest['search']['folds']}，各折上界只由该折拟合端点决定。初始化 seed={manifest['search']['search_fit_seed']}。",
        f"每个 setting 的两个主方法各 {manifest['search']['trials']} 个候选，其中 {manifest['search']['initial_trials']} 个 startup 候选；其余由 NLL 驱动的 TPE 提议。候选完成全部折后才评分。",
        f"固定 rho 搜索五个架构参数、a=0；adaptive 另搜索 a，范围0..{manifest['search']['strength_max']}。两者均使用原时间叶密度和以 U/L 为目标的叶宽 KL。",
        f"训练设置：{manifest['training']}；优化设置：{manifest['optimization']}。checkpoint 与选参仅使用未加正则验证 NLL。",
        "同架构固定对照仅用 CV 确定训练轮数。最终轮数为所选四折（或配置指定折数）最佳轮数中位数的上取整。",
        f"全部选参全局冻结后，按 seeds={manifest['experiment']['seeds']} 分别生成数据并固定轮数拟合；测试不参与选参、支持域上界或停止。",
        f"SSE 网格为0..最终训练池最大有限端点，{manifest['experiment']['grid_size']}个等距点。逐测试对象求平方误差之和后平均。测试零似然对应无穷 NLL 的记录数为 {audit['infinite_test_nll_count']}。",
        "## 所选参数", "",_markdown(selected[["setting","method","depth","shared_width","shared_layers",
            "private_width","private_layers","regularization_strength","refit_epochs","cv_nll"]]),"",
        "## adaptive 所选 penalty 与 rho 偏移", "",
        f"偏移分别对全部节点和根节点计算 |rho−0.5|，再平均 {n_seeds} 次最终拟合；深树的全部节点均值可能掩盖上层节点变化。", "",
        _markdown(adaptive[["setting","depth","regularization_strength","rho_mean_absolute_shift","rho_root_mean_absolute_shift"]]),"",
        "## 搜索覆盖", "",
        f"每个主方法使用相同的 {manifest['search']['trials']} 次 trial 预算。固定模型是离散空间，TPE 可能重复架构。下表报告实际不同配置数量，逐 setting 见 search_coverage.csv。", "",
        _markdown(coverage_summary), "", "## 解释范围", "",
        "SD 包含数据抽样与初始化变化，以本次调参数据为条件，不包含重复选参的波动。有限预算只保证已搜索候选中的最优。",
        "同架构对照仍允许不同 CV 训练轮数，且固定模型有效参数较少；其差距不是仅改变一个变量的因果估计。NLL 与 grid SSE 衡量不同目标，不能保证 NLL 获胜者在每个 setting 上都有更小 SSE。",
        "最终数据 seeds 与上一轮相同，便于配对比较；由于选择 NLL 的动机来自此前测试结果，这些数据不能视为全新的最终外部验证。当前实验内部选参仍完全不访问测试结果。",
        f"完整审计通过：{audit['independent_nll_studies']} 个独立 NLL 搜索、{audit['cv_fold_fits']} 次 CV 拟合、{audit['final_fits']} 次最终拟合；重生成 {audit['regenerated_train_test_pairs']} 组 train–test 数据，核对逐样本 SSE 归档与前三条曲线误差，重算每组均值/SD，并验证实际 Optuna 目标、逐折原时间支持域上界和 NLL checkpoint。"]
    (output/"grid_sse_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return overall,family,paired


def plot_nll_cv(output_dir):
    output = Path(output_dir)
    paired = paired_sse(output).set_index("setting")
    summary = pd.read_csv(output/"summary.csv").pivot(index="setting",columns="method",values="average_grid_sse_mean").loc[paired.index]
    ratios = np.column_stack([summary[SEARCH_METHODS[1]]/summary[reference]
                             for reference in (SEARCH_METHODS[0],MATCHED)])
    figure,ax = plt.subplots(figsize=(10,12),constrained_layout=True)
    im = ax.imshow(np.log2(ratios),cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
    ax.set_yticks(range(len(summary)),summary.index,fontsize=8)
    ax.set_xticks([0,1],["Adaptive / tuned fixed\nIndependent architecture searches",
        "Adaptive / matched fixed\nSame adaptive-selected architecture"],fontsize=9)
    manifest = json.loads((output/"config.json").read_text(encoding="utf-8"))
    ax.set_title(f"Raw-time finite-support KL: grid SSE ratio\nMean over {len(manifest['experiment']['seeds'])} train/test pairs")
    for i in range(len(summary)):
        for j in range(2):
            ax.text(j,i,f"{ratios[i,j]:.2f}",ha="center",va="center",fontsize=8,
                color="white" if abs(np.log2(ratios[i,j])) > 1 else "black")
    figure.colorbar(im,ax=ax,shrink=.6,label="log2 ratio (blue: adaptive has lower SSE)")
    (output/"figures").mkdir(exist_ok=True)
    figure.savefig(output/"figures"/"rho_comparison_grid_sse.png",dpi=180)
    return figure
