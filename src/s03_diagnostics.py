# файл: src/s03_diagnostics.py
"""Шаг 3: диагностика признаков — статистика, корреляции, VIF, малые vs крупные."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_block_map, get_selected_features, load_config,
    load_csv, save_csv, save_fig, setup_logging,
)

log = logging.getLogger("s03_diagnostics")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def _vif(df: pd.DataFrame) -> pd.Series:
    """VIF = 1/(1-R^2) из регрессии каждого признака на остальные."""
    from sklearn.linear_model import LinearRegression
    X = df.to_numpy(dtype=float)
    vifs = []
    for j in range(X.shape[1]):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        lr = LinearRegression().fit(Xo, y)
        r2 = lr.score(Xo, y)
        vifs.append(1.0 / max(1.0 - r2, 1e-12))
    return pd.Series(vifs, index=df.columns)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    feats = load_csv(dirs["cache"] / "features_raw.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    feats = feats.loc[meta.index]

    # 1. Описательная статистика + количество вне перцентилей
    p_lo, p_hi = cfg["winsor_pct"]
    stats_rows = []
    for c in feats.columns:
        col = feats[c].dropna()
        lo, hi = np.percentile(col, [p_lo, p_hi])
        stats_rows.append({
            "feature": c,
            "mean": col.mean(),
            "std": col.std(),
            "min": col.min(),
            "q01": np.percentile(col, 1),
            "q25": np.percentile(col, 25),
            "median": col.median(),
            "q75": np.percentile(col, 75),
            "q99": np.percentile(col, 99),
            "max": col.max(),
            "n_below_p_lo": int((col < lo).sum()),
            "n_above_p_hi": int((col > hi).sum()),
        })
    stats_df = pd.DataFrame(stats_rows).set_index("feature")
    save_csv(stats_df, dirs["tables"] / "feature_stats.csv", index_name="feature")

    # Гистограммы + KDE
    n = len(feats.columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(feats.columns):
        sns.histplot(feats[c].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(c)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Распределения признаков (raw)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "s03_hist_features", cfg)

    # 2. Корреляции
    pear = feats.corr(method="pearson")
    spear = feats.corr(method="spearman")
    save_csv(pear, dirs["tables"] / "corr_pearson.csv", index_name="feature")
    save_csv(spear, dirs["tables"] / "corr_spearman.csv", index_name="feature")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    axes[0].set_title("Pearson")
    sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
    axes[1].set_title("Spearman")
    fig.tight_layout()
    save_fig(fig, "s03_corr_heatmaps", cfg)

    # Пары |corr| > 0.9
    priority = list(cfg["feature_priority"])
    prio_idx = {f: i for i, f in enumerate(priority)}
    pairs = []
    cols = list(feats.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = pear.iloc[i, j]
            if abs(c) > 0.9:
                a, b = cols[i], cols[j]
                ia, ib = prio_idx.get(a, 10**9), prio_idx.get(b, 10**9)
                drop = b if ia < ib else a
                pairs.append({"f1": a, "f2": b, "pearson": c,
                              "spearman": spear.loc[a, b],
                              "recommend_drop": drop})
    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        pairs_df = pd.DataFrame(columns=["f1", "f2", "pearson", "spearman", "recommend_drop"])
    save_csv(pairs_df, dirs["tables"] / "corr_pairs.csv", index_name="row")

    # Готовый обновлённый blocks
    if not pairs_df.empty:
        drop_set = set(pairs_df["recommend_drop"].tolist())
    else:
        drop_set = set()
    new_blocks = {}
    for block, fs in cfg["blocks"].items():
        new_blocks[block] = [f for f in fs if f not in drop_set]
    log.info("Рекомендуемый обновлённый blocks (можно вставить в конфиг): %s", new_blocks)

    # 3. VIF по выбранным признакам
    sel = get_selected_features(cfg)
    sel = [f for f in sel if f in feats.columns]
    if len(sel) >= 2:
        vif = _vif(feats[sel])
        vif_df = vif.to_frame("VIF")
        save_csv(vif_df, dirs["tables"] / "vif.csv", index_name="feature")
        high = vif[vif > 10]
        if not high.empty:
            log.warning("VIF > 10 у признаков: %s", high.to_dict())

    # 4. Малые vs крупные
    small_mask = meta["small"].astype(bool)
    big_mask = ~small_mask
    rows = []
    for c in feats.columns:
        a = feats.loc[small_mask, c].dropna()
        b = feats.loc[big_mask, c].dropna()
        if len(a) >= 3 and len(b) >= 3:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            # размер эффекта: r = Z / sqrt(N)
            n1, n2 = len(a), len(b)
            z = stats.norm.isf(p / 2.0) if p > 0 else 0.0
            r_eff = z / np.sqrt(n1 + n2)
            rows.append({
                "feature": c,
                "median_small": float(a.median()),
                "iqr_small": float(a.quantile(0.75) - a.quantile(0.25)),
                "median_large": float(b.median()),
                "iqr_large": float(b.quantile(0.75) - b.quantile(0.25)),
                "mannwhitney_p": float(p),
                "effect_r": float(r_eff),
                "n_small": n1, "n_large": n2,
            })
    mw_df = pd.DataFrame(rows)
    save_csv(mw_df, dirs["tables"] / "small_vs_large.csv", index_name="row")

    # Ящичные диаграммы
    feat_long = feats.copy()
    feat_long["group"] = np.where(small_mask, "small", "large")
    n = len(feats.columns)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, c in enumerate(feats.columns):
        sns.boxplot(data=feat_long, x="group", y=c, ax=axes[i])
        axes[i].set_title(c)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Малые vs крупные МО (ящичные диаграммы)", y=1.02)
    fig.tight_layout()
    save_fig(fig, "s03_small_vs_large", cfg)

    log.info("s03_diagnostics завершён")


if __name__ == "__main__":
    main()