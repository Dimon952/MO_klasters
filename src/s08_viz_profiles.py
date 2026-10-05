# файл: src/s08_viz_profiles.py
"""Шаг 8b: визуализация профилей — тепловая карта z, радары, боксплоты, сезонные профили, состав, pairplot."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_block_map, get_palette, load_config, load_csv,
    save_fig, setup_logging,
)

log = logging.getLogger("s08_viz_profiles")

try:
    import geopandas as gpd  # noqa: F401
    HAS_GEO = True
except Exception:
    HAS_GEO = False


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--map", default=None, help="файл границ для картограммы")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    feats = load_csv(dirs["cache"] / "features_raw.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    clusters = load_csv(dirs["cache"] / "clusters.csv", index_col="mo")
    z_df = load_csv(dirs["tables"] / "profile_z.csv", index_col="cluster")
    cs = load_csv(dirs["cache"] / "cluster_series.csv", index_col=None)

    labels = clusters["kmeans"].astype(int)
    k = int(labels.nunique())
    palette = get_palette(k)
    block_map = get_block_map(cfg)
    feats = feats.loc[labels.index]
    meta = meta.loc[labels.index]

    # 1. Тепловая карта z-оценок
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(z_df.columns)), 0.6 * k + 2))
    sns.heatmap(z_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-3, vmax=3, ax=ax, cbar_kws={"label": "z"})
    # разделители блоков
    cols = list(z_df.columns)
    block_order = []
    for f in cols:
        b = block_map.get(f, "?")
        if not block_order or block_order[-1] != b:
            block_order.append(b)
    acc = 0
    for i in range(1, len(cols)):
        if block_map.get(cols[i]) != block_map.get(cols[i - 1]):
            ax.axvline(i, color="black", lw=1)
    ax.set_title("Профиль кластеров (z-оценки)")
    fig.tight_layout()
    save_fig(fig, "s08_profiles_heatmap", cfg)

    # 2. Радарные диаграммы
    ncols = min(k, 4)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).flatten()
    angles = np.linspace(0, 2 * np.pi, len(z_df.columns), endpoint=False).tolist()
    angles += angles[:1]
    for c in range(k):
        if c >= len(axes):
            break
        ax = axes[c]
        vals = z_df.loc[c].clip(-3, 3).tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=palette[c], lw=2)
        ax.fill(angles, vals, color=palette[c], alpha=0.3)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(z_df.columns, fontsize=8)
        ax.set_title(f"Кластер {c}", pad=15)
    for j in range(k, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    save_fig(fig, "s08_profiles_radar", cfg)

    # 3. Боксплоты по кластерам
    long = []
    for f in feats.columns:
        for mo in feats.index:
            long.append({"feature": f, "cluster": int(labels[mo]), "value": feats.loc[mo, f]})
    ld = pd.DataFrame(long)
    ncols = 4
    nrows = int(np.ceil(len(feats.columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for i, f in enumerate(feats.columns):
        sub = ld[ld["feature"] == f]
        sns.boxplot(data=sub, x="cluster", y="value", ax=axes[i], palette=palette)
        axes[i].set_title(f)
    for j in range(len(feats.columns), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    save_fig(fig, "s08_profiles_boxplots", cfg)

    # 4. Сезонные профили и траектории
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # seasonal
    sub_s = cs[cs["series"] == "seasonal"]
    for c in range(k):
        s = sub_s[sub_s["cluster"] == c]
        x = np.arange(1, 13)
        axes[0].plot(x, s["median"].to_numpy(), color=palette[c], label=f"c{c}")
        axes[0].fill_between(x, s["q25"].to_numpy(), s["q75"].to_numpy(),
                             color=palette[c], alpha=0.15)
    axes[0].set_title("Сезонный профиль s(m)")
    axes[0].set_xticks(np.arange(1, 13))
    axes[0].legend(fontsize=8)
    # r(t)
    sub_r = cs[cs["series"] == "r"]
    for c in range(k):
        s = sub_r[sub_r["cluster"] == c]
        axes[1].plot(range(len(s)), s["median"].to_numpy(), color=palette[c])
    axes[1].set_title("Относительный ряд r(t)")
    # y(t)
    sub_y = cs[cs["series"] == "y"]
    for c in range(k):
        s = sub_y[sub_y["cluster"] == c]
        axes[2].plot(range(len(s)), s["median"].to_numpy(), color=palette[c])
    axes[2].set_title("Уровень y(t) (медиана)")
    fig.tight_layout()
    save_fig(fig, "s08_profiles_series", cfg)

    # 5. Состав кластеров
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, ["federal_district", "mo_type", "small"]):
        ct = pd.crosstab(labels, meta[col].fillna("?").astype(str), normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.set_title(f"Состав кластеров: {col}")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    save_fig(fig, "s08_profiles_composition", cfg)

    # 6. Pairplot
    pf = [f for f in cfg["pairplot_features"] if f in feats.columns]
    if len(pf) >= 2:
        pp_df = feats[pf].copy()
        pp_df["cluster"] = labels.astype(str)
        g = sns.pairplot(pp_df, hue="cluster", diag_kind="kde",
                         plot_kws={"s": 8, "alpha": 0.5}, palette=palette)
        g.fig.suptitle("Pairplot признаков", y=1.02)
        g.savefig(Path(cfg["paths"]["out_dir"]) / "figures" / "s08_pairplot.png",
                  dpi=200, bbox_inches="tight")
        g.savefig(Path(cfg["paths"]["out_dir"]) / "figures" / "s08_pairplot.svg",
                  bbox_inches="tight")
        plt.close("all")

    # 7. Картограмма (опционально)
    if args.map:
        if not HAS_GEO:
            log.warning("geopandas недоступен — картограмма пропущена")
        else:
            try:
                gdf = gpd.read_file(args.map)
                # предполагаем, что в gdf есть колонка 'mo'
                merged = gdf.merge(labels.rename("cluster"), left_on="mo", right_index=True, how="left")
                fig, ax = plt.subplots(figsize=(12, 8))
                merged.plot(column="cluster", categorical=True, legend=True,
                            cmap="tab20", ax=ax, missing_kwds={"color": "lightgrey"})
                ax.set_title("Картограмма кластеров")
                save_fig(fig, "s08_map_clusters", cfg)
            except Exception as e:
                log.warning("Ошибка картограммы: %s", e)
    else:
        log.info("Картограмма пропущена: --map не задан")

    # 8. Сборка report.html
    out_dir = Path(cfg["paths"]["out_dir"])
    desc_md = (out_dir / "cluster_descriptions.md").read_text(encoding="utf-8") \
        if (out_dir / "cluster_descriptions.md").exists() else ""
    profile_html = pd.read_csv(dirs["tables"] / "profile.csv", encoding="utf-8-sig").to_html(
        index=False, border=1, classes="table")
    figs = sorted([p.name for p in (out_dir / "figures").glob("s08_*.png")])
    fig_html = "\n".join([f'<div><img src="figures/{n}" style="max-width:100%"/></div>' for n in figs])
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт кластеризации МО</title>
<style>body{{font-family:Arial,sans-serif;margin:20px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px}}
img{{margin:8px 0}}</style></head><body>
<h1>Отчёт: кластеризация МО по потребительскому поведению</h1>
<h2>Профиль кластеров</h2>
{profile_html}
<h2>Описания кластеров</h2>
<pre>{desc_md}</pre>
<h2>Графики</h2>
{fig_html}
</body></html>"""
    (out_dir / "report.html").write_text(html, encoding="utf-8")
    log.info("report.html собран: %s", out_dir / "report.html")

    log.info("s08_viz_profiles завершён")


if __name__ == "__main__":
    main()