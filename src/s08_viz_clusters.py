# файл: src/s08_viz_clusters.py
"""Шаг 8a: визуализация кластеров — k-панель, PCA, UMAP/t-SNE, дендрограмма, согласованность."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_palette, load_config, load_csv, load_json,
    save_csv, save_fig, setup_logging,
)

log = logging.getLogger("s08_viz_clusters")

try:
    import umap  # noqa: F401
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import plotly.express as px  # noqa: F401
    import plotly.io as pio
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--interactive", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    fm = load_csv(dirs["cache"] / "features_model.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    clusters = load_csv(dirs["cache"] / "clusters.csv", index_col="mo")
    ks = load_csv(dirs["cache"] / "k_selection.csv", index_col=None)
    k_info = load_json(dirs["cache"] / "k_chosen.json")
    k = int(k_info["k"])
    palette = get_palette(k)

    labels = clusters["kmeans"].astype(int)
    X = fm.to_numpy(dtype=float)

    # 1. Панель выбора k
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics_plot = [("silhouette", "Silhouette ↑"), ("CH", "Calinski-Harabasz ↑"),
                    ("DB", "Davies-Bouldin ↓"), ("ari_mean", "Bootstrap ARI ↑")]
    for ax, (m, title) in zip(axes.flatten(), metrics_plot):
        for method, color in zip(["kmeans", "ward", "gmm"], ["C0", "C1", "C2"]):
            sub = ks[ks["method"] == method].sort_values("k")
            if m in sub.columns:
                ax.plot(sub["k"], sub[m], marker="o", label=method, color=color)
        ax.axvline(k, color="k", ls="--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("k")
        ax.legend()
    fig.suptitle(f"Выбор k (выбрано k={k})")
    fig.tight_layout()
    save_fig(fig, "s08_k_selection_panel", cfg)

    # 2. PCA-biplot
    pca = PCA(n_components=2, random_state=int(cfg["random_state"]))
    coords = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(k):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.6,
                   color=palette[c], label=f"cluster {c}", rasterized=True)
    # Стрелки нагрузок
    loads = pca.components_.T
    scale = np.abs(coords).max() * 0.7 / max(np.abs(loads).max(), 1e-9)
    for i, f in enumerate(fm.columns):
        ax.arrow(0, 0, loads[i, 0] * scale, loads[i, 1] * scale,
                 color="gray", alpha=0.7, head_width=scale * 0.03)
        ax.text(loads[i, 0] * scale * 1.08, loads[i, 1] * scale * 1.08, f,
                color="black", fontsize=8, ha="center", va="center")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA-biplot")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "s08_pca_biplot", cfg)

    # Панель 2x2: cluster / federal_district / mo_type / log(pop)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # cluster
    for c in range(k):
        mask = labels == c
        axes[0, 0].scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                           color=palette[c], rasterized=True)
    axes[0, 0].set_title("Кластеры")
    # federal district
    fds = meta["federal_district"].fillna("?").astype(str).unique()
    fd_palette = get_palette(len(fds))
    for i, fd in enumerate(fds):
        mask = meta["federal_district"].fillna("?").astype(str).values == fd
        axes[0, 1].scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                           color=fd_palette[i], label=fd, rasterized=True)
    axes[0, 1].set_title("Федеральный округ")
    axes[0, 1].legend(fontsize=7, markerscale=2, loc="best")
    # mo_type
    mts = meta["mo_type"].fillna("?").astype(str).unique()
    mt_palette = get_palette(len(mts))
    for i, mt in enumerate(mts):
        mask = meta["mo_type"].fillna("?").astype(str).values == mt
        axes[1, 0].scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6,
                           color=mt_palette[i], label=mt, rasterized=True)
    axes[1, 0].set_title("Тип МО")
    axes[1, 0].legend(fontsize=7, markerscale=2, loc="best")
    # log pop
    logpop = np.log10(meta["pop_2023"].clip(lower=1).values)
    sc = axes[1, 1].scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6,
                            c=logpop, cmap="viridis", rasterized=True)
    axes[1, 1].set_title("log10(Население 2023)")
    fig.colorbar(sc, ax=axes[1, 1])
    fig.suptitle("PCA-проекции с разными раскрасками")
    fig.tight_layout()
    save_fig(fig, "s08_pca_panels", cfg)

    # 3. UMAP или t-SNE
    if HAS_UMAP:
        reducer = umap.UMAP(random_state=int(cfg["random_state"]), n_neighbors=15, min_dist=0.1)
        emb = reducer.fit_transform(X)
        emb_title = "UMAP"
    else:
        log.warning("umap-learn недоступен, использую t-SNE")
        emb = TSNE(n_components=2, random_state=int(cfg["random_state"]),
                   perplexity=min(30, max(5, X.shape[0] // 10))).fit_transform(X)
        emb_title = "t-SNE"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for c in range(k):
        mask = labels == c
        axes[0, 0].scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6,
                           color=palette[c], rasterized=True)
    axes[0, 0].set_title("Кластеры")
    for i, fd in enumerate(fds):
        mask = meta["federal_district"].fillna("?").astype(str).values == fd
        axes[0, 1].scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6,
                           color=fd_palette[i], label=fd, rasterized=True)
    axes[0, 1].set_title("Федеральный округ")
    axes[0, 1].legend(fontsize=7, markerscale=2)
    for i, mt in enumerate(mts):
        mask = meta["mo_type"].fillna("?").astype(str).values == mt
        axes[1, 0].scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.6,
                           color=mt_palette[i], label=mt, rasterized=True)
    axes[1, 0].set_title("Тип МО")
    axes[1, 0].legend(fontsize=7, markerscale=2)
    sc = axes[1, 1].scatter(emb[:, 0], emb[:, 1], s=10, alpha=0.6, c=logpop,
                            cmap="viridis", rasterized=True)
    axes[1, 1].set_title("log10(Население 2023)")
    fig.colorbar(sc, ax=axes[1, 1])
    fig.suptitle(f"{emb_title}-проекции")
    fig.tight_layout()
    save_fig(fig, f"s08_{emb_title.lower()}_panels", cfg)

    # 4. Дендрограмма Ward
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(12, 6))
    if X.shape[0] > 300:
        dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, no_labels=True)
    else:
        dendrogram(Z, ax=ax, no_labels=True)
    ax.axhline(y=Z[-(k - 1), 2] if k > 1 else 0, color="r", ls="--")
    ax.set_title("Дендрограмма Ward")
    fig.tight_layout()
    save_fig(fig, "s08_dendrogram_ward", cfg)

    # 5. Согласованность алгоритмов
    ct_kw = pd.crosstab(clusters["kmeans"], clusters["ward"])
    ct_kg = pd.crosstab(clusters["kmeans"], clusters["gmm"])
    ari_mat = load_csv(dirs["tables"] / "concordance_ari.csv", index_col="method")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(ct_kw, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("KMeans × Ward")
    sns.heatmap(ct_kg, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title("KMeans × GMM")
    sns.heatmap(ari_mat, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[2])
    axes[2].set_title("ARI между методами")
    fig.tight_layout()
    save_fig(fig, "s08_concordance", cfg)

    # Silhouette plot
    sil = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for c in range(k):
        vals = np.sort(sil[labels == c])
        size = len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_lower + size), 0, vals,
                         color=palette[c], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, str(c))
        y_lower += size + 10
    ax.axvline(sil.mean(), color="red", ls="--")
    ax.set_xlabel("Silhouette")
    ax.set_title(f"Silhouette plot (mean={sil.mean():.3f})")
    save_fig(fig, "s08_silhouette", cfg)

    # Interactive HTML
    if args.interactive and HAS_PLOTLY:
        df_plot = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "cluster": labels.astype(str),
            "federal_district": meta["federal_district"].fillna("?").astype(str).values,
            "mo_type": meta["mo_type"].fillna("?").astype(str).values,
            "pop": meta["pop_2023"].values,
            "mo": fm.index,
        })
        fig_px = px.scatter(df_plot, x="PC1", y="PC2", color="cluster",
                            hover_data=["mo", "federal_district", "mo_type", "pop"],
                            title="PCA-проекция (интерактивно)")
        out_html = Path(cfg["paths"]["out_dir"]) / "pca_interactive.html"
        fig_px.write_html(str(out_html))
        log.info("Интерактивный PCA: %s", out_html)
    elif args.interactive and not HAS_PLOTLY:
        log.warning("plotly не установлен — интерактив пропущен")

    log.info("s08_viz_clusters завершён")


if __name__ == "__main__":
    main()