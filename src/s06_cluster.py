# файл: src/s06_cluster.py
"""Шаг 6: кластеризация KMeans/Ward/GMM + согласование меток + ARI + пограничные."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, load_config, load_csv, load_json, save_csv, setup_logging,
)

log = logging.getLogger("s06_cluster")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def _align_labels(labels: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Согласование меток через венгерский алгоритм по матрице сопряжённости."""
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(labels, reference):
        cm[a, b] += 1
    # хотим максимизировать сумму cm[perm[i], i]
    row, col = linear_sum_assignment(-cm)
    mapping = {r: c for r, c in zip(row, col)}
    return np.array([mapping.get(x, x) for x in labels])


def _kmeans_order_by_A1(labels: np.ndarray, A1: np.ndarray, k: int) -> np.ndarray:
    """Нумерация кластеров по среднему A1 (0 — самый низкий уровень)."""
    means = np.array([A1[labels == c].mean() if (labels == c).any() else np.inf
                      for c in range(k)])
    order = np.argsort(means)
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    fm = load_csv(dirs["cache"] / "features_model.csv", index_col="mo")
    feats_raw = load_csv(dirs["cache"] / "features_raw.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    k_info = load_json(dirs["cache"] / "k_chosen.json")
    k = int(k_info["k"])
    rs = int(cfg["random_state"])
    log.info("Кластеризация: k=%d", k)

    X = fm.to_numpy(dtype=float)
    A1 = feats_raw.reindex(fm.index)["A1"].to_numpy()

    # KMeans
    km = KMeans(n_clusters=k, n_init=50, random_state=rs).fit(X)
    km_labels = _kmeans_order_by_A1(km.labels_, A1, k)

    # Ward
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    ward_labels = _align_labels(ward.labels_, km_labels, k)

    # GMM (full/tied/diag по BIC)
    best = None
    for cov in ("full", "tied", "diag"):
        try:
            g = GaussianMixture(n_components=k, covariance_type=cov,
                                n_init=5, random_state=rs).fit(X)
            bic = g.bic(X)
            if best is None or bic < best[0]:
                best = (bic, g, cov)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GMM не обучен")
    _, gmm, cov_used = best
    gmm_labels = _align_labels(gmm.predict(X), km_labels, k)
    log.info("GMM cov=%s, BIC=%.2f", cov_used, best[0])

    clusters = pd.DataFrame({
        "kmeans": km_labels,
        "ward": ward_labels,
        "gmm": gmm_labels,
    }, index=fm.index)

    # Малые МО — ближайший центр
    if bool(cfg["exclude_small"]) and (dirs["cache"] / "features_model_small.csv").exists():
        sm = load_csv(dirs["cache"] / "features_model_small.csv", index_col="mo")
        Xs = sm.to_numpy(dtype=float)
        # ближайший центр KMeans (в исходном порядке km.cluster_centers_)
        # но km_labels переупорядочены -> строим соответствие
        centers = km.cluster_centers_  # индексы 0..k-1 в исходной нумерации KMeans
        # нумерация km была переставлена; восстановим mapping исходная->новая
        # Для присвоения используем просто argmin и потом map
        raw_means = np.array([A1[km.labels_ == c].mean() if (km.labels_ == c).any() else np.inf
                              for c in range(k)])
        order = np.argsort(raw_means)
        old2new = {old: new for new, old in enumerate(order)}
        d = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        raw_assign = d.argmin(axis=1)
        assigned = np.array([old2new[int(x)] for x in raw_assign])
        clusters.loc[sm.index, "assigned_small"] = assigned
        clusters["assigned_small"] = clusters["assigned_small"].astype("Int64")

    # ARI и сопряжённость
    ari_kw = adjusted_rand_score(km_labels, ward_labels)
    ari_kg = adjusted_rand_score(km_labels, gmm_labels)
    ari_wg = adjusted_rand_score(ward_labels, gmm_labels)
    log.info("ARI kmeans-ward=%.3f, kmeans-gmm=%.3f, ward-gmm=%.3f", ari_kw, ari_kg, ari_wg)
    ari_df = pd.DataFrame(
        [[1.0, ari_kw, ari_kg], [ari_kw, 1.0, ari_wg], [ari_kg, ari_wg, 1.0]],
        index=["kmeans", "ward", "gmm"], columns=["kmeans", "ward", "gmm"],
    )
    save_csv(ari_df, dirs["tables"] / "concordance_ari.csv", index_name="method")

    ct_kw = pd.crosstab(clusters["kmeans"], clusters["ward"])
    ct_kg = pd.crosstab(clusters["kmeans"], clusters["gmm"])
    save_csv(ct_kw, dirs["tables"] / "concordance_kmeans_ward.csv", index_name="kmeans")
    save_csv(ct_kg, dirs["tables"] / "concordance_kmeans_gmm.csv", index_name="kmeans")

    # Силуэт и расстояние до центра для KMeans
    sil = silhouette_samples(X, km_labels)
    centers_ordered = np.zeros((k, X.shape[1]))
    for c in range(k):
        centers_ordered[c] = X[km_labels == c].mean(axis=0)
    dist = np.linalg.norm(X - centers_ordered[km_labels], axis=1)

    clusters["sil_kmeans"] = sil
    clusters["dist_kmeans"] = dist
    save_csv(clusters, dirs["cache"] / "clusters.csv", index_name="mo")

    centers_df = pd.DataFrame(centers_ordered, columns=fm.columns,
                              index=[f"cluster_{i}" for i in range(k)])
    save_csv(centers_df, dirs["cache"] / "centers_kmeans.csv", index_name="cluster")

    border = clusters[clusters["sil_kmeans"] < 0]
    save_csv(border, dirs["tables"] / "borderline_mo.csv", index_name="mo")
    log.info("Пограничных МО (sil<0): %d", len(border))


if __name__ == "__main__":
    main()