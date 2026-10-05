# файл: src/s05_select_k.py
"""Шаг 5: подбор числа кластеров (силуэт, CH, DB, BIC/AIC, бутстреп-стабильность)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, load_config, load_csv, save_csv, save_json, setup_logging,
)

log = logging.getLogger("s05_select_k")
warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def _gmm_cov_type(X: np.ndarray, k: int, rs: int) -> tuple[str, float, float, object]:
    """Подбирает тип ковариации по BIC: full/tied/diag."""
    best = None
    for cov in ("full", "tied", "diag"):
        try:
            g = GaussianMixture(n_components=k, covariance_type=cov,
                                n_init=5, random_state=rs)
            g.fit(X)
            bic = g.bic(X)
            if best is None or bic < best[1]:
                best = (cov, bic, g.aic(X), g)
        except Exception:
            continue
    if best is None:
        raise RuntimeError("GMM не удалось обучить")
    return best[0], best[1], best[2], best[3]


def _bootstrap_ari(X: np.ndarray, labels: np.ndarray, method: str,
                   k: int, n_boot: int, rs: int) -> tuple[float, float]:
    """Стабильность: две случайные подвыборки 80%, обучение, ARI на пересечении."""
    rng = np.random.default_rng(rs)
    n = X.shape[0]
    if n < 10:
        return float("nan"), float("nan")
    aris = []
    for _ in range(n_boot):
        idx1 = rng.choice(n, size=int(0.8 * n), replace=False)
        idx2 = rng.choice(n, size=int(0.8 * n), replace=False)
        inter = np.intersect1d(idx1, idx2)
        if len(inter) < 5:
            continue
        try:
            if method == "kmeans":
                m1 = KMeans(n_clusters=k, n_init=50, random_state=rs).fit(X[idx1])
                m2 = KMeans(n_clusters=k, n_init=50, random_state=rs).fit(X[idx2])
                l1 = m1.labels_
                l2 = m2.labels_
            elif method == "ward":
                m1 = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X[idx1])
                m2 = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X[idx2])
                l1 = m1.labels_
                l2 = m2.labels_
            else:  # gmm
                m1 = GaussianMixture(n_components=k, n_init=5, random_state=rs).fit(X[idx1])
                m2 = GaussianMixture(n_components=k, n_init=5, random_state=rs).fit(X[idx2])
                l1 = m1.predict(X[idx1])
                l2 = m2.predict(X[idx2])
            pos1 = {v: i for i, v in enumerate(idx1)}
            pos2 = {v: i for i, v in enumerate(idx2)}
            a = np.array([l1[pos1[v]] for v in inter])
            b = np.array([l2[pos2[v]] for v in inter])
            aris.append(adjusted_rand_score(a, b))
        except Exception:
            continue
    if not aris:
        return float("nan"), float("nan")
    return float(np.mean(aris)), float(np.std(aris))


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    fm = load_csv(dirs["cache"] / "features_model.csv", index_col="mo")
    X = fm.to_numpy(dtype=float)
    n = X.shape[0]
    log.info("features_model: %s", fm.shape)

    k_min, k_max = cfg["k_range"]
    n_boot = int(cfg["n_bootstrap"])
    rs = int(cfg["random_state"])

    rows = []
    for k in tqdm(range(k_min, k_max + 1), desc="k"):
        for method in ["kmeans", "ward", "gmm"]:
            row = {"k": k, "method": method}
            try:
                if method == "kmeans":
                    model = KMeans(n_clusters=k, n_init=50, random_state=rs).fit(X)
                    labels = model.labels_
                elif method == "ward":
                    model = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
                    labels = model.labels_
                else:
                    cov, bic, aic, model = _gmm_cov_type(X, k, rs)
                    labels = model.predict(X)
                    row["gmm_cov"] = cov
                    row["BIC"] = float(bic)
                    row["AIC"] = float(aic)

                # Метрики (для k=1 silhouette/CH/DB не считаем — пропускаем)
                if len(np.unique(labels)) >= 2:
                    row["silhouette"] = float(silhouette_score(X, labels))
                    row["CH"] = float(calinski_harabasz_score(X, labels))
                    row["DB"] = float(davies_bouldin_score(X, labels))
                else:
                    row["silhouette"] = np.nan
                    row["CH"] = np.nan
                    row["DB"] = np.nan

                counts = np.bincount(labels)
                row["min_share"] = float(counts.min() / counts.sum())

                ari_m, ari_s = _bootstrap_ari(X, labels, method, k, n_boot, rs)
                row["ari_mean"] = ari_m
                row["ari_std"] = ari_s
            except Exception as e:
                log.warning("Ошибка %s k=%d: %s", method, k, e)
                row["silhouette"] = np.nan
                row["CH"] = np.nan
                row["DB"] = np.nan
                row["min_share"] = np.nan
                row["ari_mean"] = np.nan
                row["ari_std"] = np.nan
            rows.append(row)

    res = pd.DataFrame(rows)
    save_csv(res, dirs["cache"] / "k_selection.csv", index_name="row")
    log.info("k_selection: %s", res.shape)

    # Выбор k
    if cfg["k"] == "auto":
        valid = res.dropna(subset=["silhouette", "CH", "DB", "ari_mean"]).copy()
        valid = valid[valid["min_share"] >= float(cfg["min_cluster_share"])]
        if valid.empty:
            log.warning("Нет валидных k по min_cluster_share. Берём лучшее по силуэту.")
            valid = res.dropna(subset=["silhouette"]).copy()

        # Ранги
        valid["rank_sil"] = valid["silhouette"].rank(ascending=False, method="average")
        valid["rank_ch"] = valid["CH"].rank(ascending=False, method="average")
        valid["rank_db"] = valid["DB"].rank(ascending=True, method="average")
        valid["rank_ari"] = valid["ari_mean"].rank(ascending=False, method="average")
        valid["rank_mean"] = valid[["rank_sil", "rank_ch", "rank_db", "rank_ari"]].mean(axis=1)

        # Лучший метод+ k по среднему рангу
        best_row = valid.sort_values("rank_mean").iloc[0]
        chosen_k = int(best_row["k"])
        chosen_method = str(best_row["method"])
        save_csv(valid.sort_values("rank_mean"), dirs["tables"] / "k_ranking.csv", index_name="row")
        save_json({"k": chosen_k, "method": chosen_method,
                   "rank_mean": float(best_row["rank_mean"])},
                  dirs["cache"] / "k_chosen.json")
        log.info("Выбрано k=%d (метод=%s, rank_mean=%.2f)",
                 chosen_k, chosen_method, best_row["rank_mean"])
    else:
        chosen_k = int(cfg["k"])
        save_json({"k": chosen_k, "method": "kmeans"}, dirs["cache"] / "k_chosen.json")
        log.info("k задано вручную: %d", chosen_k)


if __name__ == "__main__":
    main()