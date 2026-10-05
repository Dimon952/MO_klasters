# файл: src/s09_robust.py
"""Шаг 9: робастность — сценарии и ARI относительно базового KMeans."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_selected_features, load_config, load_csv, load_json,
    save_csv, save_fig, setup_logging,
)
from s02_features import build_r_matrix, compute_features, fit_regressions  # noqa: E402
from s04_preprocess import apply_preprocess, fit_preprocess  # noqa: E402

log = logging.getLogger("s09_robust")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def _kmeans_labels(X: np.ndarray, k: int, rs: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=50, random_state=rs).fit(X).labels_


def _build_features_pipeline(y: pd.DataFrame, cfg: dict,
                             drop_blocks: list[str] | None = None,
                             use_raw_log: bool | None = None) -> pd.DataFrame:
    """Восстанавливает features_model с учётом сценария."""
    cfg_local = dict(cfg)
    if use_raw_log is not None:
        cfg_local["use_raw_log"] = use_raw_log
    if drop_blocks:
        new_blocks = {b: fs for b, fs in cfg["blocks"].items() if b not in drop_blocks}
        cfg_local["blocks"] = new_blocks
    r = build_r_matrix(y, bool(cfg_local["use_raw_log"]))
    params, residuals = fit_regressions(r)
    months_int = np.array([int(c.split("-")[1]) for c in r.columns])
    feats, _ = compute_features(params, residuals, r, months_int)
    sel = get_selected_features(cfg_local)
    sel = [f for f in sel if f in feats.columns]
    p = fit_preprocess(feats, cfg_local, sel)
    fm = apply_preprocess(feats, p)
    return fm


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    y = load_csv(dirs["cache"] / "y_matrix.csv", index_col="mo")
    fm = load_csv(dirs["cache"] / "features_model.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    k_info = load_json(dirs["cache"] / "k_chosen.json")
    k = int(k_info["k"])
    rs = int(cfg["random_state"])

    X_base = fm.to_numpy(dtype=float)
    base_labels = _kmeans_labels(X_base, k, rs)
    base_series = pd.Series(base_labels, index=fm.index, name="base")

    rows = []
    changed_counter: dict[str, int] = {mo: 0 for mo in fm.index}

    def _add(scenario: str, labels: np.ndarray, index) -> None:
        ser = pd.Series(labels, index=index)
        common = base_series.index.intersection(ser.index)
        ari = adjusted_rand_score(base_series.loc[common], ser.loc[common])
        rows.append({"scenario": scenario, "ari_vs_base": float(ari), "n": len(common)})
        # счётчик смен
        for mo in common:
            if base_series[mo] != ser[mo]:
                changed_counter[mo] = changed_counter.get(mo, 0) + 1

    # Сценарий 1: use_raw_log=true
    try:
        fm1 = _build_features_pipeline(y, cfg, drop_blocks=None, use_raw_log=True)
        X1 = fm1.to_numpy(dtype=float)
        _add("use_raw_log", _kmeans_labels(X1, k, rs), fm1.index)
    except Exception as e:
        log.warning("Сценарий use_raw_log упал: %s", e)

    # Сценарий 2: без блока D
    try:
        fm2 = _build_features_pipeline(y, cfg, drop_blocks=["D"])
        X2 = fm2.to_numpy(dtype=float)
        _add("no_block_D", _kmeans_labels(X2, k, rs), fm2.index)
    except Exception as e:
        log.warning("Сценарий no_block_D упал: %s", e)

    # Сценарий 3: без блока C
    try:
        fm3 = _build_features_pipeline(y, cfg, drop_blocks=["C"])
        X3 = fm3.to_numpy(dtype=float)
        _add("no_block_C", _kmeans_labels(X3, k, rs), fm3.index)
    except Exception as e:
        log.warning("Сценарий no_block_C упал: %s", e)

    # Сценарий 4: исключение малых
    if "small" in meta.columns and meta["small"].any():
        big_mask = ~meta.loc[fm.index, "small"].astype(bool).to_numpy()
        if big_mask.sum() >= k + 1:
            Xb = X_base[big_mask]
            kb = _kmeans_labels(Xb, k, rs)
            # малым — ближайший центр
            centers = np.zeros((k, Xb.shape[1]))
            for c in range(k):
                centers[c] = Xb[kb == c].mean(axis=0)
            Xs = X_base[~big_mask]
            ds = ((Xs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            small_assign = ds.argmin(axis=1)
            full_labels = np.zeros(X_base.shape[0], dtype=int)
            full_labels[big_mask] = kb
            full_labels[~big_mask] = small_assign
            _add("exclude_small", full_labels, fm.index)

    # Сценарий 5: Ward и GMM относительно KMeans
    try:
        ward_l = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_base).labels_
        _add("ward", ward_l, fm.index)
    except Exception as e:
        log.warning("Ward упал: %s", e)
    try:
        best = None
        for cov in ("full", "tied", "diag"):
            try:
                g = GaussianMixture(n_components=k, covariance_type=cov,
                                    n_init=5, random_state=rs).fit(X_base)
                bic = g.bic(X_base)
                if best is None or bic < best[0]:
                    best = (bic, g)
            except Exception:
                continue
        if best is not None:
            _add("gmm", best[1].predict(X_base), fm.index)
    except Exception as e:
        log.warning("GMM упал: %s", e)

    # Сценарий 6: 10 случайных подвыборок 80%
    rng = np.random.default_rng(rs)
    n = X_base.shape[0]
    sub_aris = []
    for b in range(10):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        try:
            sub_l = _kmeans_labels(X_base[idx], k, rs)
            sub_series = pd.Series(sub_l, index=fm.index[idx])
            common = base_series.index.intersection(sub_series.index)
            sub_aris.append(adjusted_rand_score(base_series.loc[common], sub_series.loc[common]))
            for mo in common:
                if base_series[mo] != sub_series[mo]:
                    changed_counter[mo] = changed_counter.get(mo, 0) + 1
        except Exception:
            continue
    if sub_aris:
        rows.append({"scenario": "subsample_80_mean", "ari_vs_base": float(np.mean(sub_aris)),
                     "n": n})

    rob = pd.DataFrame(rows)
    save_csv(rob, dirs["tables"] / "robustness.csv", index_name="row")

    # Столбчатая диаграмма ARI
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(rob["scenario"], rob["ari_vs_base"], color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ARI vs base KMeans")
    ax.set_title("Робастность кластеризации")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save_fig(fig, "s09_robustness_ari", cfg)

    # Топ МО, чаще всего меняющих кластер
    change_df = pd.DataFrame(list(changed_counter.items()), columns=["mo", "n_changes"])
    change_df = change_df.sort_values("n_changes", ascending=False).head(50)
    save_csv(change_df, dirs["tables"] / "unstable_mo.csv", index_name="row")

    log.info("s09_robust завершён")


if __name__ == "__main__":
    main()