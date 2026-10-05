# файл: src/s04_preprocess.py
"""Шаг 4: предобработка — выбор признаков, winsorize, робастная стандартизация, блочный вес."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_block_map, get_selected_features, load_config,
    load_csv, save_csv, save_json, setup_logging,
)

log = logging.getLogger("s04_preprocess")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def fit_preprocess(feats: pd.DataFrame, cfg: dict, feature_list: list[str]) -> dict:
    """Обучает параметры преобразования на переданной подвыборке."""
    p_lo, p_hi = cfg["winsor_pct"]
    params = {"features": feature_list, "winsor": {}, "robust": {}, "block_weights": {}}
    for c in feature_list:
        col = feats[c].astype(float)
        lo, hi = np.percentile(col, [p_lo, p_hi])
        params["winsor"][c] = [float(lo), float(hi)]
        clipped = col.clip(lo, hi)
        med = float(clipped.median())
        iqr = float(clipped.quantile(0.75) - clipped.quantile(0.25))
        if iqr == 0:
            iqr = 1.0
        params["robust"][c] = [med, iqr]
    # Блочные веса
    block_map = get_block_map(cfg)
    block_counts: dict[str, int] = {}
    for c in feature_list:
        b = block_map.get(c, "?")
        block_counts[b] = block_counts.get(b, 0) + 1
    for b, k in block_counts.items():
        w = 1.0 / np.sqrt(k) if k > 0 else 1.0
        for c in feature_list:
            if block_map.get(c) == b:
                params["block_weights"][c] = float(w)
    return params


def apply_preprocess(feats: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Применяет параметры к данным."""
    out = pd.DataFrame(index=feats.index)
    for c in params["features"]:
        lo, hi = params["winsor"][c]
        med, iqr = params["robust"][c]
        w = params["block_weights"][c]
        col = feats[c].astype(float).clip(lo, hi)
        out[c] = (col - med) / iqr * w
    return out


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    feats = load_csv(dirs["cache"] / "features_raw.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    feats = feats.loc[meta.index]

    sel = get_selected_features(cfg)
    sel = [f for f in sel if f in feats.columns]
    log.info("Отобрано признаков: %d (%s)", len(sel), sel)

    exclude_small = bool(cfg["exclude_small"])
    if exclude_small:
        big_mask = ~meta["small"].astype(bool)
        feats_big = feats.loc[big_mask]
        feats_small = feats.loc[~big_mask]
        params = fit_preprocess(feats_big, cfg, sel)
        fm_big = apply_preprocess(feats_big, params)
        fm_small = apply_preprocess(feats_small, params)
        save_csv(fm_big, dirs["cache"] / "features_model.csv", index_name="mo")
        save_csv(fm_small, dirs["cache"] / "features_model_small.csv", index_name="mo")
    else:
        params = fit_preprocess(feats, cfg, sel)
        fm = apply_preprocess(feats, params)
        save_csv(fm, dirs["cache"] / "features_model.csv", index_name="mo")

    save_json(params, dirs["cache"] / "preprocess_params.json")
    log.info("features_model: %s", fm.shape)
    log.info("s04_preprocess завершён")


if __name__ == "__main__":
    main()