# файл: src/s02_features.py
"""Шаг 2: расчёт 11 признаков (A1,A2,B1,B2,C1..C5,D1,D2) по каждому МО."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, load_config, load_csv, save_csv, setup_logging, month_columns,
)

log = logging.getLogger("s02_features")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def build_r_matrix(y: pd.DataFrame, use_raw_log: bool) -> pd.DataFrame:
    """Считает относительный ряд r(i,t) = ln y − медиана по всем МО от ln y (или просто ln y)."""
    ly = np.log(y.astype(float))
    if use_raw_log:
        return ly
    # медиана по столбцам (по МО в каждый месяц)
    med = ly.median(axis=0)
    return ly.sub(med, axis=1)


def fit_regressions(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Одна lstsq для всех МО. Возвращает (params 13 x N_MO, residuals)."""
    n_months = r.shape[1]
    months = np.array([int(c.split("-")[1]) for c in r.columns])
    t = np.arange(n_months, dtype=float) - (n_months - 1) / 2.0  # центрировано

    # Дизайн-матрица: intercept + t + 11 dummy (месяц m=1..11: [m=m] - [m=12])
    n = n_months
    X = np.zeros((n, 13), dtype=float)
    X[:, 0] = 1.0
    X[:, 1] = t
    for m in range(1, 12):
        X[:, 1 + m] = (months == m).astype(float) - (months == 12).astype(float)

    Y = r.to_numpy(dtype=float).T  # n_months x N_MO

    # lstsq: params x N_MO
    params, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ params
    return params, residuals


def compute_features(params: np.ndarray, residuals: np.ndarray, r: pd.DataFrame,
                     months_int: np.ndarray) -> pd.DataFrame:
    """Вычисляет 11 признаков. params: 13 x N_MO, residuals: n_months x N_MO."""
    n_months = r.shape[1]
    # gamma_1..11 = params[2..12]; gamma_12 = -sum(gamma_1..11)
    g = params[2:13, :]  # 11 x N
    g12 = -g.sum(axis=0, keepdims=True)
    s = np.vstack([g, g12])  # 12 x N, s[m-1] соответствует месяцу m

    A1 = r.mean(axis=1).to_numpy()
    A2 = r.iloc[:, -12:].mean(axis=1).to_numpy()
    B1 = params[1, :]

    # B2: среднее r(t) - r(t-12) по t=13..24
    Yv = r.to_numpy(dtype=float)
    B2 = (Yv[:, 12:24] - Yv[:, 0:12]).mean(axis=1)

    C1 = s.max(axis=0) - s.min(axis=0)
    m_idx = np.arange(1, 13, dtype=float)
    C2 = (2.0 / 12.0) * (s * np.cos(2.0 * np.pi * (m_idx[:, None] - 1) / 12.0)).sum(axis=0)
    C3 = (2.0 / 12.0) * (s * np.sin(2.0 * np.pi * (m_idx[:, None] - 1) / 12.0)).sum(axis=0)
    summer_mask = np.isin(m_idx, [6, 7, 8])
    C4 = s[summer_mask, :].mean(axis=0) - s[~summer_mask, :].mean(axis=0)
    C5 = s[11, :]

    df_resid = 11
    SSE = (residuals ** 2).sum(axis=0)
    D1 = np.sqrt(SSE / df_resid)

    # D2: доля месяцев, где |Δε| > 3 * 1.4826 * MAD(ε)
    de = np.abs(np.diff(residuals, axis=0))  # (n-1) x N
    mad = np.median(np.abs(residuals - np.median(residuals, axis=0, keepdims=True)), axis=0)
    thresh = 3.0 * 1.4826 * mad
    with np.errstate(divide="ignore", invalid="ignore"):
        # для каждого МО: доля |Δε| > threshold (по столбцам)
        cnt = (de > thresh[None, :]).sum(axis=0)
    D2 = np.where(mad > 0, cnt / de.shape[0], 0.0)

    feats = pd.DataFrame({
        "A1": A1, "A2": A2, "B1": B1, "B2": B2,
        "C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5,
        "D1": D1, "D2": D2,
    }, index=r.index)
    return feats, s


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    y = load_csv(dirs["cache"] / "y_matrix.csv", index_col="mo")
    months_cols = month_columns(y)
    y = y[months_cols]
    log.info("y_matrix: %s", y.shape)

    r = build_r_matrix(y, bool(cfg["use_raw_log"]))
    save_csv(r, dirs["cache"] / "r_matrix.csv", index_name="mo")

    params, residuals = fit_regressions(r)
    months_int = np.array([int(c.split("-")[1]) for c in r.columns])
    feats, s = compute_features(params, residuals, r, months_int)

    # seasonal_profile: s01..s12
    sp = pd.DataFrame(s.T, index=r.index, columns=[f"s{m:02d}" for m in range(1, 13)])
    save_csv(sp, dirs["cache"] / "seasonal_profile.csv", index_name="mo")
    save_csv(feats, dirs["cache"] / "features_raw.csv", index_name="mo")

    log.info("features_raw: %s", feats.shape)
    log.info("Описательная статистика признаков:\n%s", feats.describe().to_string())


if __name__ == "__main__":
    main()