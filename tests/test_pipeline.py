# файл: tests/test_pipeline.py
"""Тесты пайплайна: восстановление параметров, интерполяция, дубликаты, сквозной ARI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from s02_features import build_r_matrix, compute_features, fit_regressions  # noqa: E402


# ---------- (а) Восстановление известных параметров ----------
def _make_synthetic_matrix() -> pd.DataFrame:
    """24 месяца × 3 МО с известными B1, C1, C2/C3, C4, D1."""
    rng = np.random.default_rng(0)
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    cols = [m.strftime("%Y-%m") for m in months]
    t = np.arange(24) - 11.5
    # Известные параметры
    betas = [0.01, -0.005, 0.0]
    # сезонные коэффициенты: gamma_1..11; gamma_12 = -sum
    gammas = [
        np.array([0.05, 0.03, 0.0, -0.02, -0.03, -0.05, -0.04, -0.02, 0.0, 0.02, 0.03]),  # сильная сезонность
        np.array([0.0] * 11),  # нет сезонности
        np.array([0.02, 0.01, 0.0, -0.01, -0.02, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]),
    ]
    data = {}
    for j in range(3):
        g = gammas[j]
        g12 = -g.sum()
        s_full = np.concatenate([g, [g12]])  # 12
        vals = []
        for i in range(24):
            m = months[i].month
            eps = rng.normal(0, 0.005)
            vals.append(betas[j] * t[i] + s_full[m - 1] + eps)
        data[f"MO_{j}"] = vals
    return pd.DataFrame(data, index=cols).T


def test_recover_parameters():
    r = _make_synthetic_matrix()
    params, residuals = fit_regressions(r)
    months_int = np.array([int(c.split("-")[1]) for c in r.columns])
    feats, s = compute_features(params, residuals, r, months_int)

    # B1 примерно совпадает
    assert abs(feats.loc["MO_0", "B1"] - 0.01) < 0.005
    assert abs(feats.loc["MO_1", "B1"] + 0.005) < 0.005

    # C1 (амплитуда) > 0 у MO_0
    assert feats.loc["MO_0", "C1"] > 0.05

    # Сумма gamma = 0
    s_sum = s.sum(axis=0)
    assert np.allclose(s_sum, 0.0, atol=1e-8)

    # D1 > 0
    assert (feats["D1"] > 0).all()


# ---------- (б) Интерполяция ----------
def test_interpolation_short_gap():
    """Разрыв 2 месяца закрывается, 3 — нет."""
    import tempfile
    import yaml

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # создаём минимальные table1/table2
        rows = []
        months = pd.date_range("2023-01-01", periods=24, freq="MS")
        names = ["MO_A", "MO_B"]
        for name in names:
            for i, m in enumerate(months):
                period = f"{m.month}/{m.day}/{str(m.year)[2:]} 0:00"
                rows.append({"period": period, "value": 100.0 + i,
                             "category_15": "Все категории", "mo": name,
                             "freq": "M", "decimals": 3,
                             "unit_measure": "rub", "unit_mult": 0})
        t1 = pd.DataFrame(rows)
        # MO_A: короткий пропуск (2 мес)
        mask = (t1["mo"] == "MO_A") & (t1["period"].str.startswith("2/1/23")) | \
               (t1["mo"] == "MO_A") & (t1["period"].str.startswith("3/1/23"))
        t1.loc[mask, "value"] = np.nan
        # MO_B: длинный пропуск (3 мес)
        mask_b = (t1["mo"] == "MO_B") & (
            t1["period"].str.startswith("2/1/23") |
            t1["period"].str.startswith("3/1/23") |
            t1["period"].str.startswith("4/1/23"))
        t1.loc[mask_b, "value"] = np.nan

        t2 = pd.DataFrame({
            "Федеральный округ": ["A", "A"],
            "Субъект": ["S1", "S1"],
            "МО": names,
            "Население 2023": [10000, 10000],
            "Население 2024": [10000, 10000],
        })
        data_dir = td / "data"
        data_dir.mkdir()
        t1.to_csv(data_dir / "table1.csv", index=False, encoding="utf-8-sig")
        t2.to_csv(data_dir / "table2.csv", index=False, encoding="utf-8-sig")
        out_dir = td / "out"
        cfg = {
            "paths": {"table1": str(data_dir / "table1.csv"),
                      "table2": str(data_dir / "table2.csv"),
                      "out_dir": str(out_dir)},
            "key_col_t1": None, "key_col_t2": None,
            "category_name": "Все категории", "max_gap": 2,
            "use_raw_log": False,
            "blocks": {"A": ["A1", "A2"], "B": ["B1", "B2"],
                       "C": ["C1", "C2", "C3", "C4", "C5"], "D": ["D1", "D2"]},
            "feature_priority": ["A1", "B1", "D1", "D2", "C1", "C2", "C3", "C4", "C5", "A2", "B2"],
            "winsor_pct": [1, 99], "small_threshold": 3000,
            "exclude_small": False, "k_range": [2, 5], "k": "auto",
            "min_cluster_share": 0.01, "n_bootstrap": 3,
            "random_state": 42,
            "mo_type_rules": {"муниципальный район": "муниципальный район"},
            "pairplot_features": ["A1", "B1", "C1", "D1"],
            "log_level": "WARNING",
        }
        cfg_path = td / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        res = subprocess.run([sys.executable, str(ROOT / "src" / "s01_prep.py"),
                              "--config", str(cfg_path)],
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

        y = pd.read_csv(out_dir / "cache" / "y_matrix.csv",
                        encoding="utf-8-sig", index_col="mo")
        assert "MO_A" in y.index
        assert "MO_B" not in y.index


# ---------- (в) Сквозной прогон и ARI с truth ----------
def test_end_to_end(tmp_path):
    """Полный прогон на синтетических данных, ARI с truth > 0.8."""
    # Синтетические данные
    gen = subprocess.run([sys.executable, str(ROOT / "make_synthetic_data.py"),
                          "--n", "200", "--seed", "1"],
                         capture_output=True, text=True, cwd=str(ROOT))
    assert gen.returncode == 0, gen.stderr

    cfg_path = ROOT / "config.yaml"
    # запускаем шаги 1-7
    for script in ["s01_prep.py", "s02_features.py", "s04_preprocess.py",
                   "s05_select_k.py", "s06_cluster.py"]:
        res = subprocess.run([sys.executable, str(ROOT / "src" / script),
                              "--config", str(cfg_path)],
                             capture_output=True, text=True, cwd=str(ROOT))
        assert res.returncode == 0, f"{script}: {res.stderr}"

    truth = pd.read_csv(ROOT / "data" / "truth.csv", encoding="utf-8-sig")
    clusters = pd.read_csv(ROOT / "out" / "cache" / "clusters.csv",
                           encoding="utf-8-sig", index_col="mo")
    merged = truth.set_index("mo").join(clusters[["kmeans"]], how="inner")
    # отбрасываем малые (-1)
    merged = merged[merged["true_type"] >= 0]
    if len(merged) < 10:
        pytest.skip("Слишком мало МО для оценки ARI")
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(merged["true_type"], merged["kmeans"])
    assert ari > 0.5, f"ARI слишком низкий: {ari}"