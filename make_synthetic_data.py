# файл: make_synthetic_data.py
"""Генерация синтетических данных для отладки пайплайна.

Создаёт data/table1.csv, data/table2.csv, data/truth.csv.
Запуск: python make_synthetic_data.py --n 400 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Генератор синтетических МО")
    ap.add_argument("--n", type=int, default=400, help="число МО")
    ap.add_argument("--seed", type=int, default=42, help="зерно генератора")
    return ap.parse_args()


def _make_one_type(rng: np.random.Generator, level: float, trend: float,
                   amp: float, phase: float, noise: float,
                   months: np.ndarray) -> np.ndarray:
    """Один синтетический ряд: уровень + тренд + сезонная гармоника + шум."""
    t = np.arange(len(months), dtype=float)
    seasonal = amp * np.sin(2.0 * np.pi * (t / 12.0) + phase)
    series = level + trend * t + seasonal + rng.normal(0.0, noise, size=len(t))
    return np.exp(series)


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n
    n_types = 4
    n_small = 15
    months = pd.date_range("2023-01-01", periods=24, freq="MS")

    # Параметры четырёх типов МО
    type_params = [
        {"level": 9.2, "trend": 0.003, "amp": 0.05, "phase": 0.0,   "noise": 0.02},  # 0: богатый, слабый тренд
        {"level": 9.0, "trend": 0.012, "amp": 0.10, "phase": 1.2,   "noise": 0.03},  # 1: средний, быстрый рост
        {"level": 8.7, "trend": 0.001, "amp": 0.25, "phase": 2.5,   "noise": 0.04},  # 2: курортный (сильная сезонность)
        {"level": 8.5, "trend": -0.005,"amp": 0.08, "phase": 3.7,   "noise": 0.05},  # 3: бедный, падающий
    ]

    records = []
    truth_rows = []

    n_main = max(n - n_small, 1)
    # Распределяем основное население по 4 типам
    type_ids = rng.integers(0, n_types, size=n_main)

    names_pool = [f"МО_{i:04d}" for i in range(n + 100)]
    subjects = ["Субъект_A", "Субъект_B", "Субъект_C", "Субъект_D"]
    districts = ["Центральный", "Северо-Западный", "Южный", "Приволжский"]
    mo_types_str = ["муниципальный район", "городской округ", "муниципальный округ", "ЗАТО"]

    used_names: set[str] = set()

    # Основные МО
    for i in range(n_main):
        t_id = int(type_ids[i])
        params = type_params[t_id]
        series = _make_one_type(
            rng,
            params["level"], params["trend"], params["amp"], params["phase"], params["noise"],
            months,
        )
        # имя уникально
        name = names_pool[i]
        while name in used_names:
            name = name + "_x"
        used_names.add(name)
        truth_rows.append({"mo": name, "true_type": t_id})
        for j, m in enumerate(months):
            # Формат period как в исходной таблице: M/D/YY H:00
            period_str = f"{m.month}/{m.day}/{str(m.year)[2:]} 0:00"
            records.append({
                "period": period_str,
                "value": float(series[j]),
                "category_15": "Все категории",
                "mo": name,
                "freq": "M",
                "decimals": 3,
                "unit_measure": "rub",
                "unit_mult": 0,
            })

    # Малые МО (население < 3000) — шумные ряды
    for i in range(n_small):
        idx = n_main + i
        name = names_pool[idx] if idx < len(names_pool) else f"Small_{i}"
        while name in used_names:
            name = name + "_x"
        used_names.add(name)
        base_level = rng.uniform(8.3, 9.3)
        series = _make_one_type(rng, base_level, 0.0, 0.05, rng.uniform(0, 6), 0.15, months)
        truth_rows.append({"mo": name, "true_type": -1})
        for j, m in enumerate(months):
            period_str = f"{m.month}/{m.day}/{str(m.year)[2:]} 0:00"
            records.append({
                "period": period_str,
                "value": float(series[j]),
                "category_15": "Все категории",
                "mo": name,
                "freq": "M",
                "decimals": 3,
                "unit_measure": "rub",
                "unit_mult": 0,
            })

    t1 = pd.DataFrame.from_records(records)

    # Добавим пропуски: несколько 1-2 мес и пара 3+
    all_names = list(used_names)
    # короткие пропуски
    for name in rng.choice(all_names, size=min(8, len(all_names)), replace=False):
        mask = (t1["mo"] == name)
        idxs = t1.index[mask].to_numpy()
        if len(idxs) > 5:
            k = rng.integers(3, len(idxs) - 2)
            t1.loc[idxs[k], "value"] = np.nan
    # длинные пропуски (3+)
    for name in rng.choice(all_names, size=min(2, len(all_names)), replace=False):
        mask = (t1["mo"] == name)
        idxs = t1.index[mask].to_numpy()
        if len(idxs) > 6:
            k = rng.integers(2, len(idxs) - 4)
            t1.loc[idxs[k:k + 3], "value"] = np.nan
    # выбросы
    for name in rng.choice(all_names, size=min(3, len(all_names)), replace=False):
        mask = (t1["mo"] == name)
        idxs = t1.index[mask].to_numpy()
        if len(idxs) > 5:
            k = rng.integers(1, len(idxs) - 1)
            t1.loc[idxs[k], "value"] = t1.loc[idxs[k], "value"] * rng.uniform(3.0, 6.0)

    # Таблица 2: население и адрес
    rows2 = []
    for i, name in enumerate(sorted(used_names)):
        pop23 = int(rng.integers(500, 50000)) if name.startswith("Small_") or name.startswith("МО_") else int(rng.integers(1000, 200000))
        # для малых — меньше 3000
        if name.startswith("Small_"):
            pop23 = int(rng.integers(200, 2900))
        pop24 = max(50, int(pop23 * rng.uniform(0.95, 1.03)))
        rows2.append({
            "Федеральный округ": districts[i % len(districts)],
            "Субъект": subjects[i % len(subjects)],
            "МО": name,
            "Население 2023": pop23,
            "Население 2024": pop24,
        })
    t2 = pd.DataFrame(rows2)
    truth = pd.DataFrame(truth_rows)

    t1.to_csv(out_dir / "table1.csv", index=False, encoding="utf-8-sig")
    t2.to_csv(out_dir / "table2.csv", index=False, encoding="utf-8-sig")
    truth.to_csv(out_dir / "truth.csv", index=False, encoding="utf-8-sig")
    print(f"[make_synthetic_data] Сгенерировано МО: {len(t2)}, строк t1: {len(t1)}")
    print(f"[make_synthetic_data] Файлы: data/table1.csv, data/table2.csv, data/truth.csv")


if __name__ == "__main__":
    main()