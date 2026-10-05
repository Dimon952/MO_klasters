# файл: src/utils.py
"""Общие утилиты: конфиг, логирование, IO, палитра, признаки."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # безголовый режим — не требует GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover
    sns = None


REQUIRED_KEYS = [
    "paths", "key_col_t1", "key_col_t2", "category_name", "max_gap",
    "use_raw_log", "blocks", "feature_priority", "winsor_pct",
    "small_threshold", "exclude_small", "k_range", "k",
    "min_cluster_share", "n_bootstrap", "random_state",
    "mo_type_rules", "pairplot_features", "log_level",
]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Читает YAML-конфиг и проверяет обязательные ключи."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Конфиг не найден: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Конфиг должен быть словарём")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"В конфиге отсутствуют ключи: {missing}")
    return cfg


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    """Создаёт out/cache, out/tables, out/figures. Возвращает словарь путей."""
    out = Path(cfg["paths"]["out_dir"])
    cache = out / "cache"
    tables = out / "tables"
    figures = out / "figures"
    for d in (out, cache, tables, figures):
        d.mkdir(parents=True, exist_ok=True)
    return {"out": out, "cache": cache, "tables": tables, "figures": figures}


def read_table(path: str | Path) -> pd.DataFrame:
    """Читает csv (utf-8-sig / utf-8 / cp1251, автоопределение разделителя) или xlsx."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        for sep in (None, ",", ";", "\t"):
            try:
                if sep is None:
                    df = pd.read_csv(p, sep=None, engine="python", encoding=enc)
                else:
                    df = pd.read_csv(p, sep=sep, encoding=enc)
                if df.shape[1] >= 2:
                    # снимаем BOM, если он всё же попал в имя колонки
                    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
                    return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
    raise RuntimeError(f"Не удалось прочитать {p}: {last_err}")


def save_csv(df: pd.DataFrame, path: str | Path, index_name: str = "mo") -> None:
    """Сохраняет CSV в utf-8-sig, индекс называется mo (если не RangeIndex)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df.index.name = index_name
        df.to_csv(p, encoding="utf-8-sig")
    else:
        df.to_csv(p, encoding="utf-8-sig", index=False)


def load_csv(path: str | Path, index_col: str | None = "mo") -> pd.DataFrame:
    """Загружает CSV; если index_col указан и присутствует — делает его индексом."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл не найден: {p}")
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    return df


def save_json(obj: Any, path: str | Path) -> None:
    """Сохраняет объект в JSON (utf-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Читает JSON."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_palette(n: int) -> list[tuple[float, float, float]]:
    """Colorblind-safe палитра из n цветов (на базе tab20 / Set1)."""
    base = plt.get_cmap("tab20").colors
    extra = plt.get_cmap("Set1").colors
    pool = list(base) + list(extra)
    if n <= len(pool):
        return [pool[i] for i in range(n)]
    return [pool[i % len(pool)] for i in range(n)]


def save_fig(fig: plt.Figure, name: str, cfg: dict[str, Any], dpi: int = 200) -> None:
    """Сохраняет фигуру в PNG и SVG в out/figures."""
    figures = Path(cfg["paths"]["out_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / f"{name}.png"
    svg = figures / f"{name}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)


def get_selected_features(cfg: dict[str, Any]) -> list[str]:
    """Возвращает плоский список признаков из blocks в порядке block→features."""
    feats: list[str] = []
    for _block_name, feats_list in cfg["blocks"].items():
        for f in feats_list:
            feats.append(f)
    seen = set()
    out = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def get_block_map(cfg: dict[str, Any]) -> dict[str, str]:
    """Возвращает соответствие признак → имя блока."""
    m: dict[str, str] = {}
    for block_name, feats_list in cfg["blocks"].items():
        for f in feats_list:
            m[f] = block_name
    return m


def month_columns(y: pd.DataFrame) -> list[str]:
    """Возвращает список колонок-месяцев (YYYY-MM), отсортированный."""
    return sorted([c for c in y.columns
                   if isinstance(c, str) and len(c) == 7 and c[4] == "-"])


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Деление с защитой от нуля."""
    return float(a) / float(b) if b else default