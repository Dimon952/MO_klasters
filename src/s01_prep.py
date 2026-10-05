# файл: src/s01_prep.py
"""Шаг 1: подготовка данных — очистка, пивот, интерполяция, соединение.

Устойчив к:
- xlsx/csv входу;
- разным названиям колонок (автоопределение);
- дубликатам названий МО (оставляем только уникальные в table2);
- дубликатам (mo, period) — усредняем value;
- BOM и пробелам в заголовках.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, load_config, read_table, save_csv, setup_logging,
)

log = logging.getLogger("s01_prep")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    return ap.parse_args()


def _norm(s: str) -> str:
    """Нормализация имени колонки для сопоставления."""
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Ищет колонку по списку кандидатов (регистронезависимо, без пробелов и _)."""
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        k = _norm(cand)
        if k in norm_map:
            return norm_map[k]
    for cand in candidates:
        k = _norm(cand)
        for norm_name, orig in norm_map.items():
            if k and k in norm_name:
                return orig
    return None


def _parse_period(series: pd.Series) -> pd.Series:
    """Явный формат %m/%d/%y %H:%M, запасной pd.to_datetime."""
    s = series.astype(str)
    parsed = pd.to_datetime(s, format="%m/%d/%y %H:%M", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed2 = pd.to_datetime(s[mask], errors="coerce")
        parsed.loc[mask] = parsed2
    return parsed


def _mo_type(name: str, subject: str, rules: dict[str, str]) -> str:
    """Определяет тип МО по подстроке и субъекту."""
    if isinstance(subject, str) and "Санкт-Петербург" in subject:
        return "внутригородское МО СПб"
    if not isinstance(name, str):
        return "другое"
    low = name.lower()
    for key, val in rules.items():
        if key == "Санкт-Петербург":
            continue
        if key.lower() in low:
            return val
    return "другое"


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    # 1. Чтение таблиц
    t1 = read_table(cfg["paths"]["table1"])
    t2 = read_table(cfg["paths"]["table2"])
    log.info("table1: %s; колонки=%s", t1.shape, list(t1.columns))
    log.info("table2: %s; колонки=%s", t2.shape, list(t2.columns))

    # 2. Автоопределение колонок в table1
    c_period = _find_column(t1, ["period", "период", "дата", "date", "time"])
    c_value = _find_column(t1, ["value", "значение", "расход", "сумма", "amount"])
    c_mo = _find_column(t1, ["mo", "мо", "муниципалитет", "territory",
                             "название", "муниципальноеобразование"])
    c_cat = _find_column(t1, ["category_15", "category15", "category",
                              "категория", "категория15"])
    if c_period is None or c_value is None or c_mo is None:
        raise KeyError(
            f"Не удалось найти обязательные колонки в table1. "
            f"Найдены: {list(t1.columns)}. Ожидались: period, value, mo."
        )
    log.info("table1: period='%s', value='%s', mo='%s', category='%s'",
             c_period, c_value, c_mo, c_cat)

    # 3. Фильтр по категории
    cat = cfg["category_name"]
    if c_cat is not None:
        n_before = len(t1)
        t1_f = t1[t1[c_cat].astype(str).str.strip() == cat].copy()
        if len(t1_f) > 0:
            t1 = t1_f
            log.info("Фильтр категории '%s': %d → %d строк", cat, n_before, len(t1))
        else:
            log.warning(
                "После фильтра по '%s' 0 строк — берём все строки. "
                "Уникальные значения колонки категорий (первые 20): %s",
                cat, sorted(t1[c_cat].astype(str).unique())[:20],
            )

    # 4. Нормализация имён колонок и парсинг period
    rename_map = {c_period: "period", c_value: "value", c_mo: "mo"}
    if c_cat is not None and c_cat in t1.columns:
        rename_map[c_cat] = "category_15"
    t1 = t1.rename(columns=rename_map)
    # оставим только нужные колонки (в исходнике были пустые Столбец1..4)
    keep_cols = [c for c in ["period", "value", "mo", "category_15"] if c in t1.columns]
    t1 = t1[keep_cols].copy()

    t1["period"] = _parse_period(t1["period"])
    n_before = len(t1)
    t1 = t1.dropna(subset=["period"])
    if len(t1) < n_before:
        log.warning("Отброшено %d строк с нераспознанным period", n_before - len(t1))
    t1["mo"] = t1["mo"].astype(str).str.strip()
    t1["value"] = pd.to_numeric(t1["value"], errors="coerce")
    t1.loc[t1["value"] <= 0, "value"] = np.nan

    # 5. Автоопределение колонок в table2
    t2 = t2.rename(columns=lambda c: str(c).strip())
    c2_mo = _find_column(t2, ["мо", "муниципалитет", "название", "territory"])
    c2_subj = _find_column(t2, ["субъект", "регион", "subject", "region"])
    c2_fd = _find_column(t2, ["федеральныйокруг", "фо", "округ", "federaldistrict"])
    c2_p23 = _find_column(t2, ["население2023", "population2023", "pop2023",
                               "население23"])
    c2_p24 = _find_column(t2, ["население2024", "population2024", "pop2024",
                               "население24"])
    if c2_mo is None or c2_p23 is None or c2_p24 is None:
        raise KeyError(
            f"Не удалось найти обязательные колонки в table2. "
            f"Найдены: {list(t2.columns)}. "
            f"Ожидались: МО, Население 2023, Население 2024."
        )
    log.info("table2: mo='%s', subject='%s', fd='%s', pop2023='%s', pop2024='%s'",
             c2_mo, c2_subj, c2_fd, c2_p23, c2_p24)

    # 6. Готовим t2 и вычисляем уникальные названия МО
    t2_use = t2[[c for c in [c2_mo, c2_fd, c2_subj, c2_p23, c2_p24] if c]].copy()
    t2_use.columns = ["mo", "federal_district", "subject", "pop_2023", "pop_2024"]
    t2_use["mo"] = t2_use["mo"].astype(str).str.strip()
    t2_use["pop_2023"] = pd.to_numeric(t2_use["pop_2023"], errors="coerce")
    t2_use["pop_2024"] = pd.to_numeric(t2_use["pop_2024"], errors="coerce")

    # полные дубли убираем
    n_t2 = len(t2_use)
    t2_use = t2_use.drop_duplicates().reset_index(drop=True)
    log.info("table2: полных дублей удалено: %d", n_t2 - len(t2_use))

    # считаем встречаемость названий
    counts = t2_use["mo"].value_counts()
    unique_names = set(counts[counts == 1].index)
    dup_names = set(counts[counts > 1].index)
    log.info("Уникальных названий МО в table2: %d; неуникальных: %d (всего строк %d)",
             len(unique_names), len(dup_names), len(t2_use))

    # сохраняем список неуникальных как справку
    if dup_names:
        dup_report = t2_use[t2_use["mo"].isin(dup_names)][["mo", "subject"]].copy()
        dup_report["n_rows_t2"] = dup_report["mo"].map(counts)
        dup_report = dup_report.sort_values(["mo", "subject"]).reset_index(drop=True)
        save_csv(dup_report, dirs["tables"] / "duplicated_mo_names.csv",
                 index_name="row")

    # 7. Оставляем в t1 только строки с уникальными названиями
    n_t1 = len(t1)
    t1 = t1[t1["mo"].isin(unique_names)].copy()
    log.info("table1: %d → %d строк после фильтра по уникальным МО (убрано %d)",
             n_t1, len(t1), n_t1 - len(t1))

    if t1.empty:
        raise ValueError(
            "После фильтра по уникальным названиям МО не осталось строк. "
            "Проверьте, что названия МО в table1 и table2 совпадают. "
            "См. out/tables/duplicated_mo_names.csv и out/tables/unmatched_mo.csv."
        )

    # 8. Дедупликация (mo, period) — усредняем value
    n_before = len(t1)
    dup_mask = t1.duplicated(subset=["mo", "period"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows > 0:
        # Проверим, одинаковые ли значения у дубликатов
        nun = (t1[dup_mask].groupby(["mo", "period"])["value"]
               .nunique(dropna=False))
        only_same = bool((nun <= 1).all())
        log.warning(
            "Найдено %d строк-дубликатов (mo, period) по %d парам. "
            "Значения %s — усредняю.",
            n_dup_rows, len(nun),
            "одинаковые" if only_same else "разные",
        )
        t1 = (t1.groupby(["mo", "period"], as_index=False)["value"]
              .mean())
        # category_15 восстановим — он не нужен дальше
        t1["category_15"] = cat
        log.info("После дедупликации: %d строк", len(t1))

    # 9. Пивот в матрицу МО × YYYY-MM
    t1["ym"] = t1["period"].dt.strftime("%Y-%m")
    y = t1.pivot_table(index="mo", columns="ym", values="value", aggfunc="mean")
    y = y.sort_index(axis=1)
    log.info("Матрица до интерполяции: %s, месяцев=%d", y.shape, y.shape[1])

    # 10. Интерполяция внутренних разрывов ≤ max_gap
    max_gap = int(cfg["max_gap"])
    arr = y.to_numpy(dtype=float, copy=True)
    n_rows = arr.shape[0]
    excluded: list[dict] = []
    keep_mask = np.ones(n_rows, dtype=bool)

    for i in range(n_rows):
        row = arr[i, :]
        valid = ~np.isnan(row)
        if not valid.any():
            excluded.append({"mo": y.index[i], "reason": "нет валидных наблюдений"})
            keep_mask[i] = False
            continue
        first = int(np.argmax(valid))
        last = int(len(valid) - 1 - np.argmax(valid[::-1]))
        if np.isnan(row[first]) or np.isnan(row[last]):
            excluded.append({"mo": y.index[i], "reason": "пропуски на краях"})
            keep_mask[i] = False
            continue
        j = first
        ok = True
        while j <= last:
            if np.isnan(row[j]):
                k = j
                while k <= last and np.isnan(row[k]):
                    k += 1
                gap_len = k - j
                if gap_len > max_gap:
                    excluded.append({
                        "mo": y.index[i],
                        "reason": f"разрыв длиной {gap_len} мес (> {max_gap})",
                    })
                    ok = False
                    break
                x0, x1 = j - 1, k
                y0, y1 = row[x0], row[x1]
                for m in range(j, k):
                    frac = (m - x0) / (x1 - x0)
                    row[m] = y0 + frac * (y1 - y0)
                j = k
            else:
                j += 1
        if ok:
            arr[i, :] = row
        else:
            keep_mask[i] = False

    y = pd.DataFrame(arr, index=y.index, columns=y.columns)
    y = y.loc[keep_mask].dropna(how="any")
    log.info("Матрица после интерполяции: %s", y.shape)

    ex_df = pd.DataFrame(excluded, columns=["mo", "reason"])
    save_csv(ex_df, dirs["tables"] / "excluded_mo.csv", index_name="row")

    # 11. Соединение с table2
    t2_use = t2_use.drop_duplicates(subset=["mo"], keep="first").set_index("mo")
    meta = t2_use.reindex(y.index)
    unmatched = meta[meta["pop_2023"].isna()].copy()
    meta = meta.dropna(subset=["pop_2023"])
    y = y.loc[meta.index]

    if not unmatched.empty:
        save_csv(unmatched.reset_index(), dirs["tables"] / "unmatched_mo.csv",
                 index_name="row")
    else:
        pd.DataFrame(columns=["mo"]).to_csv(
            dirs["tables"] / "unmatched_mo.csv", index=False, encoding="utf-8-sig")

    meta["pop_change"] = meta["pop_2024"] / meta["pop_2023"] - 1.0
    meta["small"] = meta["pop_2023"] < float(cfg["small_threshold"])
    meta["mo_type"] = [
        _mo_type(mo, subj if isinstance(subj, str) else "", cfg["mo_type_rules"])
        for mo, subj in zip(meta.index, meta["subject"].fillna(""))
    ]

    # 12. Сохранение
    save_csv(y, dirs["cache"] / "y_matrix.csv", index_name="mo")
    save_csv(meta, dirs["cache"] / "meta.csv", index_name="mo")

    log.info("Итог: y=%s, meta=%s", y.shape, meta.shape)
    log.info("Исключено МО: %d; несопоставлено: %d", len(ex_df), len(unmatched))


if __name__ == "__main__":
    main()