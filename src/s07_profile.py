# файл: src/s07_profile.py
"""Шаг 7: профили кластеров, тесты, описания, предупреждения, exemplars, cluster_series."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, load_config, load_csv, save_csv, setup_logging, month_columns,
)

log = logging.getLogger("s07_profile")

PHRASE_MAP = {
    ("A1", "+"): "повышенный уровень расходов",
    ("A1", "-"): "пониженный уровень расходов",
    ("A2", "+"): "высокий текущий уровень (2024)",
    ("A2", "-"): "низкий текущий уровень (2024)",
    ("B1", "+"): "устойчивый рост расходов",
    ("B1", "-"): "снижение расходов",
    ("B2", "+"): "положительная годовая динамика",
    ("B2", "-"): "отрицательная годовая динамика",
    ("C1", "+"): "ярко выраженная сезонность",
    ("C1", "-"): "слабая сезонность",
    ("C2", "+"): "выраженная гармоника (косинус)",
    ("C2", "-"): "низкая гармоника (косинус)",
    ("C3", "+"): "выраженная гармоника (синус)",
    ("C3", "-"): "низкая гармоника (синус)",
    ("C4", "+"): "летний (курортный) пик",
    ("C4", "-"): "снижение летом",
    ("C5", "+"): "предновогодний пик",
    ("C5", "-"): "снижение в декабре",
    ("D1", "+"): "высокая волатильность остатка",
    ("D1", "-"): "низкая волатильность остатка",
    ("D2", "+"): "склонность к резким скачкам",
    ("D2", "-"): "стабильный ряд",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="kmeans", choices=["kmeans", "ward", "gmm"])
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)

    feats = load_csv(dirs["cache"] / "features_raw.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    clusters = load_csv(dirs["cache"] / "clusters.csv", index_col="mo")
    y = load_csv(dirs["cache"] / "y_matrix.csv", index_col="mo")
    r = load_csv(dirs["cache"] / "r_matrix.csv", index_col="mo")
    sp = load_csv(dirs["cache"] / "seasonal_profile.csv", index_col="mo")

    label_col = args.labels
    labels = clusters[label_col].astype(int)
    # Если есть assigned_small и exclude_small — используем его
    if "assigned_small" in clusters.columns and cfg["exclude_small"]:
        mask = clusters[label_col].isna()
        labels = clusters[label_col].copy()
        labels.loc[mask] = clusters.loc[mask, "assigned_small"]
        labels = labels.astype(int)

    feats = feats.loc[labels.index]
    meta = meta.loc[labels.index]
    y = y.loc[labels.index]
    r = r.loc[labels.index]
    sp = sp.loc[labels.index]

    k = int(labels.nunique())
    n_total = len(labels)

    # 1. Профиль кластера
    profile_rows = []
    for c in sorted(labels.unique()):
        sub = labels[labels == c].index
        row = {
            "cluster": c,
            "n": len(sub),
            "share": len(sub) / n_total,
            "median_pop_2023": float(meta.loc[sub, "pop_2023"].median()),
            "share_small": float(meta.loc[sub, "small"].mean()),
            "top_fd": ", ".join(meta.loc[sub, "federal_district"].value_counts().head(3).index.astype(str)),
            "top_subject": ", ".join(meta.loc[sub, "subject"].value_counts().head(5).index.astype(str)),
        }
        for f in feats.columns:
            row[f"mean_{f}"] = float(feats.loc[sub, f].mean())
            row[f"median_{f}"] = float(feats.loc[sub, f].median())
        # mo_type shares
        for t, v in meta.loc[sub, "mo_type"].value_counts(normalize=True).items():
            row[f"share_type_{t}"] = float(v)
        profile_rows.append(row)
    profile = pd.DataFrame(profile_rows).set_index("cluster")
    save_csv(profile, dirs["tables"] / "profile.csv", index_name="cluster")

    # z-оценки кластерных средних
    overall_mean = feats.mean()
    overall_std = feats.std().replace(0, 1.0)
    z_rows = []
    for c in sorted(labels.unique()):
        sub = labels[labels == c].index
        z_rows.append(((feats.loc[sub].mean() - overall_mean) / overall_std).rename(c))
    z_df = pd.DataFrame(z_rows)
    z_df.index.name = "cluster"
    save_csv(z_df, dirs["tables"] / "profile_z.csv", index_name="cluster")

    # 2. Краскел-Уоллис
    eff_rows = []
    for f in feats.columns:
        groups = [feats.loc[labels == c, f].dropna().to_numpy() for c in sorted(labels.unique())]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        H, p = stats.kruskal(*groups)
        # eta^2 = (H - k + 1) / (n - k)
        n_obs = sum(len(g) for g in groups)
        eta2 = max(0.0, (H - len(groups) + 1) / max(n_obs - len(groups), 1))
        eff_rows.append({"feature": f, "H": float(H), "p_value": float(p), "eta2": float(eta2)})
    eff_df = pd.DataFrame(eff_rows).sort_values("eta2", ascending=False)
    save_csv(eff_df, dirs["tables"] / "feature_effects.csv", index_name="row")

    # 3. Текстовые описания
    desc_lines = ["# Описания кластеров\n"]
    for c in sorted(labels.unique()):
        z = z_df.loc[c]
        top = z.abs().sort_values(ascending=False).head(3)
        parts = []
        for f in top.index:
            sign = "+" if z[f] >= 0 else "-"
            phrase = PHRASE_MAP.get((f, sign), f"{f} {'выше' if sign == '+' else 'ниже'} среднего")
            parts.append(f"{phrase} ({f}: z={z[f]:.2f})")
        desc_lines.append(f"## Кластер {c} (n={int((labels == c).sum())})")
        desc_lines.append("- " + "; ".join(parts))
        desc_lines.append("")
    (Path(cfg["paths"]["out_dir"]) / "cluster_descriptions.md").write_text(
        "\n".join(desc_lines), encoding="utf-8")

    # 4. Предупреждения
    warns = []
    min_share = float(cfg["min_cluster_share"])
    for c in sorted(labels.unique()):
        sub = labels[labels == c].index
        n_c = len(sub)
        if n_c / n_total < min_share:
            warns.append({"cluster": c, "warning": f"размер кластера {n_c/n_total:.3f} < {min_share}"})
        if meta.loc[sub, "small"].mean() > 0.5:
            warns.append({"cluster": c, "warning": ">50% малых МО"})
        if meta.loc[sub, "subject"].value_counts(normalize=True).iloc[0] > 0.7:
            warns.append({"cluster": c, "warning": ">70% из одного субъекта"})
        if meta.loc[sub, "mo_type"].value_counts(normalize=True).iloc[0] > 0.7:
            warns.append({"cluster": c, "warning": ">70% одного типа МО"})
    wdf = pd.DataFrame(warns)
    save_csv(wdf, dirs["tables"] / "warnings.csv", index_name="row")
    if warns:
        log.warning("Предупреждения по кластерам: %s", warns)

    # 5. Exemplars
    ex_rows = []
    for c in sorted(labels.unique()):
        sub = clusters.loc[labels == c].sort_values("dist_kmeans")
        for _, row in sub.head(5).iterrows():
            ex_rows.append({"cluster": c, "mo": row.name, "type": "typical",
                            "dist_kmeans": row["dist_kmeans"], "sil_kmeans": row["sil_kmeans"]})
        bord = sub[sub["sil_kmeans"] < 0].sort_values("sil_kmeans").head(5)
        for _, row in bord.iterrows():
            ex_rows.append({"cluster": c, "mo": row.name, "type": "borderline",
                            "dist_kmeans": row["dist_kmeans"], "sil_kmeans": row["sil_kmeans"]})
    ex_df = pd.DataFrame(ex_rows)
    save_csv(ex_df, dirs["tables"] / "exemplars.csv", index_name="row")

    # 6. cluster_series (long)
    months = month_columns(y)
    rows = []
    for c in sorted(labels.unique()):
        sub = labels[labels == c].index
        for i, m in enumerate(months):
            rows.append({
                "cluster": c, "month": m, "series": "y",
                "q25": float(y.loc[sub, m].quantile(0.25)),
                "median": float(y.loc[sub, m].median()),
                "q75": float(y.loc[sub, m].quantile(0.75)),
            })
            rows.append({
                "cluster": c, "month": m, "series": "r",
                "q25": float(r.loc[sub, m].quantile(0.25)),
                "median": float(r.loc[sub, m].median()),
                "q75": float(r.loc[sub, m].quantile(0.75)),
            })
        # seasonal
        for i in range(1, 13):
            col = f"s{i:02d}"
            rows.append({
                "cluster": c, "month": f"M{i:02d}", "series": "seasonal",
                "q25": float(sp.loc[sub, col].quantile(0.25)),
                "median": float(sp.loc[sub, col].median()),
                "q75": float(sp.loc[sub, col].quantile(0.75)),
            })
    cs = pd.DataFrame(rows)
    save_csv(cs, dirs["cache"] / "cluster_series.csv", index_name="row")
    log.info("s07_profile завершён, k=%d, n=%d", k, n_total)


if __name__ == "__main__":
    main()