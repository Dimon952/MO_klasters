# файл: run_all.py
"""Единая точка входа: запускает шаги пайплайна s01..s09."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

STEPS = {
    "prep": ROOT / "src" / "s01_prep.py",
    "features": ROOT / "src" / "s02_features.py",
    "diagnostics": ROOT / "src" / "s03_diagnostics.py",
    "preprocess": ROOT / "src" / "s04_preprocess.py",
    "select_k": ROOT / "src" / "s05_select_k.py",
    "cluster": ROOT / "src" / "s06_cluster.py",
    "profile": ROOT / "src" / "s07_profile.py",
    "viz_clusters": ROOT / "src" / "s08_viz_clusters.py",
    "viz_profiles": ROOT / "src" / "s08_viz_profiles.py",
    "robust": ROOT / "src" / "s09_robust.py",
    "graph": ROOT / "src" / "s10_graph.py",
}

ORDER = [
    "prep", "features", "diagnostics", "preprocess", "select_k",
    "cluster", "profile", "viz_clusters", "viz_profiles", "robust",
    "robust", "graph",
]

DEPENDS = {
    "features": ["prep"],
    "diagnostics": ["features"],
    "preprocess": ["features"],
    "select_k": ["preprocess"],
    "cluster": ["select_k"],
    "profile": ["cluster"],
    "viz_clusters": ["cluster"],
    "viz_profiles": ["cluster", "profile"],
    "robust": ["cluster"],
    "graph": ["cluster"],
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--step", default="all", choices=["all"] + ORDER)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--map", default=None)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _read_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Конфиг не найден: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _run(cmd: list[str]) -> None:
    logging.info("Запуск: %s", " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if res.returncode != 0:
        raise RuntimeError(
            f"Команда завершилась с кодом {res.returncode}: {' '.join(str(c) for c in cmd)}"
        )


def main() -> None:
    args = _parse_args()
    cfg = _read_config(args.config)
    _setup_logging(cfg.get("log_level", "INFO"))
    log = logging.getLogger("run_all")

    if args.k is not None:
        cfg["k"] = int(args.k)
        with open(args.config, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        log.info("k переопределено: %d", args.k)

    if args.step == "all":
        steps = ORDER
    else:
        steps_set = {args.step}
        changed = True
        while changed:
            changed = False
            for s in list(steps_set):
                for dep in DEPENDS.get(s, []):
                    if dep not in steps_set:
                        steps_set.add(dep)
                        changed = True
        steps = [s for s in ORDER if s in steps_set]
        log.info("С учётом зависимостей будут запущены: %s", steps)

    for s in steps:
        script = STEPS[s]
        if not script.exists():
            raise FileNotFoundError(f"Скрипт не найден: {script}")
        cmd = [sys.executable, str(script), "--config", str(Path(args.config).resolve())]
        if s == "viz_clusters" and args.interactive:
            cmd.append("--interactive")
        if s == "viz_profiles" and args.map:
            cmd += ["--map", args.map]
        _run(cmd)

    log.info("Пайплайн завершён. Результаты в %s", cfg["paths"]["out_dir"])


if __name__ == "__main__":
    main()