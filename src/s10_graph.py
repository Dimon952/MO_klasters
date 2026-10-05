# файл: src/s10_graph.py
"""Шаг 10: граф МО ↔ МО по косинусному сходству + Louvain/Leiden.

Строит:
- узлы: МО (атрибуты: кластер KMeans, ФО, субъект, тип, население, small)
- рёбра: косинусное сходство между векторами features_model.csv
  (порог и/или kNN)
- сообщества: Louvain (python-louvain), Leiden (igraph+leidenalg) или
  fallback — networkx.greedy_modularity_communities
- сравнение с KMeans-разметкой (ARI)

Запуск: python src/s10_graph.py --config config.yaml [--knn 15] [--threshold 0.9]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, pairwise_kernels
from sklearn.neighbors import kneighbors_graph

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ensure_dirs, get_palette, load_config, load_csv, save_csv, save_fig,
    save_json, setup_logging,
)

log = logging.getLogger("s10_graph")

# Опциональные зависимости
try:
    import community as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--knn", type=int, default=15, help="число ближайших соседей (0 = отключить)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="порог косинуса (None = только kNN)")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="разрешение Louvain/Leiden")
    ap.add_argument("--save-graphml", action="store_true",
                    help="сохранить граф в out/cache/graph.graphml")
    return ap.parse_args()


def build_similarity(X: np.ndarray) -> np.ndarray:
    """Косинусное сходство между строками X."""
    S = pairwise_kernels(X, metric="cosine")
    # численная стабилизация: обнуляем диагональ
    np.fill_diagonal(S, 0.0)
    return S


def build_graph(S: np.ndarray, nodes: list[str],
                knn: int, threshold: float | None) -> nx.Graph:
    """Строит граф: kNN и/или порог."""
    n = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    edges: set[tuple[int, int]] = set()

    # kNN-рёбра (взаимные): если j в топ-k у i или i в топ-k у j
    if knn and knn > 0:
        k = min(knn, n - 1)
        # индексы топ-k для каждой строки (исключая себя)
        idx_sorted = np.argsort(-S, axis=1)[:, :k]
        for i in range(n):
            for j in idx_sorted[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                edges.add((a, b))
        log.info("kNN=%d: добавлено %d уникальных рёбер", k, len(edges))

    # пороговые рёбра
    if threshold is not None:
        extra = 0
        for i in range(n):
            for j in range(i + 1, n):
                if S[i, j] >= threshold:
                    if (i, j) not in edges:
                        edges.add((i, j))
                        extra += 1
        log.info("Порог=%.3f: добавлено %d рёбер", threshold, extra)

    for i, j in edges:
        G.add_edge(i, j, weight=float(S[i, j]))

    # атрибуты узлов
    for i, name in enumerate(nodes):
        G.nodes[i]["mo"] = name
    return G


def detect_louvain(G: nx.Graph, resolution: float, rs: int) -> dict[int, int] | None:
    """Louvain через python-louvain. Возвращает node->community или None."""
    if not HAS_LOUVAIN:
        return None
    try:
        part = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=rs,
        )
        return part
    except Exception as e:
        log.warning("Louvain упал: %s", e)
        return None


def detect_leiden(G: nx.Graph, resolution: float, rs: int) -> dict[int, int] | None:
    """Leiden через igraph + leidenalg."""
    if not HAS_LEIDEN:
        return None
    try:
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.es["weight"] = weights
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution,
            seed=rs,
        )
        return {nodes[i]: int(part.membership[i]) for i in range(len(nodes))}
    except Exception as e:
        log.warning("Leiden упал: %s", e)
        return None


def detect_fallback(G: nx.Graph) -> dict[int, int] | None:
    """Fallback: networkx greedy_modularity_communities."""
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(
            G, weight="weight",
        )
        mapping: dict[int, int] = {}
        for cid, nodes in enumerate(communities):
            for n in nodes:
                mapping[n] = cid
        return mapping
    except Exception as e:
        log.warning("Fallback сообществ упал: %s", e)
        return None


def modularity(G: nx.Graph, mapping: dict[int, int]) -> float:
    """Модулярность разбиения."""
    try:
        communities = {}
        for n, c in mapping.items():
            communities.setdefault(c, set()).add(n)
        return float(nx.algorithms.community.modularity(
            G, list(communities.values()), weight="weight",
        ))
    except Exception:
        return float("nan")


def visualize_graph(G: nx.Graph, mapping: dict[int, int],
                    kmeans_labels: np.ndarray, cfg: dict, name: str) -> None:
    """Визуализация: 2 панели — сообщества и KMeans-кластеры."""
    n = G.number_of_nodes()
    if n > 3000:
        log.warning("Граф большой (%d узлов), spring_layout может занять время", n)
    pos = nx.spring_layout(G, seed=int(cfg["random_state"]), k=None, iterations=30)

    communities = sorted(set(mapping.values()))
    palette = get_palette(max(len(communities), kmeans_labels.max() + 1))

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Панель 1: сообщества
    node_colors_c = [palette[mapping[i] % len(palette)] for i in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3, ax=axes[0])
    nx.draw_networkx_nodes(G, pos, node_size=12, node_color=node_colors_c,
                           alpha=0.8, ax=axes[0], linewidths=0)
    axes[0].set_title(f"Сообщества ({len(communities)} шт.)")
    axes[0].axis("off")

    # Панель 2: KMeans-кластеры
    node_colors_k = [palette[int(kmeans_labels[i]) % len(palette)] for i in G.nodes()]
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.3, ax=axes[1])
    nx.draw_networkx_nodes(G, pos, node_size=12, node_color=node_colors_k,
                           alpha=0.8, ax=axes[1], linewidths=0)
    axes[1].set_title(f"KMeans (k={kmeans_labels.max() + 1})")
    axes[1].axis("off")

    fig.suptitle("Граф МО по косинусному сходству признаков", y=1.02)
    fig.tight_layout()
    save_fig(fig, name, cfg)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    dirs = ensure_dirs(cfg)
    rs = int(cfg["random_state"])

    fm = load_csv(dirs["cache"] / "features_model.csv", index_col="mo")
    clusters = load_csv(dirs["cache"] / "clusters.csv", index_col="mo")
    meta = load_csv(dirs["cache"] / "meta.csv", index_col="mo")
    fm = fm.loc[clusters.index]
    meta = meta.loc[clusters.index]
    X = fm.to_numpy(dtype=float)
    nodes = list(fm.index)
    kmeans_labels = clusters["kmeans"].astype(int).to_numpy()

    log.info("Узлов: %d; признаков: %d", X.shape[0], X.shape[1])

    # 1. Косинусное сходство
    S = build_similarity(X)
    log.info("Матрица сходства: %s; средний косинус=%.4f",
             S.shape, float(S[S > 0].mean()) if (S > 0).any() else 0.0)

    # 2. Граф
    G = build_graph(S, nodes, knn=int(args.knn), threshold=args.threshold)
    log.info("Граф: %d узлов, %d рёбер, средняя степень=%.1f",
             G.number_of_nodes(), G.number_of_edges(),
             2 * G.number_of_edges() / max(G.number_of_nodes(), 1))

    # 3. Сообщества
    method = None
    mapping: dict[int, int] | None = None
    if HAS_LOUVAIN:
        mapping = detect_louvain(G, args.resolution, rs)
        if mapping is not None:
            method = "louvain"
    if mapping is None and HAS_LEIDEN:
        mapping = detect_leiden(G, args.resolution, rs)
        if mapping is not None:
            method = "leiden"
    if mapping is None:
        mapping = detect_fallback(G)
        method = "greedy_modularity"
    if mapping is None:
        raise RuntimeError("Ни один алгоритм сообществ не сработал")

    n_comm = len(set(mapping.values()))
    mod = modularity(G, mapping)
    log.info("Сообщества: %d; метод=%s; модулярность=%.4f", n_comm, method, mod)

    # 4. Сравнение с KMeans
    comm_arr = np.array([mapping[i] for i in range(len(nodes))])
    ari = adjusted_rand_score(kmeans_labels, comm_arr)
    log.info("ARI (KMeans vs %s): %.4f", method, ari)

    # 5. Сохранение
    out_comm = pd.DataFrame({
        "mo": nodes,
        "community": comm_arr,
        "kmeans": kmeans_labels,
        "federal_district": meta["federal_district"].values,
        "subject": meta["subject"].values,
        "mo_type": meta["mo_type"].values,
        "pop_2023": meta["pop_2023"].values,
        "small": meta["small"].values,
    }).set_index("mo")
    save_csv(out_comm, dirs["cache"] / "graph_communities.csv", index_name="mo")

    # Сводная таблица
    summary = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "mean_degree": float(2 * G.number_of_edges() / G.number_of_nodes()),
        "n_communities": int(n_comm),
        "modularity": float(mod),
        "method": method,
        "ari_vs_kmeans": float(ari),
        "knn": int(args.knn),
        "threshold": float(args.threshold) if args.threshold is not None else None,
        "resolution": float(args.resolution),
    }
    save_json(summary, dirs["cache"] / "graph_summary.json")
    pd.DataFrame([summary]).to_csv(
        dirs["tables"] / "graph_summary.csv", index=False, encoding="utf-8-sig",
    )

    # 6. Визуализация
    visualize_graph(G, mapping, kmeans_labels, cfg, "s10_graph_louvain")

    # 7. (опционально) GraphML и интерактивный HTML
    if args.save_graphml:
        # подставим атрибуты
        for i, name in enumerate(nodes):
            G.nodes[i]["cluster_kmeans"] = int(kmeans_labels[i])
            G.nodes[i]["community"] = int(mapping[i])
            G.nodes[i]["federal_district"] = str(meta["federal_district"].iloc[i])
            G.nodes[i]["pop_2023"] = float(meta["pop_2023"].iloc[i])
        gml = dirs["cache"] / "graph.graphml"
        nx.write_graphml(G, gml)
        log.info("GraphML сохранён: %s", gml)

    if HAS_PLOTLY:
        # лёгкий интерактивный граф (edge trace + node trace)
        pos3 = nx.spring_layout(G, seed=rs, iterations=30)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos3[u]; x1, y1 = pos3[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        node_x = [pos3[i][0] for i in G.nodes()]
        node_y = [pos3[i][1] for i in G.nodes()]
        node_col = [int(kmeans_labels[i]) for i in G.nodes()]
        node_text = [f"{nodes[i]}<br>KMeans={kmeans_labels[i]}<br>Community={mapping[i]}"
                     for i in G.nodes()]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(width=0.3, color="#888"), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                                 marker=dict(size=6, color=node_col, colorscale="Viridis",
                                             line=dict(width=0)),
                                 text=node_text, hoverinfo="text"))
        html_path = Path(cfg["paths"]["out_dir"]) / "graph_interactive.html"
        fig.write_html(str(html_path))
        log.info("Интерактивный граф: %s", html_path)
    else:
        log.warning("plotly не установлен — интерактивный HTML пропущен")

    log.info("s10_graph завершён")


if __name__ == "__main__":
    main()