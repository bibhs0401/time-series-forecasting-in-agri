'''visualize_graph.py
Create images of the crop graph for one training fold. Use them in slides or a report.

Creates five files in outputs/figures/graph/. Use --outdir to change this location.

  1. heatmaps.png: one adjacency matrix heatmap for each layer.
  2. network_per_layer.png: one network diagram for each layer.
  3. nodes_labeled.png: one combined graph with crop names and family colors.
  4. nodes_ego_top12.png: local graphs for the 12 crop nodes with most edges.
  5. layer_stats.csv and layer_stats.png: edge count, density, and symmetry for each layer.

Torch is not needed. This script only builds the graph from STL and adjacency
layers, so it runs anywhere the data pipeline runs.

Usage
    python -m src.graph.visualize_graph                 # default: train = all but last 52 wks
    python -m src.graph.visualize_graph --train-end 400 # explicit train cutoff
    python -m src.graph.visualize_graph --no-kg         # graph without KG layers
    python -m src.graph.visualize_graph --include-te    # also build the slow A_te layer
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # Save files without opening a display.
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from src.utils.io import read_panel
from src.graph.build_graph import build_multiplex_graph

DIRECTED_LAYERS = ('A_lag', 'A_te')


# Helpers

def _grid_shape(n: int, ncols: int = 3) -> tuple:
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _edge_count(A: np.ndarray, directed: bool) -> int:
    nnz = int((np.abs(A) > 0).sum())
    return nnz if directed else nnz // 2


# Per-layer heatmaps

def plot_heatmaps(graph, outpath: Path) -> None:
    names = graph.layer_names
    crops = graph.crops
    nrows, ncols = _grid_shape(len(names))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        A = graph.layers[name]
        directed = name in DIRECTED_LAYERS
        edge_suffix = ', directed' if directed else ''
        im = ax.imshow(A, cmap='viridis', vmin=0, aspect='equal')
        ax.set_title(f'{name}  ({_edge_count(A, directed)} edges'
                     f'{edge_suffix})', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(names):]:
        ax.axis('off')

    fig.suptitle(f'Multiplex adjacency layers  (N={len(crops)} crops)',
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}')


# Network diagrams with a shared layout

def _shared_layout(graph, seed: int = 0) -> dict:
    '''Create one spring layout from all undirected layers.'''
    N = graph.N
    union = np.zeros((N, N), dtype=float)
    for name, A in graph.layers.items():
        As = np.maximum(A, A.T)  # Use an undirected graph only to set the layout.
        union = np.maximum(union, As)
    G = nx.from_numpy_array(union)
    return nx.spring_layout(G, seed=seed, k=1.8 / np.sqrt(N), iterations=200)


def plot_networks(graph, outpath: Path, seed: int = 0) -> None:
    names = graph.layer_names
    crops = graph.crops
    pos = _shared_layout(graph, seed=seed)
    nrows, ncols = _grid_shape(len(names))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap('tab10')

    for t, (ax, name) in enumerate(zip(axes, names)):
        A = graph.layers[name]
        directed = name in DIRECTED_LAYERS
        edge_suffix = ', directed' if directed else ''
        G = (nx.from_numpy_array(A, create_using=nx.DiGraph) if directed
             else nx.from_numpy_array(np.maximum(A, A.T)))
        weights = np.array([d['weight'] for *_e, d in G.edges(data=True)])
        wnorm = (weights / weights.max()) if len(weights) and weights.max() > 0 else weights

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=70,
                               node_color='#333333', linewidths=0.3,
                               edgecolors='white')
        if len(weights):
            nx.draw_networkx_edges(
                G, pos, ax=ax, edge_color=[cmap(t % 10)] * len(weights),
                width=0.4 + 2.4 * wnorm, alpha=0.55,
                arrows=directed, arrowsize=7,
                connectionstyle='arc3,rad=0.05' if directed else 'arc3')
        ax.set_title(f'{name}  ({_edge_count(A, directed)} edges'
                     f'{edge_suffix})', fontsize=10)
        ax.axis('off')

    for ax in axes[len(names):]:
        ax.axis('off')

    fig.suptitle(f'Multiplex crop graph: one network per layer  (N={len(crops)} crops, '
                 f'shared layout)', fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}')


# Labeled node views

def plot_labeled_nodes(graph, outpath: Path, seed: int = 0) -> None:
    '''Show all undirected layers together with crop names and family colors.'''
    from src.features.static_attributes import get_raw_attributes

    N = graph.N
    crops = graph.crops
    union = np.zeros((N, N), dtype=float)
    for name, A in graph.layers.items():
        if name in DIRECTED_LAYERS:
            continue
        union = np.maximum(union, np.maximum(A, A.T))

    G = nx.from_numpy_array(union)
    pos = nx.spring_layout(G, seed=seed, k=2.2 / np.sqrt(max(N, 1)), iterations=300)
    degrees = dict(G.degree(weight='weight'))
    sizes = [280 + 40 * degrees.get(i, 0) for i in range(N)]

    attrs = get_raw_attributes(crops)
    families = [attrs.loc[c, 'family'] if c in attrs.index else 'unknown' for c in crops]
    fam_ids = {f: k for k, f in enumerate(sorted(set(families)))}
    cmap = plt.get_cmap('tab20')
    colors = [cmap(fam_ids[f] % 20) for f in families]

    fig, ax = plt.subplots(figsize=(14, 12))
    weights = np.array([d['weight'] for *_e, d in G.edges(data=True)], dtype=float)
    wnorm = (weights / weights.max()) if len(weights) and weights.max() > 0 else weights

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color='#bbbbbb',
        width=0.3 + 2.0 * wnorm, alpha=0.45,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=sizes, node_color=colors,
        linewidths=0.6, edgecolors='white',
    )
    nx.draw_networkx_labels(G, pos, labels={i: crops[i] for i in range(N)},
                            ax=ax, font_size=7)
    ax.set_title(
        f'Crop graph nodes: all undirected layers '
        f'(N={N}, colored by botanical family)',
        fontsize=12,
    )
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}')


def plot_ego_top_nodes(graph, outpath: Path, top_k: int = 12, seed: int = 0) -> None:
    '''Show local networks for the crop nodes with the most connections.'''
    crops = graph.crops
    N = graph.N
    union = np.zeros((N, N), dtype=float)
    for name, A in graph.layers.items():
        if name in DIRECTED_LAYERS:
            continue
        union = np.maximum(union, np.maximum(A, A.T))

    deg = union.sum(axis=1)
    order = np.argsort(-deg)[:top_k]
    ncols = 3
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, order):
        nbrs = [j for j in range(N) if j != i and union[i, j] > 0]
        nodes = [i] + nbrs
        sub = union[np.ix_(nodes, nodes)]
        G = nx.from_numpy_array(sub)
        G = nx.relabel_nodes(G, {k: crops[nodes[k]] for k in range(len(nodes))})
        pos = nx.spring_layout(G, seed=seed, k=1.4 / np.sqrt(max(len(nodes), 1)))
        center = crops[i]
        node_colors = ['#C44E52' if n == center else '#4C72B0' for n in G.nodes()]
        sizes = [700 if n == center else 320 for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes,
                               edgecolors='white', linewidths=0.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#999999', width=1.0, alpha=0.6)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        ax.set_title(f'{center}  (deg={int(deg[i])})', fontsize=10)
        ax.axis('off')

    for ax in axes[len(order):]:
        ax.axis('off')

    fig.suptitle(
        'Ego networks for highest-degree crop nodes (union of undirected layers)',
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}')


# Layer summary in CSV and bar chart form

def layer_stats_table(graph) -> pd.DataFrame:
    rows = []
    N = graph.N
    for name in graph.layer_names:
        A = graph.layers[name]
        directed = name in DIRECTED_LAYERS
        edges = _edge_count(A, directed)
        possible = N * (N - 1) if directed else N * (N - 1) / 2
        nz = A[np.abs(A) > 0]
        rows.append(dict(
            layer=name,
            kind='directed' if directed else 'undirected',
            edges=edges,
            density=round(edges / possible, 4) if possible else 0.0,
            mean_weight=round(float(nz.mean()), 4) if nz.size else 0.0,
            max_weight=round(float(np.abs(A).max()), 4),
            symmetric=bool(np.allclose(A, A.T, atol=1e-6)),
        ))
    return pd.DataFrame(rows)


def plot_layer_stats(df: pd.DataFrame, outpath: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#4C72B0' if k == 'undirected' else '#C44E52'
              for k in df['kind']]
    ax1.bar(df['layer'], df['edges'], color=colors)
    ax1.set_ylabel('edge count'); ax1.set_title('Edges per layer')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(df['layer'], df['density'], color=colors)
    ax2.set_ylabel('density'); ax2.set_title('Density per layer')
    ax2.tick_params(axis='x', rotation=45)

    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color='#4C72B0', label='undirected'),
                        Patch(color='#C44E52', label='directed')], fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {outpath}')


# Main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-end', type=int, default=None,
                   help='use panel[:train_end] to build the graph '
                        '(default: all but the last 52 weeks = the holdout)')
    p.add_argument('--include-te', action='store_true',
                   help='also build the transfer-entropy layer A_te (slow)')
    p.add_argument('--no-kg', action='store_true',
                   help='build the graph WITHOUT the KG layers (for the ablation figure)')
    p.add_argument('--threshold-pct', type=float, default=80.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--outdir', type=str, default=None,
                   help='output dir (default: outputs/figures/graph)')
    args = p.parse_args()

    panel = read_panel()
    crops = pd.read_csv(config.PANEL_DIR / 'crop_list.csv')['crop'].tolist()
    crops = [c for c in crops if c in panel.columns]
    panel = panel[crops]

    train_end = args.train_end if args.train_end is not None else max(1, len(panel) - 52)
    train = panel.iloc[:train_end]
    kg_state = 'off' if args.no_kg else 'on'
    print(f'Building graph on panel[:{train_end}] '
          f'({train.index[0].date()}..{train.index[-1].date()}), N={len(crops)} crops, '
          f'KG={kg_state}')

    graph = build_multiplex_graph(
        panel_train=train, crops=crops,
        include_te=args.include_te, include_kg=not args.no_kg,
        threshold_pct=args.threshold_pct, verbose=True)

    outdir = Path(args.outdir) if args.outdir else (config.FIGURES_DIR / 'graph')
    outdir.mkdir(parents=True, exist_ok=True)

    print('\nRendering figures ...')
    plot_heatmaps(graph, outdir / 'heatmaps.png')
    plot_networks(graph, outdir / 'network_per_layer.png', seed=args.seed)
    plot_labeled_nodes(graph, outdir / 'nodes_labeled.png', seed=args.seed)
    plot_ego_top_nodes(graph, outdir / 'nodes_ego_top12.png', seed=args.seed)
    stats = layer_stats_table(graph)
    stats_path = outdir / 'layer_stats.csv'
    stats.to_csv(stats_path, index=False)
    print(f'  wrote {stats_path}')
    plot_layer_stats(stats, outdir / 'layer_stats.png')

    print('\nLayer summary:')
    print(stats.to_string(index=False))
    print(f'\nDone. Figures in: {outdir}')


if __name__ == '__main__':
    main()
