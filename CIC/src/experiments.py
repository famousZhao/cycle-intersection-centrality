# -*- coding: utf-8 -*-
"""
Main experiments for the CIC paper.
1. Correlation analysis (Kendall tau between metric rankings)
2. MFVS greedy node-deletion experiment (cycle-native ground truth)
3. SIR epidemic spreading (boundary experiment)
4. Percolation robustness (boundary experiment)
5. Pinning control experiment
All results are written to ../results as JSON; figures to ../figures.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.stats import kendalltau

from metrics import all_metrics, enumerate_short_cycles, cycle_intersection_centrality, cycle_diversity_values, node_cycle_count
from fvs import exact_mfvs_size, greedy_fvs_by_rank, chordless_cycles

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MAX_CYCLE_LEN = 8
# C.elegans and Email-Eu-core have enormous numbers of chordless cycles;
# use a shorter length bound there to keep enumeration tractable.
CYCLE_LEN_OVERRIDE = {"celegans": 5, "email_eucore": 4}
SIR_BETA_FACTOR = 1.5   # beta relative to the network epidemic threshold
GAMMA = 1.0
SIR_REPEATS = 200
SIR_STEPS = 100
PERC_STEPS = 50
TOP_K_LIST = [1, 2, 3, 5, 8, 10, 15, 20]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_all_networks():
    nets = {}
    Gk = nx.karate_club_graph().to_undirected()
    nets["karate"] = Gk
    for name, fn in [("dolphins", "dolphins.gml"), ("polbooks", "polbooks.gml"),
                     ("powergrid", "power.gml")]:
        nets[name] = nx.read_gml(os.path.join(DATA_DIR, fn), label="id").to_undirected()
    # celegans via manual parse (duplicate edges)
    import re
    with open(os.path.join(DATA_DIR, "celegansneural.gml"), "r", encoding="utf-8") as f:
        text = f.read()
    Gc = nx.Graph()
    for m in re.finditer(r"node\s*\[(.*?)\]", text, re.S):
        im = re.search(r"id\s+(\d+)", m.group(1))
        if im:
            Gc.add_node(int(im.group(1)))
    for m in re.finditer(r"edge\s*\[(.*?)\]", text, re.S):
        sm = re.search(r"source\s+(\d+)", m.group(1))
        tm = re.search(r"target\s+(\d+)", m.group(1))
        if sm and tm:
            u, v = int(sm.group(1)), int(tm.group(1))
            if not Gc.has_edge(u, v):
                Gc.add_edge(u, v)
    nets["celegans"] = Gc
    # email-eu-core
    import gzip
    edges = []
    with gzip.open(os.path.join(DATA_DIR, "email-Eu-core.txt.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                edges.append((int(parts[0]), int(parts[1])))
    Ge = nx.Graph()
    Ge.add_edges_from(edges)
    Ge.remove_edges_from(nx.selfloop_edges(Ge))   # email-Eu-core contains self-loops
    nets["email_eucore"] = Ge
    # Two additional datasets (Euroroads and a yeast protein-interaction
    # network) are loaded here to keep the pipeline complete and reproducible,
    # while the paper reports the six core networks (karate, dolphins, polbooks,
    # power grid, C. elegans, Email-Eu-core). The extra two do not affect the
    # paper's reported results.
    # Euroroads (Subelj): sparse mesh road network, largest connected component
    nets["euroroad"] = nx.read_gml(os.path.join(DATA_DIR, "euroroad.gml"), label="id").to_undirected()
    # Yeast (Moreno PROPRO) protein-interaction network, largest connected component
    Gy = nx.Graph()
    with open(os.path.join(DATA_DIR, "moreno_propro.txt"), "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                Gy.add_edge(int(parts[0]), int(parts[1]))
    nets["yeast"] = Gy
    return nets


# ---------------------------------------------------------------------------
# Experiment 1: correlation
# ---------------------------------------------------------------------------
def correlation_experiment(nets, metrics_all):
    out = {}
    core_metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    for name, G in nets.items():
        m = metrics_all[name]
        mat = {}
        for a in core_metrics:
            for b in core_metrics:
                # ranking-based Kendall tau
                nodes = list(G.nodes())
                ra = [m[a].get(v, 0) for v in nodes]
                rb = [m[b].get(v, 0) for v in nodes]
                tau, _ = kendalltau(ra, rb)
                mat[f"{a}|{b}"] = round(float(tau), 4)
        out[name] = mat
    return out


# ---------------------------------------------------------------------------
# Experiment 2: MFVS greedy deletion
# ---------------------------------------------------------------------------
def mfvs_experiment(nets, metrics_all):
    out = {}
    metrics = ["degree", "kshell", "hindex", "CR", "CIC", "CIC_dist"]
    for name, G in nets.items():
        if name in ("powergrid", "email_eucore", "euroroad", "yeast"):   # exact MFVS only for small graphs
            continue
        L = CYCLE_LEN_OVERRIDE.get(name, MAX_CYCLE_LEN)
        cycles = metrics_all[name].get("cycles", [])
        sigma, status = exact_mfvs_size(G, max_len=L, use_milp=True, cycles=cycles)
        m = metrics_all[name]
        row = {"sigma": sigma, "sigma_status": status, "sigma_opt_norm": 0.0}
        for met in metrics:
            k = greedy_fvs_by_rank(G, m[met])
            row[met] = k
            row[f"{met}_norm"] = round(k / sigma, 4) if sigma > 0 else 0.0
        out[name] = row
        print(f"  MFVS {name}: sigma={sigma} { {m: row[m] for m in metrics} }", flush=True)
    return out


# ---------------------------------------------------------------------------
# Experiment 3: SIR
# ---------------------------------------------------------------------------
def epidemic_threshold(G):
    degs = np.array([d for _, d in G.degree()], dtype=float)
    k1 = degs.mean()
    k2 = (degs ** 2).mean()
    return k1 / k2 if k2 > 0 else 1.0


def run_sir(G, seeds, beta, gamma=1.0, steps=100):
    """Discrete-time SIR; returns final recovered fraction."""
    state = {v: 0 for v in G.nodes()}  # 0=S,1=I,2=R
    for s in seeds:
        state[s] = 1
    n = G.number_of_nodes()
    for _ in range(steps):
        new_inf = []
        for v in G.nodes():
            if state[v] == 1:
                for u in G.neighbors(v):
                    if state[u] == 0 and random.random() < beta:
                        new_inf.append(u)
        for v in list(G.nodes()):
            if state[v] == 1 and random.random() < gamma:
                state[v] = 2
        for u in new_inf:
            if state[u] == 0:
                state[u] = 1
    return sum(1 for v in state.values() if v == 2) / n


def sir_experiment(nets, metrics_all, repeats=SIR_REPEATS):
    out = {}
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    for name, G in nets.items():
        Gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        beta_c = epidemic_threshold(Gcc)
        beta = beta_c * SIR_BETA_FACTOR
        m = metrics_all[name]
        res = {"beta_c": round(float(beta_c), 4), "beta": round(float(beta), 4)}
        for met in metrics:
            ranked = sorted(Gcc.nodes(), key=lambda v: (-m[met].get(v, 0), v))
            curves = []
            for k in TOP_K_LIST:
                if k > len(ranked):
                    break
                seeds = ranked[:k]
                vals = [run_sir(Gcc, seeds, beta, GAMMA, SIR_STEPS) for _ in range(repeats)]
                curves.append(round(float(np.mean(vals)), 4))
            res[met] = curves
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# Experiment 4: Percolation
# ---------------------------------------------------------------------------
def percolation_experiment(nets, metrics_all, steps=PERC_STEPS):
    out = {}
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    for name, G in nets.items():
        n = G.number_of_nodes()
        m = metrics_all[name]
        res = {}
        for met in metrics:
            ranked = sorted(G.nodes(), key=lambda v: (-m[met].get(v, 0), v))
            H = G.copy()
            gcc_frac = []
            removed = 0
            for v in ranked:
                H.remove_node(v)
                removed += 1
                f = removed / n
                if H.number_of_nodes() == 0:
                    gcc_frac.append(0.0)
                else:
                    largest = max(len(c) for c in nx.connected_components(H))
                    gcc_frac.append(largest / n)
                if f >= 0.5:
                    break
            res[met] = [round(float(x), 4) for x in gcc_frac]
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# Experiment 5: Pinning
# ---------------------------------------------------------------------------
def pinning_experiment(nets, metrics_all):
    """Pin top-k nodes (keep them); measure the fraction of chordless short
    cycles hit (destroyed) by the pinned set. Uses cycles cached in metrics_all."""
    out = {}
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    for name, G in nets.items():
        m = metrics_all[name]
        all_cyc = m.get("cycles")
        if all_cyc is None:
            L = CYCLE_LEN_OVERRIDE.get(name, MAX_CYCLE_LEN)
            all_cyc = enumerate_short_cycles(G, max_len=L, mode="chordless")
        res = {}
        for met in metrics:
            ranked = sorted(G.nodes(), key=lambda v: (-m[met].get(v, 0), v))
            hit_rates = []
            for k in TOP_K_LIST:
                if k > len(ranked):
                    break
                pinned = set(ranked[:k])
                hit = sum(1 for c in all_cyc if c & pinned) / max(len(all_cyc), 1)
                hit_rates.append(round(float(hit), 4))
            res[met] = hit_rates
        out[name] = res
    return out


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def fig_correlation(corr, path):
    names = list(corr.keys())
    labels = ["degree", "kshell", "hindex", "CR", "CIC"]
    n = len(names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, name in enumerate(names):
        mat = np.zeros((5, 5))
        for a in range(5):
            for b in range(5):
                mat[a, b] = corr[name][f"{labels[a]}|{labels[b]}"]
        im = axes[i].imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1)
        axes[i].set_xticks(range(5))
        axes[i].set_xticklabels(["deg", "ks", "H", "CR", "CIC"], rotation=45)
        axes[i].set_yticks(range(5))
        axes[i].set_yticklabels(["deg", "ks", "H", "CR", "CIC"])
        axes[i].set_title(name)
        for a in range(5):
            for b in range(5):
                axes[i].text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Kendall rank correlation between centrality metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_mfvs(mfvs, path):
    names = list(mfvs.keys())
    metrics = ["degree", "kshell", "hindex", "CR", "CIC", "CIC_dist"]
    x = np.arange(len(names))
    width = 0.13
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    for j, met in enumerate(metrics):
        vals = [mfvs[n].get(met, 0) for n in names]
        ax.bar(x + (j - 2.5) * width, vals, width, label=met, color=colors[j])
    # optimal line
    opt = [mfvs[n]["sigma"] for n in names]
    ax.plot(x, opt, "k--", marker="o", label="exact MFVS size")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("nodes removed to break all cycles")
    ax.set_title("Greedy cycle-breaking: nodes needed by each centrality ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_sir(sir, path):
    names = list(sir.keys())
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    n = len(names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ks = TOP_K_LIST[: len(sir[name]["degree"])]
        for met in metrics:
            axes[i].plot(ks, sir[name][met], marker="o", label=met)
        axes[i].set_title(f"{name} (beta={sir[name]['beta']})")
        axes[i].set_xlabel("top-k seeds")
        axes[i].set_ylabel("final infected fraction")
        axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("SIR spreading from top-k nodes")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_percolation(perc, path):
    names = list(perc.keys())
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    n = len(names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, name in enumerate(names):
        lens = [len(perc[name][met]) for met in metrics]
        L = max(lens)
        fs = np.arange(1, L + 1) / L
        for met in metrics:
            arr = perc[name][met]
            arr = arr + [arr[-1]] * (L - len(arr))
            axes[i].plot(fs, arr, marker="o", label=met)
        axes[i].set_title(name)
        axes[i].set_xlabel("fraction removed")
        axes[i].set_ylabel("largest component fraction")
        axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Percolation: GCC under targeted node removal")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pinning(pin, path):
    names = list(pin.keys())
    metrics = ["degree", "kshell", "hindex", "CR", "CIC"]
    n = len(names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ks = TOP_K_LIST[: len(pin[name]["degree"])]
        for met in metrics:
            axes[i].plot(ks, pin[name][met], marker="o", label=met)
        axes[i].set_title(name)
        axes[i].set_xlabel("pinned top-k nodes")
        axes[i].set_ylabel("fraction of cycles hit")
        axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Pinning: cycles destroyed by controlling top-k nodes")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Theoretical mechanism analyses (Section 4.4)
# ---------------------------------------------------------------------------
def hindex_tie_analysis(nets, metrics_all):
    """Fraction of nodes sharing the modal H-index value (ranking-tie defect)."""
    out = {}
    for name, G in nets.items():
        h = metrics_all[name]["hindex"]
        vals = list(h.values())
        from collections import Counter
        cnt = Counter(vals)
        mode_val, mode_cnt = cnt.most_common(1)[0]
        out[name] = {"mode_value": mode_val, "mode_count": mode_cnt,
                     "frac_tied": round(mode_cnt / len(vals), 4),
                     "distinct_values": len(cnt)}
    return out


def cic_saturation_analysis():
    """Cactus-like family: a hub attached to k disjoint triangles.
    Shows that CIC grows with k but CIC_dist (length diversity) saturates
    at a constant, so NC alone carries cycle multiplicity."""
    import math
    out = {}
    for k in [2, 3, 5, 10, 20]:
        G = nx.Graph()
        hub = "hub"
        G.add_node(hub)
        for i in range(k):
            G.add_edge(hub, f"a{i}")
            G.add_edge(hub, f"b{i}")
            G.add_edge(f"a{i}", f"b{i}")
        L = 8
        cycles = enumerate_short_cycles(G, max_len=L, mode="chordless")
        cic = cycle_intersection_centrality(G, cycles)[hub]
        dist = cycle_diversity_values(G, cycles)[hub]
        nc = node_cycle_count(G, cycles)[hub]
        out[k] = {"CIC_hub": int(cic), "CIC_dist_hub": round(float(dist), 4),
                  "NC_hub": int(nc), "n_cycles": len(cycles)}
    return out


def k23_basis_analysis():
    """K2,3: cycle-space dimension mu = m - n + 1 = 2, but the basis-free
    chordless-cycle set has C(3,2)=3 members -> basis-free set is strictly
    larger than any cycle-space basis."""
    G = nx.complete_bipartite_graph(2, 3)
    cycles = enumerate_short_cycles(G, max_len=8, mode="chordless")
    mu = G.number_of_edges() - G.number_of_nodes() + 1
    return {"n": G.number_of_nodes(), "m": G.number_of_edges(),
            "cycle_space_dim_mu": int(mu),
            "n_chordless_cycles": len(cycles)}


def theory_analysis(nets, metrics_all):
    return {
        "hindex_ties": hindex_tie_analysis(nets, metrics_all),
        "cic_saturation": cic_saturation_analysis(),
        "k23_basis": k23_basis_analysis(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)
    nets = load_all_networks()
    metrics_all = {}
    import hashlib
    cache_path = os.path.join(RESULTS_DIR, "metrics_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    for name, G in nets.items():
        L = CYCLE_LEN_OVERRIDE.get(name, MAX_CYCLE_LEN)
        print(f"computing metrics: {name} (n={G.number_of_nodes()}, m={G.number_of_edges()}, L={L})", flush=True)
        if name in cache:
            m = cache[name]
            # JSON serialization turns integer node ids into str keys; restore int.
            fixed = {}
            for kk, vv in m.items():
                if isinstance(vv, dict):
                    fixed[kk] = {int(k2): v2 for k2, v2 in vv.items()}
                else:
                    fixed[kk] = vv
            fixed["cycles"] = None
            metrics_all[name] = fixed
            print(f"  (cached) #chordless cycles (<= {L}): {fixed['n_cycles']}", flush=True)
            continue
        metrics_all[name] = all_metrics(G, max_len=L, mode="chordless")
        print(f"  #chordless cycles (<= {L}): {metrics_all[name]['n_cycles']}", flush=True)
        dump = {k: v for k, v in metrics_all[name].items() if k != "cycles"}
        cache[name] = dump
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    print("\n[1/5] correlation", flush=True)
    corr = correlation_experiment(nets, metrics_all)
    print(json.dumps(corr, indent=1)[:1500], flush=True)

    print("\n[2/5] MFVS", flush=True)
    mfvs = mfvs_experiment(nets, metrics_all)
    print(json.dumps(mfvs, indent=1), flush=True)

    print("\n[3/5] SIR", flush=True)
    sir = sir_experiment(nets, metrics_all)
    for k, v in sir.items():
        print(k, "beta:", v["beta"], "degree:", v["degree"], "CIC:", v["CIC"], flush=True)

    print("\n[4/5] percolation", flush=True)
    perc = percolation_experiment(nets, metrics_all)

    print("\n[5/5] pinning", flush=True)
    pin = pinning_experiment(nets, metrics_all)

    print("\n[6/6] theory", flush=True)
    theory = theory_analysis(nets, metrics_all)
    print(json.dumps(theory, indent=1), flush=True)

    results = {"correlation": corr, "mfvs": mfvs, "sir": sir,
               "percolation": perc, "pinning": pin, "theory": theory}
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    fig_correlation(corr, os.path.join(FIG_DIR, "fig1_correlation.png"))
    fig_mfvs(mfvs, os.path.join(FIG_DIR, "fig2_mfvs.png"))
    fig_sir(sir, os.path.join(FIG_DIR, "fig3_sir.png"))
    fig_percolation(perc, os.path.join(FIG_DIR, "fig4_percolation.png"))
    fig_pinning(pin, os.path.join(FIG_DIR, "fig5_pinning.png"))
    print("\nfigures written to", FIG_DIR, flush=True)
    print("results written to", RESULTS_DIR, flush=True)


if __name__ == "__main__":
    main()
