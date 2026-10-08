# -*- coding: utf-8 -*-
"""
Centrality metrics for the CIC paper.
Implements: degree, k-shell, H-index, cycle ratio (CR), node cycle count (NC),
cycle-intersection centrality (CIC) and its structural-diversity variant (CIC_dist).
All metrics operate on undirected simple graphs.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict

import networkx as nx


# ---------------------------------------------------------------------------
# Classical metrics
# ---------------------------------------------------------------------------
def degree_centrality_values(G: nx.Graph) -> dict:
    return dict(G.degree())


def kshell_values(G: nx.Graph) -> dict:
    """k-shell (core number) of every node."""
    return dict(nx.core_number(G))


def hindex_values(G: nx.Graph) -> dict:
    """H-index centrality: largest h such that node has >= h neighbours with degree >= h."""
    deg = dict(G.degree())
    out = {}
    for v in G.nodes():
        neigh_deg = sorted((deg[u] for u in G.neighbors(v)), reverse=True)
        h = 0
        for i, d in enumerate(neigh_deg, start=1):
            if d >= i:
                h = i
            else:
                break
        out[v] = h
    return out


# ---------------------------------------------------------------------------
# Cycle enumeration
# ---------------------------------------------------------------------------
def enumerate_short_cycles(G: nx.Graph, max_len: int = 8, mode: str = "chordless"):
    """Enumerate short cycles (as frozensets of nodes) of length <= max_len.
    mode='chordless' returns only chordless cycles; mode='all' returns every simple cycle.
    The result is a list of frozensets (each cycle identified by its node set).
    """
    cycles = []
    seen = set()
    if mode == "chordless":
        for cyc in nx.chordless_cycles(G, length_bound=max_len):
            key = frozenset(cyc)
            if key not in seen:
                seen.add(key)
                cycles.append(key)
    else:
        # Undirected simple cycles: convert to a directed representation.
        D = nx.DiGraph()
        for u, v in G.edges():
            D.add_edge(u, v)
            D.add_edge(v, u)
        for cyc in nx.simple_cycles(D, length_bound=max_len):
            if len(cyc) < 3:
                continue
            # normalize orientation to get a canonical key
            key = frozenset(cyc)
            if key not in seen:
                seen.add(key)
                cycles.append(key)
    return cycles


# ---------------------------------------------------------------------------
# Cycle-based metrics
# ---------------------------------------------------------------------------
def cycle_ratio_values(G: nx.Graph, cycles) -> dict:
    """CR(v) = (# cycles containing v) / (# cycles)."""
    total = max(len(cycles), 1)
    cnt = Counter()
    for c in cycles:
        for v in c:
            cnt[v] += 1
    return {v: cnt[v] / total for v in G.nodes()}


def node_cycle_count(G: nx.Graph, cycles) -> dict:
    """NC(v) = number of short cycles containing v."""
    cnt = Counter()
    for c in cycles:
        for v in c:
            cnt[v] += 1
    return {v: cnt[v] for v in G.nodes()}


def cycle_intersection_centrality(G: nx.Graph, cycles) -> dict:
    """CIC(v): cycle-intersection strength.

    CIC(v) = sum_{u != v} C(c_{vu}, 2),

    where c_{vu} is the number of chordless short cycles containing BOTH v and u.
    NC / CR measure *convergence*: how many cycles pass through v.  CIC measures
    *intersection*: how much structure the pairs of cycles meeting at v genuinely
    share beyond v itself.  A hub on k disjoint triangles has NC = k but CIC = 0,
    because no two triangles share an edge; cycle pairs that share an edge or a
    longer path through v contribute one per shared node u.  Evaluated by
    accumulating, for every cycle, one co-occurrence count on each pair of its
    nodes: O(#cycles * L^2) with L the mean cycle length, tractable even for
    Email-Eu-core (1.01M chordless cycles of length <= 4).
    """
    coccur = defaultdict(Counter)          # node -> {other node: co-occurrence count}
    for c in cycles:
        nodes = list(c)
        L = len(nodes)
        for i in range(L):
            vi = nodes[i]
            for j in range(i + 1, L):
                vj = nodes[j]
                coccur[vi][vj] += 1
                coccur[vj][vi] += 1
    out = {}
    for v in G.nodes():
        s = 0
        for u, cnt in coccur.get(v, {}).items():
            s += cnt * (cnt - 1) // 2
        out[v] = s
    return out

def cycle_diversity_values(G: nx.Graph, cycles) -> dict:
    """CIC_dist(v): structural diversity of the cycles passing through v,
    measured by the normalised Shannon entropy of cycle lengths.
    Saturation behaviour: for a hub on k identical triangles the diversity
    term stays constant as k grows (only NC carries cycle multiplicity).
    """
    node_lengths = defaultdict(list)
    for c in cycles:
        for v in c:
            node_lengths[v].append(len(c))
    out = {}
    for v in G.nodes():
        lens = node_lengths.get(v, [])
        if not lens:
            out[v] = 0.0
            continue
        cnt = Counter(lens)
        n = len(lens)
        ent = -sum((c / n) * (c / n and math.log2(c / n)) for c in cnt.values())
        # normalised by log2(#distinct lengths)
        distinct = len(cnt)
        out[v] = ent / math.log2(distinct) if distinct > 1 else 0.0
    return out


def all_metrics(G: nx.Graph, max_len: int = 8, mode: str = "chordless") -> dict:
    """Return dict of metric name -> {node: score} for every implemented metric."""
    cycles = enumerate_short_cycles(G, max_len=max_len, mode=mode)
    return {
        "degree": degree_centrality_values(G),
        "kshell": kshell_values(G),
        "hindex": hindex_values(G),
        "CR": cycle_ratio_values(G, cycles),
        "NC": node_cycle_count(G, cycles),
        "CIC": cycle_intersection_centrality(G, cycles),
        "CIC_dist": cycle_diversity_values(G, cycles),
        "n_cycles": len(cycles),
        "cycles": cycles,
    }
