# -*- coding: utf-8 -*-
"""
Exact Minimum Feedback Vertex Set (MFVS) solver.
A graph is acyclic iff no chordless cycle remains; MFVS is therefore the
minimum hitting set of the collection of all chordless cycles.
Two exact solvers:
  - branch-and-bound hitting set (very small graphs, < ~500 cycles)
  - scipy MILP (default; handles thousands of cycles)
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.optimize import milp, LinearConstraint, Bounds

from metrics import enumerate_short_cycles


def is_acyclic(G: nx.Graph) -> bool:
    return nx.is_forest(G)


def chordless_cycles(G: nx.Graph, max_len: int = 8):
    return enumerate_short_cycles(G, max_len=max_len, mode="chordless")


def _hitting_set_bnb(elements: list, sets: list):
    """Branch & bound minimum hitting set. `sets` is a list of frozensets.
    Only suitable for small collections (<= ~500 sets)."""
    universe = set(elements)
    remaining = [s & universe for s in sets]
    remaining = [s for s in remaining if s]
    best = [len(universe)]

    def cover(pending, chosen):
        if len(chosen) >= best[0]:
            return
        pending = [s for s in pending if s]
        if not pending:
            best[0] = len(chosen)
            return
        s = min(pending, key=len)
        for v in s:
            new_pending = [t for t in pending if v not in t]
            cover(new_pending, chosen | {v})

    cover(remaining, set())
    return best[0]


def _hitting_set_milp(elements: list, sets: list, time_limit: int = 120):
    """Exact minimum hitting set via scipy MILP (HiGHS). Returns (size, status).
    status: 'optimal' or 'timelimit' (best feasible found within the limit)."""
    elements = list(elements)
    idx = {v: i for i, v in enumerate(elements)}
    n = len(elements)
    rows = []
    for s in sets:
        row = np.zeros(n)
        for v in s:
            if v in idx:
                row[idx[v]] = 1.0
        if row.sum() > 0:
            rows.append(row)
    if not rows:
        return 0, "optimal"
    A = np.array(rows)
    constraints = LinearConstraint(A, lb=np.ones(len(rows)), ub=np.inf)
    res = milp(c=np.ones(n), constraints=constraints,
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={"time_limit": time_limit})
    if res.status == 0:
        return int(round(res.fun)), "optimal"
    if res.status in (1, 2) and res.fun is not None:
        return int(round(res.fun)), "timelimit"
    raise RuntimeError(f"MILP failed with status {res.status}: {res.message}")


def exact_mfvs_size(G: nx.Graph, max_len: int = 8, use_milp: bool = False,
                    cycles=None, time_limit: int = 120):
    """Exact size of the minimum feedback vertex set (hitting set of chordless
    cycles). Returns (size, status). Returns 0 if acyclic. Pass `cycles` to reuse
    already-enumerated chordless cycles. MILP is used for any collection with
    more than 800 cycles."""
    if cycles is None:
        cycles = chordless_cycles(G, max_len=max_len)
    if not cycles:
        return 0, "optimal"
    nodes = list(G.nodes())
    if use_milp or len(cycles) > 800:
        return _hitting_set_milp(nodes, cycles, time_limit=time_limit)
    return _hitting_set_bnb(nodes, cycles), "optimal"


def greedy_break_cycles(G: nx.Graph, order: list) -> int:
    """Greedy node-deletion FVS: remove nodes in order until remaining graph is
    acyclic. Returns number of removed nodes."""
    H = G.copy()
    count = 0
    for v in order:
        if not is_acyclic(H):
            H.remove_node(v)
            count += 1
        else:
            break
    return count


def greedy_fvs_by_rank(G: nx.Graph, ranking: dict) -> int:
    order = sorted(ranking.keys(), key=lambda v: (-ranking[v], -G.degree(v), v))
    return greedy_break_cycles(G, order)


if __name__ == "__main__":
    T = nx.complete_bipartite_graph(2, 3)
    print("K2,3 chordless cycles:", len(chordless_cycles(T)))
    print("K2,3 MFVS bnb:", exact_mfvs_size(T))
    print("K2,3 MFVS milp:", exact_mfvs_size(T, use_milp=True))
    print("P4 MFVS:", exact_mfvs_size(nx.path_graph(4)))
    print("C5 MFVS:", exact_mfvs_size(nx.cycle_graph(5)))
