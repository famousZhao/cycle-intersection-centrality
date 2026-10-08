# -*- coding: utf-8 -*-
"""Load and verify all six datasets, dump a dataset summary JSON.
C.elegans GML contains duplicate edges -> parsed manually and deduplicated.
"""
import gzip
import json
import os
import re
import networkx as nx

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def parse_gml_simple(path):
    """Parse a GML file manually, returning a simple undirected graph (dedup edges)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    G = nx.Graph()
    # node blocks
    for m in re.finditer(r"node\s*\[(.*?)\]", text, re.S):
        block = m.group(1)
        im = re.search(r"id\s+(\d+)", block)
        if im:
            G.add_node(int(im.group(1)))
    # edge blocks
    for m in re.finditer(r"edge\s*\[(.*?)\]", text, re.S):
        block = m.group(1)
        sm = re.search(r"source\s+(\d+)", block)
        tm = re.search(r"target\s+(\d+)", block)
        if sm and tm:
            u, v = int(sm.group(1)), int(tm.group(1))
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    return G


def summarize():
    out = {}
    Gk = nx.karate_club_graph().to_undirected()
    out["karate"] = {"nodes": Gk.number_of_nodes(), "edges": Gk.number_of_edges(),
                     "connected": nx.is_connected(Gk)}
    for name, fn in [("dolphins", "dolphins.gml"), ("polbooks", "polbooks.gml"),
                     ("powergrid", "power.gml")]:
        G = nx.read_gml(os.path.join(DATA_DIR, fn), label="id").to_undirected()
        out[name] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                     "connected": nx.is_connected(G)}
    Gc = parse_gml_simple(os.path.join(DATA_DIR, "celegansneural.gml"))
    out["celegans"] = {"nodes": Gc.number_of_nodes(), "edges": Gc.number_of_edges(),
                       "connected": nx.is_connected(Gc)}
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
    out["email_eucore"] = {"nodes": Ge.number_of_nodes(), "edges": Ge.number_of_edges(),
                           "connected": nx.is_connected(Ge)}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(DATA_DIR, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    summarize()
