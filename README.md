Reproducible companion for the manuscript

> **Cycle-Intersection Centrality: A High-Order Topological Metric for Identifying
> Cycle-Structure Control Nodes in Complex Networks**

This repository contains everything needed to re-derive every figure, table and
claim in the paper: pinned datasets with SHA-256 checksums, metric definitions,
experiment code, results, and the script that generates the manuscript.

---

## 1. Layout

```
CIC_paper/
├── data/                  raw dataset archives + extracted files + dataset_summary.json
├── figures/               fig1_correlation ... fig5_pinning (generated)
├── results/               results.json (authoritative results), metrics_cache.json
├── src/
│   ├── metrics.py         degree, k-shell, H-index, CR, NC, CIC, CIC_dist + chordless-cycle enumeration
│   ├── fvs.py             MFVS via MILP on the chordless-cycle hypergraph
│   ├── experiments.py     full experiment suite (6 networks, 5 experiments, 5 figures)
│   ├── download_data.py   dataset download (primary sources)
│   ├── download_data2.py  fallback downloads (zip archives)
│   ├── verify_data.py     dataset verification (n, m, SHA-256) 
├── requirements.txt

```

## 2. Environment

Python 3.14 (3.10+ should work). Install dependencies:

```bash
python -m pip install -r requirements.txt
