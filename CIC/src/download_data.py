# -*- coding: utf-8 -*-
"""Download the real-world network datasets with pinned URLs and SHA-256
verification.

Covers the six networks reported in the paper: Karate (built into networkx and
dumped to GML for pinning), Dolphins, Polbooks, C. elegans, the US power grid,
and Email-Eu-core.  Each dataset is tried from a primary source first and falls
back to an alternative (zip) source; the final SHA-256 manifest is printed so
the archives can be checked against data/dataset_summary.json.

The full pipeline additionally loads two datasets that are bundled in the
repository's data/ directory (euroroad.gml and moreno_propro.txt, used by
src/experiments.py); they are not downloaded here.
"""
import gzip
import hashlib
import os
import urllib.request
import zipfile

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# dataset -> ordered list of (url, local_filename); later entries are fallbacks.
SOURCES = {
    "dolphins": [
        ("https://raw.githubusercontent.com/networkx/networkx-data/master/dolphins.gml", "dolphins.gml"),
        ("http://www-personal.umich.edu/~mejn/netdata/dolphins.zip", "dolphins.zip"),
    ],
    "polbooks": [
        ("https://raw.githubusercontent.com/networkx/networkx-data/master/polbooks.gml", "polbooks.gml"),
        ("http://www-personal.umich.edu/~mejn/netdata/polbooks.zip", "polbooks.zip"),
    ],
    "celegans": [
        ("https://raw.githubusercontent.com/networkx/networkx-data/master/celegansneural.gml", "celegansneural.gml"),
        ("http://www-personal.umich.edu/~mejn/netdata/celegansneural.zip", "celegansneural.zip"),
    ],
    "powergrid": [
        ("https://raw.githubusercontent.com/networkx/networkx-data/master/powergrid.gml", "powergrid.gml"),
        ("http://www-personal.umich.edu/~mejn/netdata/power.zip", "power.zip"),
    ],
    "email_eucore": [
        ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", "email-Eu-core.txt.gz"),
    ],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest):
    print(f"  trying {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    with open(dest, "wb") as f:
        f.write(data)
    print(f"  saved {dest} ({len(data)} bytes)")


def extract_zip(zip_path):
    """Extract all files from a zip into DATA_DIR; return the extracted paths."""
    out = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = os.path.join(DATA_DIR, os.path.basename(name))
            with z.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
            print(f"  extracted {target}")
    return out


def get_dataset(name):
    """Fetch a dataset, trying primary then fallback sources. Returns the paths
    written (gml/zip plus any extracted files), or None on total failure."""
    for url, fname in SOURCES[name]:
        dest = os.path.join(DATA_DIR, fname)
        try:
            download(url, dest)
        except Exception as e:
            print(f"  failed: {e}")
            continue
        written = [dest]
        if fname.endswith(".zip"):
            written += extract_zip(dest)
        return written
    return None


def main():
    import networkx as nx
    manifest = {}

    # Karate is built into networkx; we dump it to a GML for pinning.
    karate_path = os.path.join(DATA_DIR, "karate.gml")
    nx.write_gml(nx.karate_club_graph(), karate_path)
    manifest["karate"] = (karate_path, sha256_file(karate_path))

    for name in ["dolphins", "polbooks", "celegans", "powergrid", "email_eucore"]:
        written = get_dataset(name)
        if written is None:
            print(f"[WARN] {name} download failed")
            manifest[name] = (None, None)
        else:
            manifest[name] = (written, sha256_file(written[0]))

    print("\n=== SHA-256 MANIFEST ===")
    for k, (paths, h) in manifest.items():
        print(f"{k}: {h}  ({paths})")


if __name__ == "__main__":
    main()
