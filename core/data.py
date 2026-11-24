"""Data access layer for Hải đồ số.

Provides cached accessors (@st.cache_data and @st.cache_resource) for all 15
Parquet/JSON artifacts + manifest.json, along with graph and entity lookup helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx
import pandas as pd
import streamlit as st


def get_artifacts_dir() -> Path:
    """Resolves and returns the authoritative artifacts directory."""
    candidates = [
        Path(__file__).resolve().parent.parent / "artifacts",
        Path("artifacts").resolve(),
        Path("../artifacts").resolve(),
    ]
    for p in candidates:
        if p.exists() and (p / "nodes.parquet").exists():
            return p
    # Default to standard project artifacts dir
    return (Path(__file__).resolve().parent.parent / "artifacts").resolve()


# ---------------------------------------------------------------------------
# Individual Artifact Loaders with @st.cache_data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_nodes(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads ports node table from nodes.parquet."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "nodes.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_edges(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads maritime directed edges from edges.parquet."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "edges.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_edges_geo(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads precomputed maritime geometries from edges_geo.parquet."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "edges_geo.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_trips(artifacts_dir: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Loads simulated vessel trajectories for TripsLayer from trips.json."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "trips.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_communities(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads community partitions per node across 5 algorithms."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "communities.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_community_eval(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads community detection algorithm benchmarks and evaluation metrics."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "community_eval.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_linkpred_eval(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads link prediction heuristic and ML performance evaluations."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "linkpred_eval.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_linkpred_top(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads top 10 strategic candidate maritime connections for VN ports."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "linkpred_top.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_robustness_baseline(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads 4-strategy network attack simulation curves."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "robustness_baseline.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_structure_stats(artifacts_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads network macro structure statistics (small-world, power-law, rich-club)."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "structure_stats.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_lsbci_long(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads UNCTAD bilateral shipping connectivity index longitudinal panel."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "lsbci_long.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_lsbci_backbone(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads multiscale disparity filter trade backbone edges across quarters."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "lsbci_backbone.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_daily_vn(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads daily vessel calls panel for Vietnamese ports."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "daily_vn.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_ferry_nodes(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads domestic ferry terminals in Vietnam."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "ferry_nodes.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_ferry_edges(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Loads domestic passenger and vehicle ferry routes in Vietnam."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "ferry_edges.parquet"
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_manifest(artifacts_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Loads pipeline build manifest and lineage metadata."""
    path = Path(artifacts_dir or get_artifacts_dir()) / "manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Graph Construction & Caching with @st.cache_resource
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_nx_graph(artifacts_dir: Optional[str | Path] = None) -> nx.DiGraph:
    """Constructs and caches in-memory NetworkX DiGraph populated with attributes."""
    nodes = load_nodes(artifacts_dir)
    edges = load_edges(artifacts_dir)

    G = nx.DiGraph()
    for d in nodes.to_dict(orient="records"):
        node_id = str(d.pop("node_id"))
        G.add_node(node_id, **d)

    srcs = edges["source"].astype(str).tolist()
    tgts = edges["target"].astype(str).tolist()
    weights = edges["weight"].astype(float).tolist()
    transit_days = edges["transit_days"].astype(float).tolist()
    dist_kms = edges["distance_km"].astype(float).tolist()
    src_col = edges["src"].astype(str).tolist() if "src" in edges.columns else ["portwatch"] * len(edges)
    gfw_col = edges["gfw_trips"].fillna(0).astype(int).tolist() if "gfw_trips" in edges.columns else [0] * len(edges)
    touches_col = edges["touches_vn"].fillna(False).astype(bool).tolist() if "touches_vn" in edges.columns else [False] * len(edges)

    edge_tuples = [
        (s, t, {
            "weight": w,
            "transit_days": td,
            "distance_km": dk,
            "src": sc,
            "gfw_trips": gt,
            "touches_vn": tv,
        })
        for s, t, w, td, dk, sc, gt, tv in zip(
            srcs, tgts, weights, transit_days, dist_kms, src_col, gfw_col, touches_col
        )
    ]
    G.add_edges_from(edge_tuples)
    return G


@st.cache_resource(show_spinner=False)
def get_undirected_graph(artifacts_dir: Optional[str | Path] = None) -> nx.Graph:
    """Constructs and caches undirected projection of maritime network."""
    G = get_nx_graph(artifacts_dir)
    U = G.to_undirected()
    U.remove_edges_from(list(nx.selfloop_edges(U)))
    return U


@st.cache_resource(show_spinner=False)
def get_ferry_graph(artifacts_dir: Optional[str | Path] = None) -> nx.Graph:
    """Constructs and caches domestic ferry network graph."""
    nodes = load_ferry_nodes(artifacts_dir)
    edges = load_ferry_edges(artifacts_dir)

    F = nx.Graph()
    id_col = "terminal_id" if "terminal_id" in nodes.columns else "node_id"
    for d in nodes.to_dict(orient="records"):
        tid = str(d.pop(id_col))
        F.add_node(tid, **d)

    for d in edges.to_dict(orient="records"):
        u = str(d.pop("source"))
        v = str(d.pop("target"))
        F.add_edge(u, v, **d)
    return F


# ---------------------------------------------------------------------------
# Query Helpers
# ---------------------------------------------------------------------------

def get_port_by_id(node_id: str, artifacts_dir: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieves single port attributes dictionary by its node_id."""
    nodes = load_nodes(artifacts_dir)
    matched = nodes[nodes["node_id"] == node_id]
    if len(matched) > 0:
        return matched.iloc[0].to_dict()
    return None


def get_vn_ports(artifacts_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Returns subset of nodes representing Vietnamese ports."""
    nodes = load_nodes(artifacts_dir)
    return nodes[nodes["is_vn"] == True].copy()


def get_countries(artifacts_dir: Optional[str | Path] = None) -> List[str]:
    """Returns sorted unique list of countries present in the port network."""
    nodes = load_nodes(artifacts_dir)
    return sorted([c for c in nodes["country"].dropna().unique() if c])
