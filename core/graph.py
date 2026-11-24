"""Graph operations, spatial snapping, routing algorithms, and network disruption analysis.

Provides DiGraph construction, 25km Haversine port fusion, shortest path routing
(Dijkstra, Yen's k-shortest paths, A* search), chokepoint simulation, and igraph conversion.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def snap_gfw_to_portwatch(
    pw_ports_df: pd.DataFrame,
    gfw_visits_clean_df: pd.DataFrame,
    max_dist_km: float = 25.0,
) -> Dict[str, str]:
    """Snaps GFW port nodes to closest PortWatch ports within max_dist_km Haversine distance.
    
    Returns a dictionary mapping GFW node names to either existing PortWatch node_id
    (if distance <= 25km) or 'gfw:{node}' (if distance > 25km).
    """
    # 1. Compute median coordinates for GFW nodes
    valid_visits = gfw_visits_clean_df.dropna(subset=["lat", "lon", "node"])
    gfw_coords = valid_visits.groupby("node")[["lat", "lon"]].median()

    # 2. Extract PortWatch candidate points
    pw_pts: List[Tuple[str, float, float, str]] = []
    for _, row in pw_ports_df.iterrows():
        nid = str(row["node_id"]) if "node_id" in row else str(row["portid"])
        lat = float(row["lat"])
        lon = float(row["lon"])
        iso3 = str(row.get("ISO3", row.get("iso3", ""))).strip().upper()
        pw_pts.append((nid, lat, lon, iso3))

    # 3. Match each GFW node
    mapping: Dict[str, str] = {}
    for node_name, row in gfw_coords.iterrows():
        node_str = str(node_name)
        lat_gfw, lon_gfw = float(row["lat"]), float(row["lon"])
        iso_prefix = node_str.split(":")[0].strip().upper() if ":" in node_str else ""

        # Prioritize matching ISO3 country code first
        candidates = [(haversine_distance_km(lat_gfw, lon_gfw, pla, plo), pid)
                      for pid, pla, plo, piso in pw_pts if piso == iso_prefix]
        if not candidates:
            candidates = [(haversine_distance_km(lat_gfw, lon_gfw, pla, plo), pid)
                          for pid, pla, plo, _ in pw_pts]

        if candidates:
            dmin, best_pid = min(candidates)
            if dmin <= max_dist_km:
                mapping[node_str] = best_pid
            else:
                mapping[node_str] = f"gfw:{node_str}"
        else:
            mapping[node_str] = f"gfw:{node_str}"

    return mapping


def build_network(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    """Constructs a directed graph populated with all node and edge attributes."""
    G = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        d = row.to_dict()
        node_id = str(d.pop("node_id"))
        G.add_node(node_id, **d)

    for _, row in edges_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        if src == tgt:
            continue  # Enforce strict absence of self-loops
        G.add_edge(
            src,
            tgt,
            weight=float(row["weight"]),
            transit_days=float(row["transit_days"]),
            distance_km=float(row["distance_km"]),
            src=str(row.get("src", "portwatch")),
            gfw_trips=int(row.get("gfw_trips", 0)),
            touches_vn=bool(row.get("touches_vn", False)),
        )

    return G


def extract_lcc(G: nx.DiGraph, undirected: bool = True) -> nx.DiGraph | nx.Graph:
    """Extracts the largest connected component of the network."""
    if undirected:
        U = G.to_undirected()
        components = sorted(nx.connected_components(U), key=len, reverse=True)
        if not components:
            return U
        largest_nodes = components[0]
        return G.subgraph(largest_nodes).copy()
    else:
        components = sorted(nx.strongly_connected_components(G), key=len, reverse=True)
        if not components:
            return G
        largest_nodes = components[0]
        return G.subgraph(largest_nodes).copy()


def find_shortest_path(
    G: nx.DiGraph,
    source: str,
    target: str,
    weight: str = "transit_days",
) -> Dict[str, Any]:
    """Finds the optimal maritime path between source and target using Dijkstra's algorithm.
    
    Returns a dictionary with path nodes, total transit days, nautical distance,
    transfer count, and transfer port details.
    """
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Source '{source}' or Target '{target}' not found in graph.")

    path = nx.dijkstra_path(G, source, target, weight=weight)
    total_transit = 0.0
    total_distance = 0.0
    legs = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G[u][v]
        t_days = float(edge_data.get("transit_days", 1.0))
        dist_km = float(edge_data.get("distance_km", 0.0))
        total_transit += t_days
        total_distance += dist_km
        legs.append({
            "from": u,
            "to": v,
            "from_name": G.nodes[u].get("name", u),
            "to_name": G.nodes[v].get("name", v),
            "transit_days": t_days,
            "distance_km": dist_km,
        })

    transfer_ports = path[1:-1]
    transfers = len(transfer_ports)

    return {
        "path": path,
        "transit_days": round(total_transit, 2),
        "distance_km": round(total_distance, 1),
        "transfers": transfers,
        "transfer_ports": transfer_ports,
        "legs": legs,
    }


def find_k_shortest_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    k: int = 3,
    weight: str = "transit_days",
    avoid_nodes: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Computes Yen's k-shortest loopless paths with optional node avoidance.
    
    Returns a list of path summary dictionaries in non-decreasing order of cost.
    """
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Source '{source}' or Target '{target}' not found in graph.")

    graph_view = G
    if avoid_nodes:
        avoid_set = set(avoid_nodes) - {source, target}
        if avoid_set:
            graph_view = G.subgraph([n for n in G.nodes if n not in avoid_set])

    try:
        path_generator = nx.shortest_simple_paths(graph_view, source, target, weight=weight)
        k_paths = list(itertools.islice(path_generator, k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fallback to undirected graph to allow bidirectional navigation when directed voyages are one-way in dataset
        try:
            u_graph = graph_view.to_undirected()
            path_generator = nx.shortest_simple_paths(u_graph, source, target, weight=weight)
            k_paths = list(itertools.islice(path_generator, k))
            graph_view = u_graph
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    results = []
    for rank, p in enumerate(k_paths, start=1):
        transit_days = sum(
            (graph_view[p[i]][p[i + 1]].get("transit_days") or graph_view[p[i + 1]][p[i]].get("transit_days", 1.0))
            if graph_view.has_edge(p[i], p[i + 1])
            else 1.0
            for i in range(len(p) - 1)
        )
        dist_km = sum(
            (graph_view[p[i]][p[i + 1]].get("distance_km") or graph_view[p[i + 1]][p[i]].get("distance_km", 100.0))
            if graph_view.has_edge(p[i], p[i + 1])
            else 100.0
            for i in range(len(p) - 1)
        )
        results.append({
            "rank": rank,
            "path": p,
            "transit_days": round(float(transit_days), 2),
            "distance_km": round(float(dist_km), 1),
            "transfers": max(0, len(p) - 2),
            "transfer_ports": p[1:-1],
        })

    return results


def astar_search(
    G: nx.DiGraph,
    source: str,
    target: str,
    weight: str = "transit_days",
) -> Tuple[List[str], int, float, float]:
    """Computes A* shortest path using an admissible great-circle heuristic.
    
    Returns (path, nodes_visited, runtime_sec, total_cost).
    """
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Source '{source}' or Target '{target}' not found in graph.")

    target_node = G.nodes[target]
    tgt_lat = float(target_node.get("lat", 0.0))
    tgt_lon = float(target_node.get("lon", 0.0))
    max_km_per_day = 1200.0  # Max theoretical vessel speed for admissible heuristic

    def heuristic(u: str, v: str) -> float:
        node_u = G.nodes[u]
        dist_km = haversine_distance_km(
            float(node_u.get("lat", 0.0)),
            float(node_u.get("lon", 0.0)),
            tgt_lat,
            tgt_lon,
        )
        return dist_km / max_km_per_day

    t0 = time.perf_counter()
    path = nx.astar_path(G, source, target, heuristic=heuristic, weight=weight)
    runtime_sec = time.perf_counter() - t0

    cost = nx.astar_path_length(G, source, target, heuristic=heuristic, weight=weight)
    nodes_visited = len(path)  # Approximate proxy

    return path, nodes_visited, runtime_sec, cost


def simulate_node_removal(
    G: nx.DiGraph,
    nodes_to_remove: List[str],
) -> Dict[str, Any]:
    """Simulates network disruption caused by removing specific nodes (e.g., chokepoint closure).
    
    Evaluates relative LCC degradation, approximate global efficiency, and affected VN ports.
    """
    U0 = G.to_undirected()
    n0 = U0.number_of_nodes()
    if n0 == 0:
        return {"lcc_frac": 0.0, "efficiency_frac": 0.0, "vn_isolated_count": 0}

    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)
    UH = H.to_undirected()

    if UH.number_of_nodes() == 0:
        return {
            "lcc_frac": 0.0,
            "efficiency_frac": 0.0,
            "vn_isolated_count": sum(1 for _, d in G.nodes(data=True) if d.get("is_vn")),
        }

    # LCC fraction
    comps = list(nx.connected_components(UH))
    lcc = max(comps, key=len) if comps else set()
    lcc_frac = len(lcc) / n0

    # VN isolated count
    vn_nodes = [n for n, d in G.nodes(data=True) if d.get("is_vn")]
    vn_isolated_count = sum(1 for v in vn_nodes if v not in lcc)

    # Approximate global efficiency using sampled sources
    sample_size = min(100, len(UH))
    sampled_nodes = list(UH.nodes())[:sample_size]
    inv_dist_sum = 0.0
    pairs_count = 0

    for s in sampled_nodes:
        lengths = nx.single_source_shortest_path_length(UH, s)
        for t, d in lengths.items():
            if s != t and d > 0:
                inv_dist_sum += 1.0 / d
                pairs_count += 1

    eff = inv_dist_sum / pairs_count if pairs_count > 0 else 0.0

    return {
        "lcc_frac": round(lcc_frac, 4),
        "efficiency_frac": round(eff, 4),
        "vn_isolated_count": vn_isolated_count,
        "remaining_nodes": UH.number_of_nodes(),
        "remaining_edges": UH.number_of_edges(),
    }


def to_igraph(G: nx.DiGraph | nx.Graph) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
    """Converts a NetworkX graph to an igraph.Graph preserving weights and attributes.
    
    Returns (igraph_graph, node_to_idx, idx_to_node).
    """
    import igraph as ig

    directed = G.is_directed()
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for i, n in enumerate(nodes)}

    edges = []
    weights = []
    transits = []

    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(float(data.get("weight", 1.0)))
        transits.append(float(data.get("transit_days", 1.0)))

    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=directed)
    g_ig.vs["name"] = nodes
    g_ig.es["weight"] = weights
    g_ig.es["transit_days"] = transits

    return g_ig, node_to_idx, idx_to_node
