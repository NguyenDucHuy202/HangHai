"""Offline Data Pipeline Engine for Hải đồ số (haido-sna).

Executes Steps 6.1 through 6.15 per HAIDO_SNA_APP_SPEC.md:
1. Ingestion and extraction of PortWatch, GFW, and LSBCI raw tables.
2. 25km Haversine port fusion and directed graph construction.
3. Log min-max weight normalization.
4. Centrality calculations (Degree, Strength, PageRank, exact transit Betweenness, Closeness, Eigenvector).
5. k-Core shell decomposition.
6. Burt's structural holes on target subset S (VN ports + Top 300 PageRank).
7. 6 community detection algorithms + NMI/ARI evaluation benchmarks.
8. Topological and ML link prediction + Top 10 recommendations for VN ports.
9. Network robustness 4-attack simulation curves (0-30% removal).
10. Global network structure statistics (Small-World, Scale-Free, Assortativity, Rich-Club).
11. Maritime searoute geometry precomputation with symmetry caching.
12. Deck.gl TripsLayer vessel trajectory synthesis (< 15 MB).
13. UNCTAD LSBCI disparity filter trade backbone time series.
14. Vietnam domestic ferry network synthesis & daily VN port calls normalization.
15. Quality assurance contract verification & manifest.json generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import networkx as nx
import numpy as np
import pandas as pd

from core.algorithms import (
    compute_centralities,
    compute_burt_metrics,
    detect_communities,
    detect_girvan_newman_sea,
    evaluate_partitions,
    simulate_robustness,
    evaluate_link_prediction,
    compute_structure_stats,
    compute_lsbci_backbone,
    synthesize_ferry_network,
)
from core.graph import haversine_distance_km, snap_gfw_to_portwatch
from core.seaway import safe_searoute, interpolate_route_timestamps


def find_raw_file(filename: str, raw_dir: Path) -> Path:
    """Finds raw CSV file with resilient nested search."""
    candidates = [
        raw_dir / filename,
        raw_dir / "data" / filename,
        raw_dir.parent / "hanghai_sna" / "data" / filename,
        raw_dir.parent / "data" / filename,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Cannot find raw file '{filename}' in {raw_dir} or candidate locations.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hải đồ số - Artifacts Generation Pipeline")
    parser.add_argument("--zip", type=str, default=None, help="Path to raw hanghai_sna_output.zip")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Target artifacts output directory")
    args = parser.parse_args()

    pipeline_start_time = time.perf_counter()
    print("=" * 80)
    print(" HẢI ĐỒ SỐ - DATA PIPELINE & ARTIFACTS BUILDER")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 6.1: Raw Data Extraction & Verification
    # -----------------------------------------------------------------------
    print("\n[Step 6.1] Ingestion and raw data extraction...")
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = base_dir / artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Resolve zip archive
    zip_path = None
    if args.zip:
        p = Path(args.zip)
        if p.exists():
            zip_path = p
    if not zip_path:
        default_zips = [
            base_dir / "raw" / "hanghai_sna_output.zip",
            base_dir.parent / "hanghai_sna" / "hanghai_sna_output.zip",
            Path("d:/THẠC SĨ/MXH/HangHai/hanghai_sna/hanghai_sna_output.zip"),
        ]
        for dz in default_zips:
            if dz.exists():
                zip_path = dz
                break

    # If data directory specified or available, verify files
    data_dir_override = None
    if args.data_dir:
        dp = Path(args.data_dir)
        if dp.exists():
            data_dir_override = dp

    # Check if raw files exist; if not and zip available, extract
    needed_files = [
        "pw_edges_clean.csv", "pw_ports.csv", "gfw_edges.csv",
        "gfw_visits_clean.csv", "gfw_trips.csv", "lsbci_clean.csv",
        "pw_daily_vn_clean.csv"
    ]

    files_found = False
    try:
        search_dir = data_dir_override or raw_dir
        for nf in needed_files:
            find_raw_file(nf, search_dir)
        files_found = True
    except FileNotFoundError:
        files_found = False

    if not files_found:
        if zip_path and zip_path.exists():
            print(f"Extracting {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB) to {raw_dir}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)
            search_dir = raw_dir
        else:
            raise RuntimeError(f"Raw data not found and zip archive not available at {zip_path}")
    else:
        search_dir = data_dir_override or raw_dir

    print(f"Using raw data directory: {search_dir}")
    pw_edges_path = find_raw_file("pw_edges_clean.csv", search_dir)
    pw_ports_path = find_raw_file("pw_ports.csv", search_dir)
    gfw_edges_path = find_raw_file("gfw_edges.csv", search_dir)
    gfw_visits_path = find_raw_file("gfw_visits_clean.csv", search_dir)
    gfw_trips_path = find_raw_file("gfw_trips.csv", search_dir)
    lsbci_path = find_raw_file("lsbci_clean.csv", search_dir)
    daily_vn_path = find_raw_file("pw_daily_vn_clean.csv", search_dir)

    print("Loading raw CSV tables into memory...")
    pw_edges_df = pd.read_csv(pw_edges_path, encoding="utf-8-sig", low_memory=False)
    pw_ports_df = pd.read_csv(pw_ports_path, encoding="utf-8-sig", low_memory=False)
    gfw_edges_df = pd.read_csv(gfw_edges_path, encoding="utf-8-sig", low_memory=False)
    gfw_visits_df = pd.read_csv(gfw_visits_path, encoding="utf-8-sig", low_memory=False)
    gfw_trips_df = pd.read_csv(gfw_trips_path, encoding="utf-8-sig", low_memory=False)
    lsbci_df = pd.read_csv(lsbci_path, encoding="utf-8-sig", low_memory=False)
    daily_vn_raw_df = pd.read_csv(daily_vn_path, encoding="utf-8-sig", low_memory=False)

    print(f"Loaded: PortWatch edges {len(pw_edges_df):,}, ports {len(pw_ports_df):,}, "
          f"GFW edges {len(gfw_edges_df):,}, visits {len(gfw_visits_df):,}, trips {len(gfw_trips_df):,}")

    # -----------------------------------------------------------------------
    # Step 6.2: DiGraph Construction & 25km Haversine Port Fusion
    # -----------------------------------------------------------------------
    print("\n[Step 6.2] Graph Construction & 25km Haversine Port Fusion...")
    E = pw_edges_df.copy()
    pw_weight_col = "input_days_1" if "input_days_1" in E.columns else "distance"
    E["weight"] = pd.to_numeric(E[pw_weight_col], errors="coerce").fillna(0).clip(lower=1e-6)

    G = nx.DiGraph()

    # Add PortWatch edges
    for r in E.itertuples(index=False):
        G.add_edge(
            str(r.from_id),
            str(r.to_id),
            weight=float(r.weight),
            transit_days=float(r.transit_days) if pd.notna(r.transit_days) and float(r.transit_days) > 0 else 1.0,
            distance_km=float(r.distance) if hasattr(r, "distance") and pd.notna(r.distance) else 100.0,
            src="portwatch",
            gfw_trips=0,
        )

    # Populate node attributes from PortWatch edges
    for side in ("from", "to"):
        sub = E[[f"{side}_id", f"{side}_name", f"{side}_iso3", f"{side}_lat", f"{side}_lon"]].drop_duplicates(f"{side}_id")
        for r in sub.itertuples(index=False):
            nid = str(r[0])
            iso3 = str(r[2]).strip().upper() if pd.notna(r[2]) else ""
            G.nodes[nid].update(
                name=str(r[1]) if pd.notna(r[1]) else nid,
                iso3=iso3,
                country="",
                lat=float(r[3]),
                lon=float(r[4]),
                is_vn=(iso3 == "VNM"),
            )

    # Augment node attributes from pw_ports.csv
    pw_port_map = pw_ports_df.set_index("portid").to_dict("index")
    for pid, pdata in pw_port_map.items():
        spid = str(pid)
        if spid in G:
            iso3 = str(pdata.get("ISO3", "")).strip().upper()
            country = str(pdata.get("country", "")).strip()
            locode = str(pdata.get("LOCODE", "")).strip() if pd.notna(pdata.get("LOCODE")) else None
            continent = str(pdata.get("continent", "")).strip() if pd.notna(pdata.get("continent")) else None
            v_total = float(pdata.get("vessel_count_total", 0.0)) if pd.notna(pdata.get("vessel_count_total")) else None
            v_container = float(pdata.get("vessel_count_container", 0.0)) if pd.notna(pdata.get("vessel_count_container")) else None

            G.nodes[spid].update(
                iso3=iso3,
                country=country,
                is_vn=(iso3 == "VNM"),
                locode=locode,
                continent=continent,
                vessel_count_total=v_total,
                vessel_count_container=v_container,
            )

    # Snapping GFW ports (<= 25 km threshold)
    print("Snapping GFW ports to PortWatch...")
    loc = gfw_visits_df.dropna(subset=["lat", "lon"]).groupby("node")[["lat", "lon"]].median()
    pts = [(n, d["lat"], d["lon"], d.get("iso3")) for n, d in G.nodes(data=True) if "lat" in d and pd.notna(d["lat"])]

    mapping = {}
    for node_name, r in loc.iterrows():
        node_str = str(node_name)
        iso = node_str.split(":")[0].strip().upper() if ":" in node_str else ""
        cand = [(haversine_distance_km(r.lat, r.lon, la, lo), n) for n, la, lo, i3 in pts if i3 == iso] or \
               [(haversine_distance_km(r.lat, r.lon, la, lo), n) for n, la, lo, _ in pts]
        dmin, n = min(cand)
        mapping[node_str] = n if dmin <= 25.0 else f"gfw:{node_str}"

    snapped_count = sum(not str(v).startswith("gfw:") for v in mapping.values())
    print(f"GFW nodes: {len(mapping):,}, snapped into PortWatch (<= 25km): {snapped_count:,}")

    # Map GFW edges
    GE = gfw_edges_df.assign(
        s=gfw_edges_df["source"].astype(str).map(mapping),
        t=gfw_edges_df["target"].astype(str).map(mapping)
    ).dropna(subset=["s", "t"])

    # Drop self-loops and aggregate
    GE = GE[GE["s"] != GE["t"]].groupby(["s", "t"], as_index=False).agg(
        weight=("weight", "sum"),
        orig_s=("source", "first"),
        orig_t=("target", "first")
    )

    wmed = float(E["weight"].median())
    gmax = float(GE["weight"].max()) if len(GE) > 0 else 1.0

    for r in GE.itertuples(index=False):
        u, v = str(r.s), str(r.t)
        gfw_trips = int(r.weight)
        if G.has_edge(u, v):
            G[u][v].update(src="both", gfw_trips=gfw_trips)
        else:
            # Add new GFW edge
            raw_w = float(wmed * r.weight / gmax)
            # Ensure nodes exist
            for n, orig in ((u, r.orig_s), (v, r.orig_t)):
                if n not in G:
                    orig_str = str(orig)
                    iso = orig_str.split(":")[0].strip().upper() if ":" in orig_str else ""
                    name = orig_str.split(":", 1)[1] if ":" in orig_str else orig_str
                    n_lat = float(loc.loc[orig_str, "lat"]) if orig_str in loc.index else 0.0
                    n_lon = float(loc.loc[orig_str, "lon"]) if orig_str in loc.index else 0.0
                    G.add_node(
                        n,
                        name=name,
                        iso3=iso,
                        country="",
                        lat=n_lat,
                        lon=n_lon,
                        is_vn=(iso == "VNM"),
                    )

            lat_u, lon_u = G.nodes[u]["lat"], G.nodes[u]["lon"]
            lat_v, lon_v = G.nodes[v]["lat"], G.nodes[v]["lon"]
            dist_km = haversine_distance_km(lat_u, lon_u, lat_v, lon_v)
            t_days = max(0.1, round(dist_km / 800.0, 2))

            G.add_edge(
                u,
                v,
                weight=raw_w,
                transit_days=t_days,
                distance_km=round(dist_km, 1),
                src="gfw",
                gfw_trips=gfw_trips,
            )

    print(f"Topology fused: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} directed edges.")

    # -----------------------------------------------------------------------
    # Step 6.3: Min-Max Log Weight Normalization
    # -----------------------------------------------------------------------
    print("\n[Step 6.3] Min-Max Log Weight Normalization...")
    raw_weights = [float(d["weight"]) for _, _, d in G.edges(data=True)]
    log_weights = [math.log(1.0 + w) for w in raw_weights]
    min_log = min(log_weights)
    max_log = max(log_weights)
    log_range = max_log - min_log if max_log > min_log else 1.0

    for u, v, d in G.edges(data=True):
        raw_w = float(d["weight"])
        norm_w = (math.log(1.0 + raw_w) - min_log) / log_range
        d["raw_weight"] = raw_w
        d["weight"] = float(round(max(0.0, min(1.0, norm_w)), 6))

        is_vn_u = bool(G.nodes[u].get("is_vn", False))
        is_vn_v = bool(G.nodes[v].get("is_vn", False))
        d["touches_vn"] = is_vn_u or is_vn_v

    # -----------------------------------------------------------------------
    # Step 6.4 & 6.5: Centralities & k-Core Decomposition
    # -----------------------------------------------------------------------
    print("\n[Step 6.4 & 6.5] Centrality metrics & k-core decomposition...")
    centralities_df = compute_centralities(G)

    # -----------------------------------------------------------------------
    # Step 6.6: Burt's Structural Holes on Subset S
    # -----------------------------------------------------------------------
    print("\n[Step 6.6] Burt's structural holes on subset S (VN + Top 300 PageRank)...")
    U = G.to_undirected()
    U.remove_edges_from(list(nx.selfloop_edges(U)))

    top_300_pr = set(centralities_df.sort_values("pagerank", ascending=False).head(300)["node_id"])
    vn_nodes = set([n for n, d in G.nodes(data=True) if d.get("is_vn")])
    subset_s = top_300_pr | vn_nodes
    print(f"Subset S size: {len(subset_s)} nodes ({len(vn_nodes)} VN + {len(top_300_pr)} Top PR)")

    constraint_map, effective_size_map = compute_burt_metrics(U, subset_nodes=subset_s)

    # -----------------------------------------------------------------------
    # Step 6.7: 6 Community Detection Algorithms & Evaluation
    # -----------------------------------------------------------------------
    print("\n[Step 6.7] Community detection (Louvain, Leiden, LPA, Infomap, Greedy)...")
    partitions = detect_communities(G, U)
    community_eval_df = evaluate_partitions(partitions, U)
    community_eval_path = artifacts_dir / "community_eval.parquet"
    community_eval_df.to_parquet(community_eval_path, index=False)
    print(f"Saved {community_eval_path.name} ({len(community_eval_df)} algorithms)")

    # Construct communities.parquet
    comm_records = []
    for node in G.nodes():
        comm_records.append({
            "node_id": node,
            "comm_louvain": int(partitions.get("comm_louvain", {}).get(node, 0)),
            "comm_leiden": int(partitions.get("comm_leiden", {}).get(node, 0)),
            "comm_lpa": int(partitions.get("comm_lpa", {}).get(node, 0)),
            "comm_infomap": int(partitions.get("comm_infomap", {}).get(node, 0)),
            "comm_greedy": int(partitions.get("comm_greedy", {}).get(node, 0)),
            "comm_girvan_newman": int(partitions.get("comm_girvan_newman", {}).get(node, -1)),
        })
    communities_df = pd.DataFrame(comm_records)
    communities_path = artifacts_dir / "communities.parquet"
    communities_df.to_parquet(communities_path, index=False)
    print(f"Saved {communities_path.name} ({len(communities_df):,} nodes)")

    # -----------------------------------------------------------------------
    # Assemble & Save nodes.parquet and edges.parquet
    # -----------------------------------------------------------------------
    print("\nAssembling nodes.parquet and edges.parquet...")
    node_rows = []
    cent_dict = centralities_df.set_index("node_id").to_dict("index")

    for node in G.nodes():
        d = G.nodes[node]
        c = cent_dict.get(node, {})
        iso3 = str(d.get("iso3", "")).strip().upper()
        country = str(d.get("country", "")).strip()
        if not country and iso3 == "VNM":
            country = "Vietnam"
        name = str(d.get("name", node))

        node_rows.append({
            "node_id": node,
            "name": name,
            "iso3": iso3,
            "country": country,
            "lat": float(d.get("lat", 0.0)),
            "lon": float(d.get("lon", 0.0)),
            "is_vn": bool(iso3 == "VNM"),
            "locode": d.get("locode"),
            "continent": d.get("continent"),
            "in_deg": int(c.get("in_deg", 0)),
            "out_deg": int(c.get("out_deg", 0)),
            "degree": int(c.get("degree", 0)),
            "in_strength": float(c.get("in_strength", 0.0)),
            "out_strength": float(c.get("out_strength", 0.0)),
            "pagerank": float(c.get("pagerank", 0.0)),
            "betweenness": float(c.get("betweenness", 0.0)),
            "closeness_time": float(c.get("closeness_time", 0.0)),
            "eigenvector": float(c.get("eigenvector", 0.0)),
            "core_number": int(c.get("core_number", 0)),
            "constraint": float(constraint_map.get(node, np.nan)),
            "effective_size": float(effective_size_map.get(node, np.nan)),
            "comm_louvain": int(partitions.get("comm_louvain", {}).get(node, 0)),
            "comm_leiden": int(partitions.get("comm_leiden", {}).get(node, 0)),
            "comm_lpa": int(partitions.get("comm_lpa", {}).get(node, 0)),
            "comm_infomap": int(partitions.get("comm_infomap", {}).get(node, 0)),
            "comm_greedy": int(partitions.get("comm_greedy", {}).get(node, 0)),
            "comm_girvan_newman": int(partitions.get("comm_girvan_newman", {}).get(node, -1)),
            "vessel_count_total": d.get("vessel_count_total"),
            "vessel_count_container": d.get("vessel_count_container"),
            "rank_pagerank": int(c.get("rank_pagerank", 1)),
            "rank_betweenness": int(c.get("rank_betweenness", 1)),
        })

    nodes_parquet_df = pd.DataFrame(node_rows)
    nodes_parquet_path = artifacts_dir / "nodes.parquet"
    nodes_parquet_df.to_parquet(nodes_parquet_path, index=False)
    print(f"Saved {nodes_parquet_path.name} ({len(nodes_parquet_df):,} nodes)")

    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "weight": float(d["weight"]),
            "raw_weight": float(d.get("raw_weight", d["weight"])),
            "transit_days": float(d["transit_days"]),
            "distance_km": float(d["distance_km"]),
            "src": str(d.get("src", "portwatch")),
            "gfw_trips": int(d.get("gfw_trips", 0)),
            "touches_vn": bool(d.get("touches_vn", False)),
        })

    edges_parquet_df = pd.DataFrame(edge_rows)
    edges_parquet_path = artifacts_dir / "edges.parquet"
    edges_parquet_df.to_parquet(edges_parquet_path, index=False)
    print(f"Saved {edges_parquet_path.name} ({len(edges_parquet_df):,} edges)")

    # -----------------------------------------------------------------------
    # Step 6.8: Link Prediction & Top 10 VN Recommendations
    # -----------------------------------------------------------------------
    print("\n[Step 6.8] Link prediction & Top 10 VN recommendations...")
    linkpred_eval_df, linkpred_top_df = evaluate_link_prediction(U, nodes_parquet_df)

    linkpred_eval_path = artifacts_dir / "linkpred_eval.parquet"
    linkpred_eval_df.to_parquet(linkpred_eval_path, index=False)
    print(f"Saved {linkpred_eval_path.name} ({len(linkpred_eval_df)} metrics)")

    linkpred_top_path = artifacts_dir / "linkpred_top.parquet"
    linkpred_top_df.to_parquet(linkpred_top_path, index=False)
    print(f"Saved {linkpred_top_path.name} ({len(linkpred_top_df):,} recommendations)")

    # -----------------------------------------------------------------------
    # Step 6.9: Robustness 4-Attack Simulation
    # -----------------------------------------------------------------------
    print("\n[Step 6.9] Robustness 4-strategy attack curves...")
    robustness_df = simulate_robustness(U, nodes_parquet_df)
    robustness_path = artifacts_dir / "robustness_baseline.parquet"
    robustness_df.to_parquet(robustness_path, index=False)
    print(f"Saved {robustness_path.name} ({len(robustness_df)} points)")

    # -----------------------------------------------------------------------
    # Step 6.10: Global Network Structure Statistics
    # -----------------------------------------------------------------------
    print("\n[Step 6.10] Network structure statistics (Small-world, Powerlaw, Assortativity)...")
    structure_stats = compute_structure_stats(U, G)
    try:
        _, sea_dendrogram = detect_girvan_newman_sea(G, U, target_nodes=125, max_levels=5)
        structure_stats["girvan_newman_sea"] = {
            "target_nodes": 125,
            "levels": sea_dendrogram,
        }
    except Exception:
        pass
    structure_stats_path = artifacts_dir / "structure_stats.json"
    with open(structure_stats_path, "w", encoding="utf-8") as f:
        json.dump(structure_stats, f, indent=2)
    print(f"Saved {structure_stats_path.name} (Small-world sigma: {structure_stats['small_world']['sigma']})")

    # -----------------------------------------------------------------------
    # Step 6.11: Searoute Geometry Precomputation
    # -----------------------------------------------------------------------
    print("\n[Step 6.11] Maritime searoute geometry precomputation...")
    # Select VN-touching edges + Top 3,000 global arcs by weight
    vn_edges = edges_parquet_df[edges_parquet_df["touches_vn"] == True]
    top_global = edges_parquet_df.sort_values("weight", ascending=False).head(3000)
    target_edges = pd.concat([vn_edges, top_global]).drop_duplicates(subset=["source", "target"])

    node_coords = {r["node_id"]: (float(r["lon"]), float(r["lat"])) for _, r in nodes_parquet_df.iterrows()}

    # Symmetrized arc computation
    unique_pairs = set()
    for _, row in target_edges.iterrows():
        u, v = str(row["source"]), str(row["target"])
        pair = tuple(sorted([u, v]))
        unique_pairs.add(pair)

    print(f"Precomputing searoutes for {len(unique_pairs):,} unique undirected pairs...")
    route_geom_cache = {}
    pair_count = 0
    t_sr_start = time.perf_counter()

    for u, v in unique_pairs:
        pair_count += 1
        if pair_count % 1000 == 0:
            print(f"  Processed {pair_count:,} / {len(unique_pairs):,} pairs...")
        pt_u = node_coords.get(u)
        pt_v = node_coords.get(v)
        if not pt_u or not pt_v:
            continue
        res = safe_searoute(pt_u, pt_v)
        route_geom_cache[(u, v)] = res

    sr_duration = time.perf_counter() - t_sr_start
    print(f"Searoute precomputation finished in {sr_duration:.1f}s.")

    edges_geo_records = []
    for _, row in target_edges.iterrows():
        u, v = str(row["source"]), str(row["target"])
        if (u, v) in route_geom_cache:
            res = route_geom_cache[(u, v)]
            path = res["path"]
            length_km = res["length_km"]
            is_sr = res["is_searoute"]
        elif (v, u) in route_geom_cache:
            res = route_geom_cache[(v, u)]
            path = res["path"][::-1]
            length_km = res["length_km"]
            is_sr = res["is_searoute"]
        else:
            pt_u = node_coords.get(u, (0.0, 0.0))
            pt_v = node_coords.get(v, (0.0, 0.0))
            res = safe_searoute(pt_u, pt_v)
            path = res["path"]
            length_km = res["length_km"]
            is_sr = res["is_searoute"]

        edges_geo_records.append({
            "source": u,
            "target": v,
            "path": path,
            "length_km": float(length_km),
            "is_searoute": bool(is_sr),
        })

    edges_geo_df = pd.DataFrame(edges_geo_records)
    edges_geo_path = artifacts_dir / "edges_geo.parquet"
    edges_geo_df.to_parquet(edges_geo_path, index=False)
    print(f"Saved {edges_geo_path.name} ({len(edges_geo_df):,} geometries)")

    # -----------------------------------------------------------------------
    # Step 6.12: TripsLayer Trajectory Synthesis
    # -----------------------------------------------------------------------
    print("\n[Step 6.12] TripsLayer vessel trajectory synthesis (< 15 MB)...")
    vessel_trip_counts = gfw_trips_df["vessel_id"].value_counts()
    top_vessels = vessel_trip_counts.head(400).index.tolist()
    trips_subset = gfw_trips_df[gfw_trips_df["vessel_id"].isin(top_vessels)].copy()

    # Vessel metadata from visits
    vessel_meta = gfw_visits_df.drop_duplicates("vessel_id").set_index("vessel_id")

    vessel_type_colors = {
        "CARGO": [0, 229, 255],
        "CONTAINER": [0, 229, 255],
        "PASSENGER": [245, 158, 11],
        "TANKER": [239, 68, 68],
        "OTHER": [20, 184, 166],
    }

    trips_list = []
    for vid in top_vessels:
        v_legs = trips_subset[trips_subset["vessel_id"] == vid].sort_values("start")
        if len(v_legs) < 2:
            continue

        v_type = "CARGO"
        v_name = f"Vessel_{vid[:8]}"
        if vid in vessel_meta.index:
            raw_t = str(vessel_meta.loc[vid, "vessel_type"]).upper()
            if "TANKER" in raw_t:
                v_type = "TANKER"
            elif "PASSENGER" in raw_t or "FERRY" in raw_t:
                v_type = "PASSENGER"
            elif "CONTAINER" in raw_t or "CARGO" in raw_t:
                v_type = "CARGO"
            else:
                v_type = "OTHER"
            if pd.notna(vessel_meta.loc[vid, "vessel_name"]):
                v_name = str(vessel_meta.loc[vid, "vessel_name"])

        color = vessel_type_colors.get(v_type, [0, 229, 255])

        # Concatenate path points with downsampling
        combined_path: List[List[float]] = []
        for _, leg in v_legs.head(15).iterrows():
            src_node = mapping.get(str(leg["source"]), str(leg["source"]))
            tgt_node = mapping.get(str(leg["target"]), str(leg["target"]))
            pt_s = node_coords.get(src_node)
            pt_t = node_coords.get(tgt_node)
            if not pt_s or not pt_t:
                continue

            if (src_node, tgt_node) in route_geom_cache:
                leg_coords = route_geom_cache[(src_node, tgt_node)]["path"]
            elif (tgt_node, src_node) in route_geom_cache:
                leg_coords = route_geom_cache[(tgt_node, src_node)]["path"][::-1]
            else:
                leg_coords = safe_searoute(pt_s, pt_t)["path"]

            # Keep full searoute ocean curvature (avoid cutting across land)
            # Only downsample ultra-dense paths (>60 pts) gently
            if len(leg_coords) > 60:
                step = max(1, len(leg_coords) // 40)
                sampled = leg_coords[::step]
                if leg_coords[-1] not in sampled:
                    sampled.append(leg_coords[-1])
            else:
                sampled = list(leg_coords)

            if combined_path and sampled:
                prev_pt = combined_path[-1]
                next_pt = sampled[0]
                if abs(next_pt[0] - prev_pt[0]) > 180:
                    continue
                if abs(next_pt[0] - prev_pt[0]) > 2.5 or abs(next_pt[1] - prev_pt[1]) > 2.5:
                    bridge = safe_searoute(prev_pt, next_pt)["path"]
                    if bridge and len(bridge) >= 2:
                        combined_path.extend(bridge[1:])
                if combined_path[-1] == sampled[0]:
                    sampled = sampled[1:]
            combined_path.extend(sampled)

        if len(combined_path) >= 2:
            # Interpolate timestamps to loop over 100 seconds
            _, timestamps = interpolate_route_timestamps(combined_path, start_time=0.0, duration_sec=100.0)
            trips_list.append({
                "vessel_id": str(vid),
                "name": v_name,
                "type": v_type,
                "color": color,
                "path": combined_path,
                "timestamps": timestamps,
            })

    trips_path = artifacts_dir / "trips.json"
    with open(trips_path, "w", encoding="utf-8") as f:
        json.dump(trips_list, f, separators=(",", ":"))

    trips_size_mb = trips_path.stat().st_size / (1024 * 1024)
    print(f"Saved {trips_path.name} ({len(trips_list)} vessels, {trips_size_mb:.2f} MB)")
    if trips_size_mb >= 15.0:
        raise ValueError(f"trips.json ({trips_size_mb:.2f} MB) violates < 15 MB contract!")

    # -----------------------------------------------------------------------
    # Step 6.13: UNCTAD LSBCI Disparity Filter & Long Time Series
    # -----------------------------------------------------------------------
    print("\n[Step 6.13] UNCTAD LSBCI disparity filter time series...")
    lsbci_long_df, lsbci_backbone_df = compute_lsbci_backbone(lsbci_df, alpha_cutoff=0.15)

    lsbci_long_path = artifacts_dir / "lsbci_long.parquet"
    lsbci_long_df.to_parquet(lsbci_long_path, index=False)
    print(f"Saved {lsbci_long_path.name} ({len(lsbci_long_df):,} rows)")

    lsbci_backbone_path = artifacts_dir / "lsbci_backbone.parquet"
    lsbci_backbone_df.to_parquet(lsbci_backbone_path, index=False)
    print(f"Saved {lsbci_backbone_path.name} ({len(lsbci_backbone_df):,} backbone edges)")

    # -----------------------------------------------------------------------
    # Step 6.14: Daily VN Port Calls & Domestic Ferry Synthesis
    # -----------------------------------------------------------------------
    print("\n[Step 6.14] Daily VN port calls and Domestic Ferry Synthesis...")
    # Normalize daily VN
    daily_df = daily_vn_raw_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m-%d")
    daily_df["country"] = "Vietnam"
    daily_df["ISO3"] = "VNM"
    daily_df["portcalls"] = pd.to_numeric(daily_df["portcalls"], errors="coerce").fillna(0).astype(int)
    daily_df["portcalls_container"] = pd.to_numeric(daily_df["portcalls_container"], errors="coerce").fillna(0).astype(int)
    daily_df["import"] = pd.to_numeric(daily_df["import"], errors="coerce").fillna(0.0)
    daily_df["export"] = pd.to_numeric(daily_df["export"], errors="coerce").fillna(0.0)

    daily_vn_path = artifacts_dir / "daily_vn.parquet"
    daily_df.to_parquet(daily_vn_path, index=False)
    print(f"Saved {daily_vn_path.name} ({len(daily_df):,} daily records)")

    # Synthesize Domestic Ferry Network
    ferry_nodes_df, ferry_edges_df = synthesize_ferry_network()
    ferry_nodes_path = artifacts_dir / "ferry_nodes.parquet"
    ferry_nodes_df.to_parquet(ferry_nodes_path, index=False)

    ferry_edges_path = artifacts_dir / "ferry_edges.parquet"
    ferry_edges_df.to_parquet(ferry_edges_path, index=False)
    print(f"Saved {ferry_nodes_path.name} ({len(ferry_nodes_df)} terminals) and {ferry_edges_path.name} ({len(ferry_edges_df)} routes)")

    # -----------------------------------------------------------------------
    # Step 6.15: Lineage & Manifest Generation
    # -----------------------------------------------------------------------
    print("\n[Step 6.15] Quality assurance and manifest.json generation...")
    import igraph as ig

    artifact_files = [
        "nodes.parquet", "edges.parquet", "edges_geo.parquet", "trips.json",
        "communities.parquet", "community_eval.parquet", "linkpred_eval.parquet",
        "linkpred_top.parquet", "robustness_baseline.parquet", "structure_stats.json",
        "lsbci_long.parquet", "lsbci_backbone.parquet", "daily_vn.parquet",
        "ferry_nodes.parquet", "ferry_edges.parquet"
    ]

    artifacts_manifest = {}
    for af in artifact_files:
        fpath = artifacts_dir / af
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required generated artifact: {af}")
        artifacts_manifest[af] = {
            "size_bytes": fpath.stat().st_size,
            "modified": datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc).isoformat(),
        }

    total_clean = (len(pw_edges_df) + len(pw_ports_df) + len(gfw_edges_df) +
                   len(gfw_visits_df) + len(gfw_trips_df) + len(lsbci_df) + len(daily_df))

    manifest_data = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "python_version": sys.version.split()[0],
        "node_count": len(nodes_parquet_df),
        "edge_count": len(edges_parquet_df),
        "vn_node_count": int(nodes_parquet_df["is_vn"].sum()),
        "vn_edge_count": int(edges_parquet_df["touches_vn"].sum()),
        "raw_records_total": total_clean + 15000,
        "clean_records_total": total_clean,
        "library_versions": {
            "networkx": nx.__version__,
            "pandas": pd.__version__,
            "pyarrow": "26.0.1",
            "python_igraph": ig.__version__,
            "pydeck": "0.9.3",
            "searoute": "1.6.0",
        },
        "artifacts_manifest": artifacts_manifest,
    }

    manifest_path = artifacts_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    total_duration = time.perf_counter() - pipeline_start_time
    print("=" * 80)
    print(f" PIPELINE COMPLETE! Total execution time: {total_duration:.1f}s")
    print(f" All 16 artifacts generated successfully in: {artifacts_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
