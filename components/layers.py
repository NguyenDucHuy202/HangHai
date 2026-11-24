"""Pydeck / Deck.gl Layer Factory for Maritime Transport SNA Application.

Provides modular, high-performance visual encodings adhering to Nautical Light Mode:
- ports_layer: ScatterplotLayer with radius scaled by centrality, VN red stroke.
- ports_glow_layer: ScatterplotLayer Top 30 ports halo.
- vn_stars_layer: Golden star markers pinned on Vietnamese ports.
- arcs_layer: Great-circle ArcLayer with flow gradients and weight scaling.
- searoute_layer: PathLayer rendering realistic bathymetric waterways from edges_geo.
- chokepoints_layer: Ring ScatterplotLayer + TextLayer for 7 strategic chokepoints.
- seamark_layer: OpenSeaMap TileLayer.
- suggested_routes_layer: Amber ArcLayer for link prediction candidate routes.
- disrupted_layer: Red alert ScatterplotLayer for severed and spillover ports.
- selected_route_layer: Highlighted navigation path and waypoint markers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pydeck as pdk

from core.theme import (
    CHOKEPOINTS,
    COMMUNITY_PALETTE_RGB,
    RGB_AMBER,
    RGB_DISRUPTED,
    RGB_PRIMARY_BLUE,
    RGB_PRIMARY_DARK,
    RGB_STAR,
    RGB_TEAL,
    RGB_VN_RED,
    get_community_color,
    hex_to_rgb,
)

# Universal Vietnamese Unicode Character Set for Deck.gl GPU Font Atlas
VIETNAMESE_DECK_CHARSET: List[str] = list(set(
    " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
    ".,-_/()@#:?!★☆—&"
))


# ===========================================================================
# 1. Ports Layer (ScatterplotLayer)
# ===========================================================================

def ports_layer(
    nodes_df: pd.DataFrame,
    metric_col: str = "pagerank",
    color_by: str = "metric",  # "metric" | "community" | "is_vn"
    radius_scale: float = 1.0,
    selected_id: Optional[str] = None,
    selected_port_id: Optional[str] = None,
) -> pdk.Layer:
    """Builds a Deck.gl ScatterplotLayer representing ports.

    Radius is scaled proportional to the square root of metric_col.
    Vietnamese ports have thick Vietnam-Red borders.
    """
    if nodes_df is None or len(nodes_df) == 0:
        return pdk.Layer("ScatterplotLayer", id="ports", data=[], get_position=["lon", "lat"])

    df = nodes_df.dropna(subset=["lat", "lon"]).copy()
    if len(df) == 0:
        return pdk.Layer("ScatterplotLayer", id="ports", data=[], get_position=["lon", "lat"])

    target_selected = selected_id or selected_port_id

    # Safe extraction of is_vn Series
    is_vn_series = df["is_vn"] if "is_vn" in df.columns else pd.Series(False, index=df.index)

    # Normalize metric for radius calculation
    if metric_col not in df.columns:
        for fallback in ["pagerank", "betweenness", "degree", "in_deg"]:
            if fallback in df.columns:
                metric_col = fallback
                break

    if metric_col in df.columns:
        vals = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)
        max_val = vals.max()
        norm_metric = (vals / max_val) if max_val > 0 else vals
    else:
        norm_metric = pd.Series(0.5, index=df.index)

    # Compute radius: R = (R_base + R_scale * sqrt(norm)) * radius_scale
    base_r = 3000.0 * radius_scale
    scale_r = 45000.0 * radius_scale
    df["deck_radius"] = base_r + scale_r * np.sqrt(norm_metric.clip(lower=0.0))

    # Boost selected port radius if matched
    if target_selected and "node_id" in df.columns:
        df.loc[df["node_id"] == target_selected, "deck_radius"] *= 2.2

    # Compute fill colors
    if color_by == "community" and "comm_louvain" in df.columns:
        comm_ids = df["comm_louvain"].fillna(0).astype(int).tolist()
        num_colors = len(COMMUNITY_PALETTE_RGB)
        df["deck_fill_color"] = [
            COMMUNITY_PALETTE_RGB[cid % num_colors] + [220] for cid in comm_ids
        ]
    elif color_by == "is_vn":
        df["deck_fill_color"] = [
            RGB_VN_RED + [235] if bool(v) else RGB_PRIMARY_BLUE + [180]
            for v in is_vn_series
        ]
    else:  # "metric"
        # Oceanic gradient: light blue [2, 132, 199] -> dark maritime navy [3, 4, 94]
        # Vietnamese ports get reddish-amber tint
        fill_colors: List[List[int]] = []
        for is_vn_val, norm_val in zip(is_vn_series, norm_metric):
            if bool(is_vn_val):
                fill_colors.append(RGB_VN_RED + [235])
            else:
                # Interpolate blue intensity based on norm_val
                r = int(2 + (3 - 2) * norm_val)
                g = int(132 + (4 - 132) * norm_val)
                b = int(199 + (94 - 199) * norm_val)
                fill_colors.append([max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), 210])
        df["deck_fill_color"] = fill_colors

    # Compute stroke / border colors: VN ports get red border, others white
    df["deck_line_color"] = [
        RGB_VN_RED + [255] if bool(v) else [255, 255, 255, 220]
        for v in is_vn_series
    ]
    df["deck_line_width"] = [
        3 if bool(v) else 1.5
        for v in is_vn_series
    ]

    # Format tooltip display fields to prevent raw template leak
    df["country"] = df["country"].fillna("Quốc tế").astype(str) if "country" in df.columns else "Cảng biển"
    df["iso3"] = df["iso3"].fillna("").astype(str) if "iso3" in df.columns else ""
    df["pagerank"] = df["pagerank"].apply(lambda v: f"PageRank: {float(v):.5f}" if pd.notna(v) and str(v) != "N/A" else "PageRank: N/A") if "pagerank" in df.columns else "PageRank: N/A"
    df["degree"] = df["degree"].apply(lambda v: f"Bậc kết nối: {int(v):,} cảng" if pd.notna(v) and str(v) != "N/A" else "Bậc kết nối: N/A") if "degree" in df.columns else "Bậc kết nối: N/A"

    return pdk.Layer(
        "ScatterplotLayer",
        id="ports",
        data=df,
        get_position=["lon", "lat"],
        get_radius="deck_radius",
        get_fill_color="deck_fill_color",
        get_line_color="deck_line_color",
        get_line_width="deck_line_width",
        stroked=True,
        filled=True,
        line_width_min_pixels=1,
        radius_min_pixels=2,
        radius_max_pixels=60,
        pickable=True,
        auto_highlight=True,
    )


# ===========================================================================
# 2. Ports Glow Layer (Top 30 Hubs Halo)
# ===========================================================================

def ports_glow_layer(nodes_df: pd.DataFrame, top_n: int = 30) -> pdk.Layer:
    """Builds a halo ScatterplotLayer around top N maritime hubs (opacity 0.15)."""
    if nodes_df is None or len(nodes_df) == 0:
        return pdk.Layer("ScatterplotLayer", id="ports_glow", data=[], get_position=["lon", "lat"])

    df = nodes_df.dropna(subset=["lat", "lon"]).copy()
    if len(df) == 0:
        return pdk.Layer("ScatterplotLayer", id="ports_glow", data=[], get_position=["lon", "lat"])
    sort_col = "pagerank" if "pagerank" in df.columns else ("degree" if "degree" in df.columns else df.columns[0])
    top_df = df.sort_values(by=sort_col, ascending=False).head(top_n).copy()

    # Radius is 3x standard hub radius
    top_df["glow_radius"] = 80000.0

    return pdk.Layer(
        "ScatterplotLayer",
        id="ports_glow",
        data=top_df,
        get_position=["lon", "lat"],
        get_radius="glow_radius",
        get_fill_color=[0, 119, 182, 38],  # 15% opacity oceanic blue halo
        stroked=False,
        filled=True,
        pickable=False,
    )


# ===========================================================================
# 3. Vietnam Stars Layer (Pinned Gold Star Markers)
# ===========================================================================

def vn_stars_layer(nodes_df: pd.DataFrame) -> pdk.Layer:
    """Builds a pinned top layer rendering golden yellow markers on all VN ports."""
    if nodes_df is None or len(nodes_df) == 0 or "is_vn" not in nodes_df.columns:
        return pdk.Layer("ScatterplotLayer", id="vn_stars", data=[], get_position=["lon", "lat"])

    vn_df = nodes_df[nodes_df["is_vn"] == True].dropna(subset=["lat", "lon"]).copy()
    if len(vn_df) == 0:
        return pdk.Layer("ScatterplotLayer", id="vn_stars", data=[], get_position=["lon", "lat"])

    vn_df["star_radius"] = 9000.0
    vn_df["country"] = "Việt Nam"
    vn_df["iso3"] = "VNM 🇻🇳"
    if "pagerank" in vn_df.columns:
        vn_df["pagerank"] = vn_df["pagerank"].apply(lambda v: f"PageRank: {float(v):.5f}" if pd.notna(v) and str(v) != "N/A" else "PageRank: Cảng VN")
    else:
        vn_df["pagerank"] = "Cảng biển Việt Nam"
    if "degree" in vn_df.columns:
        vn_df["degree"] = vn_df["degree"].apply(lambda v: f"Bậc kết nối: {int(v):,} cảng" if pd.notna(v) and str(v) != "N/A" else "Bậc kết nối: N/A")
    else:
        vn_df["degree"] = "Hệ sinh thái hàng hải VN"

    return pdk.Layer(
        "ScatterplotLayer",
        id="vn_stars",
        data=vn_df,
        get_position=["lon", "lat"],
        get_radius="star_radius",
        get_fill_color=RGB_STAR + [255],
        get_line_color=RGB_VN_RED + [255],
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        radius_min_pixels=3,
        pickable=True,
    )


# ===========================================================================
# 4. Shipping Arcs Layer (ArcLayer)
# ===========================================================================

def arcs_layer(
    edges_df: pd.DataFrame,
    limit: int = 5000,
    vn_only: bool = False,
    nodes_df: Optional[pd.DataFrame] = None,
) -> pdk.Layer:
    """Builds a great-circle ArcLayer representing direct maritime shipping connections."""
    if edges_df is None or len(edges_df) == 0:
        return pdk.Layer(
            "ArcLayer",
            id="arcs",
            data=[],
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
        )

    df = edges_df.copy()
    if vn_only and "touches_vn" in df.columns:
        df = df[df["touches_vn"] == True]

    # Enforce safe upper bound to prevent WebGL GPU context overflow
    clamped_limit = min(max(10, limit), 20000)
    df = df.head(clamped_limit).copy()

    # Merge or map coordinates if not already present
    if "src_lon" not in df.columns or "dst_lon" not in df.columns:
        if nodes_df is None:
            from core.data import load_nodes
            nodes_df = load_nodes()
        coords_map = nodes_df.set_index("node_id")[["lon", "lat"]].to_dict("index")
        df["src_lon"] = df["source"].map(lambda x: coords_map.get(str(x), {}).get("lon"))
        df["src_lat"] = df["source"].map(lambda x: coords_map.get(str(x), {}).get("lat"))
        df["dst_lon"] = df["target"].map(lambda x: coords_map.get(str(x), {}).get("lon"))
        df["dst_lat"] = df["target"].map(lambda x: coords_map.get(str(x), {}).get("lat"))
        df = df.dropna(subset=["src_lon", "src_lat", "dst_lon", "dst_lat"]).copy()

    if len(df) == 0:
        return pdk.Layer(
            "ArcLayer",
            id="arcs",
            data=[],
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
        )

    # Width scaling
    if "weight" in df.columns:
        w_vals = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
        max_w = w_vals.max()
        df["deck_width"] = 1.0 + (3.0 * (w_vals / max_w)) if max_w > 0 else 1.5
    else:
        df["deck_width"] = 1.5

    # Color gradient: VN touching edges colored [VN_RED -> Oceanic Blue]
    # Normal edges colored [Oceanic Blue -> Sky Blue]
    touches_vn_series = df["touches_vn"] if "touches_vn" in df.columns else pd.Series(False, index=df.index)
    df["deck_src_color"] = [
        RGB_VN_RED + [190] if bool(v) else RGB_PRIMARY_BLUE + [120]
        for v in touches_vn_series
    ]
    df["deck_dst_color"] = [
        [2, 132, 199, 180] if bool(v) else [14, 165, 233, 100]
        for v in touches_vn_series
    ]

    # Format tooltip display fields to prevent raw template leak
    src_col = df["source_name"] if "source_name" in df.columns else df["source"]
    dst_col = df["target_name"] if "target_name" in df.columns else df["target"]
    df["name"] = src_col.astype(str) + " → " + dst_col.astype(str)
    df["country"] = "Hành lang luồng hàng hải"
    if "weight" in df.columns:
        df["iso3"] = df["weight"].apply(lambda w: f"(Trọng số: {float(w):,.0f})" if pd.notna(w) else "")
    else:
        df["iso3"] = ""
    if "distance_km" in df.columns:
        df["pagerank"] = df["distance_km"].apply(lambda d: f"Cự ly: {float(d):,.0f} km" if pd.notna(d) else "Hải trình trực tiếp")
    else:
        df["pagerank"] = "Tuyến kết nối đại vòng (Great Circle)"
    if "transit_days" in df.columns:
        df["degree"] = df["transit_days"].apply(lambda t: f"Thời gian: {float(t):.1f} ngày hải trình" if pd.notna(t) else "Luồng chính tuyến")
    else:
        df["degree"] = "Hải trình trực tiếp"

    return pdk.Layer(
        "ArcLayer",
        id="arcs",
        data=df,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dst_lon", "dst_lat"],
        get_source_color="deck_src_color",
        get_target_color="deck_dst_color",
        get_width="deck_width",
        width_min_pixels=1,
        great_circle=True,
        pickable=True,
        auto_highlight=True,
    )


# ===========================================================================
# 5. Maritime Searoutes Layer (PathLayer)
# ===========================================================================

def searoute_layer(
    edges_geo_df: pd.DataFrame,
    color: Optional[List[int]] = None,
    width: int = 2,
    top_n: Optional[int] = None,
) -> pdk.Layer:
    """Builds a Deck.gl PathLayer representing real bathymetric searoutes from edges_geo.parquet."""
    if edges_geo_df is None or len(edges_geo_df) == 0:
        return pdk.Layer("PathLayer", id="searoutes", data=[], get_path="path")

    df = edges_geo_df.dropna(subset=["path"]).copy()
    if top_n and top_n > 0:
        df = df.head(top_n)

    route_color = color or [0, 119, 182, 160]

    # Format tooltip display fields to prevent raw template leak
    if "name" not in df.columns:
        df["name"] = df["source"].astype(str) + " → " + df["target"].astype(str)
    df["country"] = "Tuyến hải hành biển thật (Searoute)"
    df["iso3"] = ""
    if "length_km" in df.columns:
        df["pagerank"] = df["length_km"].apply(lambda d: f"Cự ly: {float(d):,.0f} km" if pd.notna(d) else "Luồng biển tự nhiên")
    else:
        df["pagerank"] = "Luồng biển sâu tự nhiên"
    df["degree"] = "Bathymetric Searoute (Men theo luồng hàng hải an toàn)"

    return pdk.Layer(
        "PathLayer",
        id="searoutes",
        data=df,
        get_path="path",
        get_color=route_color,
        get_width=width,
        width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )


# ===========================================================================
# 6. Strategic Chokepoints Layer (Rings + Labels)
# ===========================================================================

def chokepoints_layer() -> List[pdk.Layer]:
    """Builds a composite layer list (rings + text labels) for the 7 strategic maritime chokepoints."""
    choke_df = pd.DataFrame(CHOKEPOINTS).copy()
    choke_df["country"] = "Hải trình huyết mạch thế giới"
    choke_df["iso3"] = choke_df["en_name"].apply(lambda x: f"({x})" if x else "")
    choke_df["pagerank"] = choke_df["traffic_share"].apply(lambda x: f"Lưu lượng: {x}")
    choke_df["degree"] = choke_df["desc"]

    rings_layer = pdk.Layer(
        "ScatterplotLayer",
        id="chokepoint_rings",
        data=choke_df,
        get_position=["lon", "lat"],
        get_radius=40000,
        stroked=True,
        filled=False,
        get_line_color=RGB_AMBER + [230],
        line_width_min_pixels=2,
        pickable=True,
    )

    labels_layer = pdk.Layer(
        "TextLayer",
        id="chokepoint_labels",
        data=choke_df,
        get_position=["lon", "lat"],
        get_text="name",
        get_color=[15, 23, 42, 255],
        get_size=13,
        get_alignment_baseline="'bottom'",
        get_pixel_offset=[0, -14],
        character_set=VIETNAMESE_DECK_CHARSET,
        font_family="'Arial', 'Segoe UI', sans-serif",
        billboard=True,
        background_color=[255, 255, 255, 230],
        background_padding=[5, 3],
        pickable=False,
    )

    return [rings_layer, labels_layer]


# ===========================================================================
# 7. OpenSeaMap Seamark Layer & Satellite Layer (TileLayer)
# ===========================================================================

def seamark_layer() -> pdk.Layer:
    """Builds a Deck.gl TileLayer fetching OpenSeaMap seamark navigation tiles."""
    return pdk.Layer(
        "TileLayer",
        id="seamarks",
        data="https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png",
        min_zoom=0,
        max_zoom=18,
        tile_size=256,
        opacity=0.85,
    )


def satellite_layer() -> pdk.Layer:
    """Builds a Deck.gl TileLayer fetching global high-resolution satellite imagery (ESRI World Imagery).
    Provides pristine photorealistic terrain and ocean basemap, token-free and label-free.
    """
    return pdk.Layer(
        "TileLayer",
        id="satellite_base_tiles",
        data="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        min_zoom=0,
        max_zoom=19,
        tile_size=256,
        pickable=False,
    )


# ===========================================================================
# 8. Suggested Routes Layer (Link Prediction Recommendations)
# ===========================================================================

def suggested_routes_layer(
    candidate_edges_df: pd.DataFrame,
    nodes_df: Optional[pd.DataFrame] = None,
) -> pdk.Layer:
    """Builds an ArcLayer for link prediction candidate routes in distinct Amber color."""
    if candidate_edges_df is None or len(candidate_edges_df) == 0:
        return pdk.Layer(
            "ArcLayer",
            id="suggested_routes",
            data=[],
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
        )

    df = candidate_edges_df.copy()
    src_col = "source" if "source" in df.columns else ("vn_node_id" if "vn_node_id" in df.columns else None)
    dst_col = "target" if "target" in df.columns else ("candidate_node_id" if "candidate_node_id" in df.columns else None)

    if "src_lon" not in df.columns or "dst_lon" not in df.columns:
        if nodes_df is None:
            from core.data import load_nodes
            nodes_df = load_nodes()
        coords_map = nodes_df.set_index("node_id")[["lon", "lat"]].to_dict("index")
        if src_col and dst_col:
            df["src_lon"] = df[src_col].map(lambda x: coords_map.get(str(x), {}).get("lon"))
            df["src_lat"] = df[src_col].map(lambda x: coords_map.get(str(x), {}).get("lat"))
            df["dst_lon"] = df[dst_col].map(lambda x: coords_map.get(str(x), {}).get("lon"))
            df["dst_lat"] = df[dst_col].map(lambda x: coords_map.get(str(x), {}).get("lat"))
            df = df.dropna(subset=["src_lon", "src_lat", "dst_lon", "dst_lat"]).copy()

    # Format tooltip display fields to prevent raw template leak
    if "name" not in df.columns:
        c_name = df["candidate_name"] if "candidate_name" in df.columns else (df[dst_col] if dst_col else "Cảng đích")
        df["name"] = "Tuyến đề xuất → " + c_name.astype(str)
    df["country"] = "Gợi ý liên kết tiềm năng (AI)"
    if "candidate_country" in df.columns:
        df["iso3"] = df["candidate_country"].apply(lambda c: f"({c})" if pd.notna(c) else "")
    else:
        df["iso3"] = ""
    if "score" in df.columns:
        df["pagerank"] = df["score"].apply(lambda s: f"Điểm liên kết: {float(s):.4f}" if pd.notna(s) else "Điểm số cao")
    else:
        df["pagerank"] = "Link Prediction"
    if "searoute_distance_km" in df.columns:
        df["degree"] = df["searoute_distance_km"].apply(lambda d: f"Cự ly: {float(d):,.0f} km" if pd.notna(d) else "Tuyến thẳng")
    else:
        df["degree"] = "Tuyến đề xuất mở rộng"

    return pdk.Layer(
        "ArcLayer",
        id="suggested_routes",
        data=df,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dst_lon", "dst_lat"],
        get_source_color=RGB_AMBER + [240],
        get_target_color=[245, 158, 11, 180],
        get_width=3,
        width_min_pixels=2,
        great_circle=True,
        pickable=True,
        auto_highlight=True,
    )


# ===========================================================================
# 9. Disrupted / Spillover Layer (Red Alert ScatterplotLayer)
# ===========================================================================

def disrupted_layer(
    disrupted_nodes_df: pd.DataFrame,
    mode: str = "severed",  # "severed" | "spillover"
) -> pdk.Layer:
    """Builds a ScatterplotLayer displaying severed or spillover-impacted ports."""
    if disrupted_nodes_df is None or len(disrupted_nodes_df) == 0:
        return pdk.Layer("ScatterplotLayer", id="disrupted_ports", data=[], get_position=["lon", "lat"])

    df = disrupted_nodes_df.dropna(subset=["lat", "lon"]).copy()
    if len(df) == 0:
        return pdk.Layer("ScatterplotLayer", id="disrupted_ports", data=[], get_position=["lon", "lat"])

    if mode == "spillover":
        color = RGB_AMBER + [220]
        radius = 20000.0
    else:  # "severed"
        color = RGB_DISRUPTED + [230]
        radius = 35000.0

    # Format tooltip display fields to prevent raw template leak
    if "name" not in df.columns:
        df["name"] = "Cảng bị ảnh hưởng"
    if "country" not in df.columns:
        df["country"] = "Khu vực gián đoạn sự cố"
    if "iso3" not in df.columns:
        df["iso3"] = "⚠️"
    if "pagerank" not in df.columns:
        df["pagerank"] = "Trạng thái: Gián đoạn kết nối" if mode == "severed" else "Trạng thái: Chịu tải lan truyền (Spillover)"
    if "degree" not in df.columns:
        df["degree"] = "Mất tuyến kết nối trực tiếp" if mode == "severed" else "Nguy cơ ùn ứ & quá tải"

    return pdk.Layer(
        "ScatterplotLayer",
        id="disrupted_ports",
        data=df,
        get_position=["lon", "lat"],
        get_radius=radius,
        get_fill_color=color,
        get_line_color=[185, 28, 28, 255],
        stroked=True,
        filled=True,
        line_width_min_pixels=3,
        pickable=True,
    )


# ===========================================================================
# 10. Selected Navigation Route & Waypoints Layer
# ===========================================================================

def selected_route_layer(
    path_coords: List[List[float]],
    waypoints: Optional[List[Dict[str, Any]]] = None,
) -> List[pdk.Layer]:
    """Builds PathLayer and ScatterplotLayer highlighting a selected route and its ports."""
    layers: List[pdk.Layer] = []

    if path_coords and len(path_coords) >= 2:
        route_df = pd.DataFrame([{"path": path_coords}])
        path_layer = pdk.Layer(
            "PathLayer",
            id="active_selected_route",
            data=route_df,
            get_path="path",
            get_color=[0, 119, 182, 230],
            get_width=4,
            width_min_pixels=3,
            pickable=False,
        )
        layers.append(path_layer)

    if waypoints and len(waypoints) > 0:
        wp_df = pd.DataFrame(waypoints).dropna(subset=["lat", "lon"]).copy()
        if "name" not in wp_df.columns:
            wp_df["name"] = "Điểm hành trình"
        if "country" not in wp_df.columns:
            wp_df["country"] = "Hải trình điều hướng"
        if "iso3" not in wp_df.columns:
            wp_df["iso3"] = "📍"
        if "pagerank" not in wp_df.columns:
            wp_df["pagerank"] = "Chặng đường biển"
        if "degree" not in wp_df.columns:
            wp_df["degree"] = "Điểm chuyển tiếp / ghé cảng"
        wp_layer = pdk.Layer(
            "ScatterplotLayer",
            id="active_route_waypoints",
            data=wp_df,
            get_position=["lon", "lat"],
            get_radius=12000,
            get_fill_color=[245, 158, 11, 240],
            get_line_color=[0, 119, 182, 255],
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            pickable=True,
        )
        layers.append(wp_layer)

    return layers


# ===========================================================================
# 11. Vietnam Maritime Sovereignty Layers (Hoàng Sa, Trường Sa & Biển Đông)
# ===========================================================================

def vietnam_sovereignty_layers() -> List[pdk.Layer]:
    """Builds authoritative layers asserting Vietnam's maritime territorial sovereignty:
    - Quần đảo Hoàng Sa (Thuộc TP. Đà Nẵng, Việt Nam)
    - Quần đảo Trường Sa (Thuộc Tỉnh Khánh Hòa, Việt Nam)
    - Biển Đông (Vùng biển chủ quyền Việt Nam)
    - Đảo Phú Quốc & Côn Đảo
    Includes glowing sovereignty badges, island location markers, and high-contrast labels.
    """
    # 1. Glow circles around major archipelagos
    glow_data = pd.DataFrame([
        {"lon": 112.00, "lat": 16.50, "radius": 45000, "name": "Quần đảo Hoàng Sa (Việt Nam)"},
        {"lon": 114.20, "lat": 9.90, "radius": 65000, "name": "Quần đảo Trường Sa (Việt Nam)"},
    ])
    glow_layer = pdk.Layer(
        "ScatterplotLayer",
        id="vn_sov_archipelago_glow",
        data=glow_data,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color=[218, 37, 29, 45],
        get_line_color=[218, 37, 29, 180],
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        pickable=False,
    )

    # 2. Island & Archipelago Core Points
    islands_data = pd.DataFrame([
        # Hoàng Sa
        {"name": "Quần đảo Hoàng Sa", "country": "Chủ quyền Việt Nam (TP. Đà Nẵng)", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 112.00, "lat": 16.50, "radius": 22000},
        {"name": "Đảo Hoàng Sa", "country": "Quần đảo Hoàng Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 111.60, "lat": 16.53, "radius": 12000},
        {"name": "Đảo Phú Lâm", "country": "Quần đảo Hoàng Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 112.33, "lat": 16.83, "radius": 12000},
        {"name": "Đảo Tri Tôn", "country": "Quần đảo Hoàng Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 111.20, "lat": 15.78, "radius": 12000},
        # Trường Sa
        {"name": "Quần đảo Trường Sa", "country": "Chủ quyền Việt Nam (Tỉnh Khánh Hòa)", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 114.20, "lat": 9.90, "radius": 25000},
        {"name": "Đảo Trường Sa Lớn", "country": "Quần đảo Trường Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 111.92, "lat": 8.64, "radius": 14000},
        {"name": "Đảo Song Tử Tây", "country": "Quần đảo Trường Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 114.33, "lat": 11.43, "radius": 12000},
        {"name": "Đảo Sinh Tồn", "country": "Quần đảo Trường Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 114.32, "lat": 9.88, "radius": 12000},
        {"name": "Đảo Nam Yết", "country": "Quần đảo Trường Sa, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 114.36, "lat": 10.18, "radius": 12000},
        # Đảo ven bờ quan trọng
        {"name": "Côn Đảo", "country": "Bà Rịa - Vũng Tàu, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 106.60, "lat": 8.68, "radius": 15000},
        {"name": "Đảo Phú Quốc", "country": "Tỉnh Kiên Giang, Việt Nam", "iso3": "🇻🇳", "pagerank": "Lãnh thổ thiêng liêng", "degree": "Chủ quyền bất khả xâm phạm", "lon": 103.96, "lat": 10.25, "radius": 18000},
    ])
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        id="vn_sov_islands_core",
        data=islands_data,
        get_position=["lon", "lat"],
        get_radius="radius",
        get_fill_color=[255, 215, 0, 255],
        get_line_color=[218, 37, 29, 255],
        stroked=True,
        filled=True,
        line_width_min_pixels=2.5,
        radius_min_pixels=4,
        pickable=True,
    )

    # 3. Main Text Labels (High Visibility with background box)
    islands_labels_data = pd.DataFrame([
        {"text": "★ QUẦN ĐẢO HOÀNG SA (VIỆT NAM)", "lon": 112.00, "lat": 16.50, "size": 16, "color": [220, 38, 38, 255]},
        {"text": "★ QUẦN ĐẢO TRƯỜNG SA (VIỆT NAM)", "lon": 114.20, "lat": 9.90, "size": 16, "color": [220, 38, 38, 255]},
        {"text": "Đảo Phú Quốc (Việt Nam)", "lon": 103.96, "lat": 10.25, "size": 13, "color": [220, 38, 38, 255]},
        {"text": "Côn Đảo (Việt Nam)", "lon": 106.60, "lat": 8.68, "size": 13, "color": [220, 38, 38, 255]},
    ])
    islands_text_layer = pdk.Layer(
        "TextLayer",
        id="vn_sov_island_labels",
        data=islands_labels_data,
        get_position=["lon", "lat"],
        get_text="text",
        get_color="color",
        get_size="size",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        get_pixel_offset=[0, -18],
        character_set=VIETNAMESE_DECK_CHARSET,
        font_family="'Arial', 'Segoe UI', sans-serif",
        billboard=True,
        background_color=[255, 255, 255, 245],
        background_padding=[8, 5],
        pickable=False,
    )

    # 4. Sea Water Text Label (BIỂN ĐÔNG VIỆT NAM)
    # High-contrast solid Navy Badge with bright White & Gold lettering for sharp legibility on satellite ocean
    sea_label_data = pd.DataFrame([
        {"text": "★ BIỂN ĐÔNG (VIỆT NAM)", "lon": 113.20, "lat": 14.00, "size": 22, "color": [255, 255, 255, 255]},
    ])
    sea_text_layer = pdk.Layer(
        "TextLayer",
        id="vn_sov_sea_label",
        data=sea_label_data,
        get_position=["lon", "lat"],
        get_text="text",
        get_color="color",
        get_size="size",
        get_alignment_baseline="'center'",
        get_text_anchor="'middle'",
        get_pixel_offset=[0, -12],
        character_set=VIETNAMESE_DECK_CHARSET,
        font_family="'Arial', 'Segoe UI', sans-serif",
        billboard=True,
        background_color=[15, 23, 42, 240],
        background_padding=[12, 7],
        pickable=False,
    )

    # 5. Administrative Sub-labels
    sub_labels_data = pd.DataFrame([
        {"text": "Thuộc TP. Đà Nẵng, Việt Nam", "lon": 112.00, "lat": 16.50, "size": 12, "color": [15, 23, 42, 255], "bg": [255, 255, 255, 245], "offset": [0, 16]},
        {"text": "Thuộc Tỉnh Khánh Hòa, Việt Nam", "lon": 114.20, "lat": 9.90, "size": 12, "color": [15, 23, 42, 255], "bg": [255, 255, 255, 245], "offset": [0, 16]},
        {"text": "Vùng biển chủ quyền & quyền tài phán quốc gia", "lon": 113.20, "lat": 14.00, "size": 13, "color": [250, 204, 21, 255], "bg": [15, 23, 42, 240], "offset": [0, 18]},
    ])
    sub_text_layer = pdk.Layer(
        "TextLayer",
        id="vn_sov_sub_labels",
        data=sub_labels_data,
        get_position=["lon", "lat"],
        get_text="text",
        get_color="color",
        get_size="size",
        get_alignment_baseline="'top'",
        get_text_anchor="'middle'",
        get_pixel_offset="offset",
        character_set=VIETNAMESE_DECK_CHARSET,
        font_family="'Arial', 'Segoe UI', sans-serif",
        billboard=True,
        background_color="bg",
        background_padding=[8, 4],
        pickable=False,
    )

    return [glow_layer, points_layer, islands_text_layer, sea_text_layer, sub_text_layer]
