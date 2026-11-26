"""01. Hồ sơ cảng biển — Port Dossier & Ego-Network Analysis.

Features:
- Port selector with auto-selection from query params (e.g. ?port=port434) and searchable dropdown.
- Live HUD metrics (Bậc in/out, PageRank, Betweenness Centrality, Burt Constraint).
- Comprehensive port dossier: administrative info, LOCODE, ECDIS nautical coordinates, Leiden community, k-Core.
- 6-metric percentile Radar Chart (Plotly go.Scatterpolar) comparing the port against Top 10 VN Average and Singapore (port1201).
- Interactive Pydeck Ego-Network map with radius slider (1-2 hops), incoming edges in Teal and outgoing edges in Oceanic Blue, central port in VN red/gold.
- Neighboring ports table (port, country, direction, weight, transit_days, nautical distance km, source).
- PortWatch monthly aggregated stacked bar chart for VN ports (container, tanker, dry bulk, general cargo, roro) or partner distribution for international ports.
- Enhanced "💡 Kiến thức SNA" expander on Burt's Structural Holes & Network Constraint.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.graph_objects as go

from core.data import load_nodes, load_edges, load_daily_vn, get_port_by_id
from core.theme import (
    apply_theme,
    format_nautical_coords,
    COLOR_PRIMARY_BLUE,
    COLOR_VN_RED,
    COLOR_AMBER,
    RGB_PRIMARY_BLUE,
    RGB_VN_RED,
    RGB_TEAL,
    RGB_AMBER,
)
from components.hud import render_hud
from components.deck_map import render_deck

# Apply Light Mode design system
apply_theme()

st.title("Hồ sơ cảng biển")
st.caption("Tra cứu thông tin chi tiết, phân tích radar độ đo SNA & trực quan hóa mạng lưới lân cận (Ego-Network)")

# 1. Load data
nodes_df = load_nodes()
edges_df = load_edges()
daily_vn = load_daily_vn()

# 2. Query param handling and port selector
query_port = st.query_params.get("port", None)
default_port_id = "port434"  # Hải Phòng
if query_port and (nodes_df["node_id"] == str(query_port)).any():
    default_port_id = str(query_port)

port_options = nodes_df["node_id"].tolist()
port_labels = {
    row["node_id"]: f"{row['name']} ({row['country'] if pd.notna(row['country']) else row['iso3']}) — {row['node_id']}"
    for _, row in nodes_df.iterrows()
}

default_idx = port_options.index(default_port_id) if default_port_id in port_options else 0
selected_port_id = st.selectbox(
    "Chọn cảng biển tra cứu chuyên sâu:",
    options=port_options,
    index=default_idx,
    format_func=lambda x: port_labels.get(x, x),
)

port_data = get_port_by_id(selected_port_id)
if not port_data:
    st.error(f"Không tìm thấy thông tin cảng: {selected_port_id}")
    st.stop()

# 3. Live HUD Metric Banner
in_deg = int(port_data.get("in_deg", 0))
out_deg = int(port_data.get("out_deg", 0))
pr_val = float(port_data.get("pagerank", 0.0))
betw_val = float(port_data.get("betweenness", 0.0))
constraint_val = port_data.get("constraint", None)

hud_cards = [
    {
        "title": "BẬC KẾT NỐI (IN / OUT)",
        "value": f"{in_deg} / {out_deg}",
        "delta": f"Tổng: {in_deg + out_deg} luồng",
        "delta_color": "normal",
        "icon": "git-commit",
        "tooltip": "Số tuyến đường biển trực tiếp kết nối đến và đi từ cảng",
    },
    {
        "title": "PAGERANK",
        "value": f"{pr_val:.5f}",
        "delta": "Uy tín trung chuyển",
        "delta_color": "normal",
        "icon": "activity",
        "tooltip": "Chỉ số tầm quan trọng tích lũy theo thuật toán PageRank (damping factor d = 0.85)",
    },
    {
        "title": "BETWEENNESS CENTRALITY",
        "value": f"{betw_val:.5f}",
        "delta": "Cầu nối hàng hải",
        "delta_color": "neutral",
        "icon": "network",
        "tooltip": "Tần suất cảng nằm trên đường đi ngắn nhất giữa tất cả các cặp cảng khác",
    },
    {
        "title": "BURT CONSTRAINT (C)",
        "value": f"{float(constraint_val):.4f}" if constraint_val is not None and not pd.isna(constraint_val) else "N/A",
        "delta": "Ràng buộc mạng",
        "delta_color": "inverse" if constraint_val and float(constraint_val) > 0.4 else "normal",
        "icon": "anchor",
        "tooltip": "Mức độ phụ thuộc mạng lưới theo lý thuyết Lỗ hổng Cấu trúc của Ronald Burt (Càng nhỏ càng có lợi thế môi giới)",
    },
]
render_hud(hud_cards)

# 4. Two-Column Dossier & Radar Chart
col_dossier, col_radar = st.columns([5, 5])

with col_dossier:
    st.subheader(":material/badge: Thẻ lý lịch & Thông số kỹ thuật")
    lat = float(port_data.get("lat", 0.0))
    lon = float(port_data.get("lon", 0.0))
    nautical_str = format_nautical_coords(lat, lon)

    comm_id = port_data.get("comm_leiden", port_data.get("comm_louvain", "N/A"))
    eff_size = port_data.get("effective_size", None)
    eff_str = f"{float(eff_size):.2f}" if eff_size is not None and not pd.isna(eff_size) else "N/A"

    info_df = pd.DataFrame([
        {"Chỉ tiêu": "Tên cảng biển", "Giá trị": str(port_data.get("name", "N/A"))},
        {"Chỉ tiêu": "Quốc gia", "Giá trị": f"{port_data.get('country', 'N/A')} ({port_data.get('iso3', 'N/A')})"},
        {"Chỉ tiêu": "Mã LOCODE", "Giá trị": str(port_data.get("locode", "N/A"))},
        {"Chỉ tiêu": "Tọa độ hàng hải (ECDIS)", "Giá trị": nautical_str},
        {"Chỉ tiêu": "Phân loại", "Giá trị": "Cảng biển Việt Nam 🇻🇳" if port_data.get("is_vn") else "Cảng biển Quốc tế 🌐"},
        {"Chỉ tiêu": "Phân cụm cộng đồng (Leiden)", "Giá trị": f"Cộng đồng #{comm_id}"},
        {"Chỉ tiêu": "Cấp độ lõi (k-Core)", "Giá trị": f"Tầng k={port_data.get('core_number', 'N/A')}"},
        {"Chỉ tiêu": "Quy mô hiệu quả (Burt Effective Size)", "Giá trị": eff_str},
    ])
    st.dataframe(info_df, use_container_width=True, hide_index=True)

with col_radar:
    st.subheader(":material/radar: Đánh giá Đa chiều (Percentile Radar)")
    st.caption("Chuẩn hóa bách phân vị (0–100%) so sánh với Top 10 Cảng biển Việt Nam và Chuẩn Siêu cảng Singapore")

    metrics = ["degree", "pagerank", "betweenness", "closeness_time", "eigenvector", "core_number"]
    metric_labels = [
        "Bậc kết nối<br>(Degree)",
        "PageRank<br>(Uy tín)",
        "Betweenness<br>(Trung gian)",
        "Closeness<br>(Độ gần)",
        "Eigenvector<br>(Lõi trung tâm)",
        "k-Core<br>(Tầng lõi)",
    ]

    # Calculate percentiles across all global ports
    pct_df = nodes_df[metrics].rank(pct=True) * 100.0

    # 1. Selected Port
    port_mask = nodes_df["node_id"] == selected_port_id
    if port_mask.any():
        port_row = pct_df[port_mask].iloc[0]
        r_port = [float(port_row[m]) for m in metrics]
        r_port.append(r_port[0])
    else:
        r_port = [50.0] * 7

    # 2. Vietnam Top 10 Ports Benchmark
    vn_top10_idx = nodes_df[nodes_df["is_vn"] == True].sort_values("degree", ascending=False).head(10).index
    vn_avg = pct_df.loc[vn_top10_idx].mean()
    r_vn = [float(vn_avg[m]) for m in metrics]
    r_vn.append(r_vn[0])

    # 3. Singapore Benchmark (port1201)
    sg_mask = nodes_df["node_id"] == "port1201"
    if sg_mask.any():
        sg_row = pct_df[sg_mask].iloc[0]
        r_sg = [float(sg_row[m]) for m in metrics]
        r_sg.append(r_sg[0])
    else:
        r_sg = [99.9, 100.0, 99.5, 76.8, 100.0, 91.0, 99.9]

    theta_closed = metric_labels + [metric_labels[0]]

    fig_radar = go.Figure()

    # Singapore trace
    fig_radar.add_trace(go.Scatterpolar(
        r=r_sg,
        theta=theta_closed,
        fill="toself",
        name="Singapore (Chuẩn toàn cầu)",
        line=dict(color="#7C3AED", width=1.5, dash="dot"),
        fillcolor="rgba(124, 58, 237, 0.10)",
        hovertemplate="Singapore: %{r:.1f}th percentile<extra></extra>",
    ))

    # VN Top 10 benchmark trace
    fig_radar.add_trace(go.Scatterpolar(
        r=r_vn,
        theta=theta_closed,
        fill="toself",
        name="TB Top 10 Cảng biển VN",
        line=dict(color="#DA251D", width=2),
        fillcolor="rgba(218, 37, 29, 0.15)",
        hovertemplate="Top 10 VN: %{r:.1f}th percentile<extra></extra>",
    ))

    # Selected Port trace
    p_name = str(port_data.get("name", selected_port_id))
    fig_radar.add_trace(go.Scatterpolar(
        r=r_port,
        theta=theta_closed,
        fill="toself",
        name=f"Cảng chọn: {p_name}",
        line=dict(color="#0077B6", width=2.5),
        fillcolor="rgba(0, 119, 182, 0.30)",
        hovertemplate="%{theta}: %{r:.1f}th percentile<extra></extra>",
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20%", "40%", "60%", "80%", "100%"],
                tickfont=dict(size=9, family="JetBrains Mono"),
                gridcolor="#E2E8F0",
                linecolor="#CBD5E1",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, family="Be Vietnam Pro, sans-serif", color="#0F172A"),
                gridcolor="#E2E8F0",
                linecolor="#CBD5E1",
            ),
            bgcolor="#FFFFFF",
        ),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Be Vietnam Pro, sans-serif", color="#0F172A"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=35, r=35, t=35, b=35),
        height=380,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# 5. Ego-Network Map Section
st.subheader(":material/hub: Mạng lưới Lân cận (Ego-Network)")
col_ctrl1, col_ctrl2 = st.columns([4, 6])
with col_ctrl1:
    ego_radius = st.slider(
        "Bán kính mạng lân cận (Hops):",
        min_value=1,
        max_value=2,
        value=1,
        help="Bán kính 1 hop: các cảng nối trực tiếp; Bán kính 2 hops: mở rộng thêm đối tác bậc 2",
    )
with col_ctrl2:
    st.caption("🔵 **Xanh dương**: Tuyến đi (Outgoing) | 🟢 **Xanh ngọc (Teal)**: Tuyến đến (Incoming) | 🔴 **Đỏ/Vàng**: Cảng trung tâm")

# Extract 1-hop outgoing and incoming edges
coords_dict = nodes_df.set_index("node_id")[["lon", "lat", "name", "country", "degree", "is_vn"]].to_dict("index")

out_edges = edges_df[edges_df["source"] == selected_port_id].copy()
in_edges = edges_df[edges_df["target"] == selected_port_id].copy()

for df_e in [out_edges, in_edges]:
    df_e["src_lon"] = df_e["source"].map(lambda x: coords_dict.get(str(x), {}).get("lon"))
    df_e["src_lat"] = df_e["source"].map(lambda x: coords_dict.get(str(x), {}).get("lat"))
    df_e["dst_lon"] = df_e["target"].map(lambda x: coords_dict.get(str(x), {}).get("lon"))
    df_e["dst_lat"] = df_e["target"].map(lambda x: coords_dict.get(str(x), {}).get("lat"))

out_clean = out_edges.dropna(subset=["src_lon", "src_lat", "dst_lon", "dst_lat"])
in_clean = in_edges.dropna(subset=["src_lon", "src_lat", "dst_lon", "dst_lat"])

ego_layers = []

# If 2 hops selected, add 2nd-hop edges (clamped to top 400 by weight to maintain high frame rate)
if ego_radius == 2:
    hop1_neighbors = set(out_clean["target"]).union(set(in_clean["source"]))
    hop2_edges = edges_df[
        (edges_df["source"].isin(hop1_neighbors) | edges_df["target"].isin(hop1_neighbors)) &
        (edges_df["source"] != selected_port_id) &
        (edges_df["target"] != selected_port_id)
    ].sort_values(by="weight", ascending=False).head(400).copy()

    hop2_edges["src_lon"] = hop2_edges["source"].map(lambda x: coords_dict.get(str(x), {}).get("lon"))
    hop2_edges["src_lat"] = hop2_edges["source"].map(lambda x: coords_dict.get(str(x), {}).get("lat"))
    hop2_edges["dst_lon"] = hop2_edges["target"].map(lambda x: coords_dict.get(str(x), {}).get("lon"))
    hop2_edges["dst_lat"] = hop2_edges["target"].map(lambda x: coords_dict.get(str(x), {}).get("lat"))
    hop2_clean = hop2_edges.dropna(subset=["src_lon", "src_lat", "dst_lon", "dst_lat"])

    if len(hop2_clean) > 0:
        ego_layers.append(pdk.Layer(
            "ArcLayer",
            id="ego_hop2_arcs",
            data=hop2_clean,
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
            get_source_color=[148, 163, 184, 90],
            get_target_color=[148, 163, 184, 90],
            get_width=1.2,
            width_min_pixels=1,
            great_circle=True,
            pickable=False,
        ))

# Outgoing Arcs Layer (Oceanic Blue)
if len(out_clean) > 0:
    ego_layers.append(pdk.Layer(
        "ArcLayer",
        id="ego_out_arcs",
        data=out_clean,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dst_lon", "dst_lat"],
        get_source_color=[0, 119, 182, 220],
        get_target_color=[2, 132, 199, 160],
        get_width=2.5,
        width_min_pixels=1.5,
        great_circle=True,
        pickable=True,
        auto_highlight=True,
    ))

# Incoming Arcs Layer (Deep Teal)
if len(in_clean) > 0:
    ego_layers.append(pdk.Layer(
        "ArcLayer",
        id="ego_in_arcs",
        data=in_clean,
        get_source_position=["src_lon", "src_lat"],
        get_target_position=["dst_lon", "dst_lat"],
        get_source_color=[13, 148, 136, 180],
        get_target_color=[5, 150, 105, 230],
        get_width=2.5,
        width_min_pixels=1.5,
        great_circle=True,
        pickable=True,
        auto_highlight=True,
    ))

# 1-hop Neighbor nodes
neighbor_ids = set(out_clean["target"]).union(set(in_clean["source"]))
neighbor_nodes = nodes_df[nodes_df["node_id"].isin(neighbor_ids)].copy()
neighbor_nodes["radius_m"] = 12000.0

ego_layers.append(pdk.Layer(
    "ScatterplotLayer",
    id="ego_neighbors",
    data=neighbor_nodes,
    get_position=["lon", "lat"],
    get_radius="radius_m",
    get_fill_color=[0, 119, 182, 200],
    get_line_color=[255, 255, 255, 220],
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    radius_min_pixels=3,
    pickable=True,
))

# Center Ego Port Node (Prominent Vietnam-Red/Gold Pin)
center_df = pd.DataFrame([port_data])
ego_layers.append(pdk.Layer(
    "ScatterplotLayer",
    id="ego_center_node",
    data=center_df,
    get_position=["lon", "lat"],
    get_radius=28000.0,
    get_fill_color=[218, 37, 29, 255],
    get_line_color=[255, 255, 255, 255],
    stroked=True,
    filled=True,
    line_width_min_pixels=3,
    radius_min_pixels=8,
    pickable=True,
))

# Text label for Center Port
ego_layers.append(pdk.Layer(
    "TextLayer",
    id="ego_center_label",
    data=center_df,
    get_position=["lon", "lat"],
    get_text="name",
    get_color=[255, 255, 255, 255],
    get_size=14,
    get_alignment_baseline="'bottom'",
    get_pixel_offset=[0, -15],
    character_set="'auto'",
    font_family="'Arial'",
    billboard=True,
    background_color=[15, 23, 42, 230],
    background_padding=[6, 4],
    pickable=False,
))

ego_view = pdk.ViewState(
    latitude=lat,
    longitude=lon,
    zoom=4.8 if ego_radius == 1 else 3.8,
    pitch=20,
    bearing=0,
)
ego_deck = render_deck(ego_layers, view_state=ego_view, height=520)
st.pydeck_chart(ego_deck, use_container_width=True)

# 6. Neighboring Ports Table & PortWatch Chart
col_table, col_ts = st.columns([5, 5])

with col_table:
    st.subheader(":material/table_rows: Bảng Cảng Biển Lân cận Trực tiếp")
    st.caption("Các cảng đối tác có luồng tàu chạy trực tiếp với cảng đang chọn")

    out_e = edges_df[edges_df["source"] == selected_port_id].copy()
    out_e["direction"] = "Đi (Outgoing ↗)"
    out_e["partner_id"] = out_e["target"]

    in_e = edges_df[edges_df["target"] == selected_port_id].copy()
    in_e["direction"] = "Đến (Incoming ↘)"
    in_e["partner_id"] = in_e["source"]

    combined_neighbors = pd.concat([out_e, in_e], ignore_index=True)
    if len(combined_neighbors) > 0:
        combined_neighbors["partner_name"] = combined_neighbors["partner_id"].map(
            lambda x: coords_dict.get(str(x), {}).get("name", x)
        )
        combined_neighbors["partner_country"] = combined_neighbors["partner_id"].map(
            lambda x: coords_dict.get(str(x), {}).get("country", "")
        )

        display_neighbors = combined_neighbors[[
            "partner_id", "partner_name", "partner_country",
            "direction", "weight", "transit_days", "distance_km", "src"
        ]].rename(columns={
            "partner_id": "Mã cảng",
            "partner_name": "Tên cảng đối tác",
            "partner_country": "Quốc gia",
            "direction": "Hướng tuyến",
            "weight": "Trọng số",
            "transit_days": "Ngày đi",
            "distance_km": "Khoảng cách (km)",
            "src": "Nguồn",
        }).sort_values(by="Trọng số", ascending=False).reset_index(drop=True)
        display_neighbors.index = display_neighbors.index + 1

        st.dataframe(
            display_neighbors.style.format({
                "Trọng số": "{:.3f}",
                "Ngày đi": "{:.1f}",
                "Khoảng cách (km)": "{:,.0f}",
            }),
            use_container_width=True,
            height=360,
        )
    else:
        st.info("Không có liên kết lân cận trực tiếp.")

with col_ts:
    st.subheader(":material/analytics: Phân tích Hoạt động & Đối tác")
    
    # Matching logic for Vietnamese ports in PortWatch daily_vn
    VN_PORTWATCH_ALIAS = {
        "port434": "Hai Phong",
        "port269": "Da Nang",
        "port1291": "Saigon",
        "port1375": "Vung Tau",
        "port1078": "Quy Nhon",
        "port983": "Can Tho",
        "port473": "Hon Gai",
        "port819": "Nghe Tinh",
        "port213": "Dung Quat",
        "port903": "Cai Mep",
    }

    matched_name = None
    if port_data.get("is_vn") and daily_vn is not None and len(daily_vn) > 0:
        avail_names = set(daily_vn["portname"].unique()) if "portname" in daily_vn.columns else set()
        if selected_port_id in VN_PORTWATCH_ALIAS and VN_PORTWATCH_ALIAS[selected_port_id] in avail_names:
            matched_name = VN_PORTWATCH_ALIAS[selected_port_id]
        else:
            clean_name = str(port_data.get("name", "")).replace("_Vietnam", "").strip().lower()
            for an in avail_names:
                if an.lower() in clean_name or clean_name in an.lower():
                    matched_name = an
                    break

    if matched_name and daily_vn is not None:
        st.caption(f"Xu hướng lượt tàu ghé cảng **{matched_name}** theo tháng (IMF PortWatch 2019-2026)")
        sub = daily_vn[daily_vn["portname"] == matched_name].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub["month_period"] = sub["date"].dt.to_period("M").dt.to_timestamp()

        cols_to_sum = [c for c in [
            "portcalls_container",
            "portcalls_dry_bulk",
            "portcalls_general_cargo",
            "portcalls_tanker",
            "portcalls_roro",
            "portcalls",
        ] if c in sub.columns]

        monthly = sub.groupby("month_period")[cols_to_sum].sum().reset_index()

        fig_pw = go.Figure()
        if "portcalls_container" in monthly.columns:
            fig_pw.add_trace(go.Bar(x=monthly["month_period"], y=monthly["portcalls_container"], name="Tàu Container", marker_color="#0077B6"))
        if "portcalls_tanker" in monthly.columns:
            fig_pw.add_trace(go.Bar(x=monthly["month_period"], y=monthly["portcalls_tanker"], name="Tàu Dầu (Tanker)", marker_color="#F59E0B"))
        if "portcalls_dry_bulk" in monthly.columns:
            fig_pw.add_trace(go.Bar(x=monthly["month_period"], y=monthly["portcalls_dry_bulk"], name="Tàu Hàng rời", marker_color="#059669"))
        if "portcalls_general_cargo" in monthly.columns:
            fig_pw.add_trace(go.Bar(x=monthly["month_period"], y=monthly["portcalls_general_cargo"], name="Tàu Bách hóa", marker_color="#0D9488"))
        if "portcalls_roro" in monthly.columns:
            fig_pw.add_trace(go.Bar(x=monthly["month_period"], y=monthly["portcalls_roro"], name="Tàu Ro-Ro", marker_color="#6366F1"))

        if "portcalls" in monthly.columns:
            fig_pw.add_trace(go.Scatter(x=monthly["month_period"], y=monthly["portcalls"], name="Tổng lượt", line=dict(color="#0F172A", width=1.5, dash="dash"), mode="lines"))

        fig_pw.update_layout(
            barmode="stack",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Be Vietnam Pro, sans-serif", color="#0F172A"),
            hovermode="x unified",
            xaxis=dict(gridcolor="#F1F5F9", linecolor="#CBD5E1"),
            yaxis=dict(title="Số lượt tàu ghé", gridcolor="#F1F5F9", linecolor="#CBD5E1"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            margin=dict(l=35, r=15, t=30, b=30),
            height=360,
        )
        st.plotly_chart(fig_pw, use_container_width=True)

    else:
        # For International ports: Top 10 partner countries by connection weight
        st.caption("Cơ cấu quốc gia đối tác hàng hải theo tổng trọng số kết nối (Top 10)")
        if len(combined_neighbors) > 0:
            partner_country_df = combined_neighbors.groupby("partner_country")["weight"].sum().reset_index()
            partner_country_df = partner_country_df[partner_country_df["partner_country"] != ""].sort_values(by="weight", ascending=True).tail(10)

            fig_p = go.Figure(go.Bar(
                x=partner_country_df["weight"],
                y=partner_country_df["partner_country"],
                orientation="h",
                marker_color="#0077B6",
                text=partner_country_df["weight"].round(1),
                textposition="auto",
            ))
            fig_p.update_layout(
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                font=dict(family="Be Vietnam Pro, sans-serif", color="#0F172A"),
                xaxis=dict(title="Tổng trọng số kết nối", gridcolor="#F1F5F9", linecolor="#CBD5E1"),
                yaxis=dict(linecolor="#CBD5E1"),
                margin=dict(l=40, r=20, t=20, b=30),
                height=360,
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("Chưa có dữ liệu đối tác cho cảng này.")

# 6b. Port-to-Port Comparison (So sánh đối đầu 2 cảng)
st.markdown("---")
st.subheader(":material/compare_arrows: So sánh Đối đầu 2 Cảng (Port-to-Port Comparison)")
st.caption("Đặt cạnh nhau các độ đo SNA, đối tác trung chuyển chung và biểu đồ Radar giữa cảng đang xem và cảng đối chứng.")

top_global_id = str(nodes_df.sort_values("pagerank", ascending=False).iloc[0]["node_id"])
alt_global_id = str(nodes_df.sort_values("pagerank", ascending=False).iloc[1]["node_id"])
default_comp_id = top_global_id if top_global_id != selected_port_id else alt_global_id
comp_idx = port_options.index(default_comp_id) if default_comp_id in port_options else 0

compare_port_id = st.selectbox(
    "Chọn Cảng đối chứng (Cảng B) để so sánh với cảng đang xem:",
    options=port_options,
    index=comp_idx,
    format_func=lambda x: port_labels.get(x, x),
    key="compare_port_select",
)

comp_data = get_port_by_id(compare_port_id)
if comp_data:
    c_cmp1, c_cmp2 = st.columns([5, 5])
    with c_cmp1:
        comp_metrics = [
            ("PageRank", f"{float(port_data.get('pagerank', 0)):.5f}", f"{float(comp_data.get('pagerank', 0)):.5f}"),
            ("Betweenness", f"{float(port_data.get('betweenness', 0)):.5f}", f"{float(comp_data.get('betweenness', 0)):.5f}"),
            ("Closeness", f"{float(port_data.get('closeness_time', 0)):.4f}", f"{float(comp_data.get('closeness_time', 0)):.4f}"),
            ("Degree (Tổng bậc)", f"{int(port_data.get('degree', 0)):,}", f"{int(comp_data.get('degree', 0)):,}"),
            ("k-Core", f"{int(port_data.get('core_number', 0))}", f"{int(comp_data.get('core_number', 0))}"),
            ("Cộng đồng Leiden", f"Cụm #{int(port_data.get('comm_leiden', -1))}", f"Cụm #{int(comp_data.get('comm_leiden', -1))}"),
        ]
        comp_table = pd.DataFrame(
            comp_metrics,
            columns=["Chỉ số SNA", f"Cảng A: {port_data.get('name', selected_port_id)}", f"Cảng B: {comp_data.get('name', compare_port_id)}"],
        )
        st.dataframe(comp_table, use_container_width=True, hide_index=True)

        # Common neighbors & direct connection check
        u_graph = get_undirected_graph()
        nbrs_a = set(u_graph.neighbors(selected_port_id)) if selected_port_id in u_graph else set()
        nbrs_b = set(u_graph.neighbors(compare_port_id)) if compare_port_id in u_graph else set()
        common_nbrs = nbrs_a & nbrs_b
        union_nbrs = nbrs_a | nbrs_b
        jacc = len(common_nbrs) / len(union_nbrs) if union_nbrs else 0.0
        direct_connected = u_graph.has_edge(selected_port_id, compare_port_id) if (selected_port_id in u_graph and compare_port_id in u_graph) else False

        m1, m2, m3 = st.columns(3)
        m1.metric("Đối tác chung", f"{len(common_nbrs):,} cảng")
        m2.metric("Tương đồng Jaccard", f"{jacc:.3f}")
        m3.metric("Kết nối trực tiếp", "Có tuyến thẳng" if direct_connected else "Qua trung chuyển")

        st.download_button(
            label=":material/download: Tải bảng So sánh (CSV)",
            data=comp_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"so_sanh_{selected_port_id}_vs_{compare_port_id}.csv",
            mime="text/csv",
        )

    with c_cmp2:
        radar_cols = ["pagerank", "betweenness", "closeness_time", "degree", "core_number"]
        radar_labels = ["PageRank", "Betweenness", "Closeness", "Degree", "k-Core"]
        vals_a, vals_b = [], []
        for col in radar_cols:
            max_v = pd.to_numeric(nodes_df[col], errors="coerce").max() if col in nodes_df.columns else 1.0
            max_v = float(max_v) if max_v and max_v > 0 else 1.0
            vals_a.append(min(1.0, float(port_data.get(col, 0) or 0) / max_v))
            vals_b.append(min(1.0, float(comp_data.get(col, 0) or 0) / max_v))

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatterpolar(
            r=vals_a + [vals_a[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=str(port_data.get("name", selected_port_id)),
            line_color="#0077B6",
            fillcolor="rgba(0, 119, 182, 0.22)",
        ))
        fig_cmp.add_trace(go.Scatterpolar(
            r=vals_b + [vals_b[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=str(comp_data.get("name", compare_port_id)),
            line_color="#DA251D",
            fillcolor="rgba(218, 37, 29, 0.16)",
        ))
        fig_cmp.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0")),
            paper_bgcolor="#FFFFFF",
            font=dict(family="Be Vietnam Pro, sans-serif", color="#0F172A"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=40, b=20),
            height=340,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# 7. Mandatory SNA Knowledge Expander
with st.expander(":material/lightbulb: Kiến thức SNA: Lỗ hổng Cấu trúc (Structural Holes) & Chỉ số Ràng buộc Burt"):
    st.markdown("""
    **1. Lý thuyết Lỗ hổng Cấu trúc (Ronald Burt, 1992):**
    Trong mạng lưới vận tải hàng hải toàn cầu, các cảng biển không kết nối đồng nhất với nhau mà phân chia thành các cụm thị trường địa phương. Khi hai nhóm cảng không có tuyến tàu nối trực tiếp, giữa chúng tồn tại một "lỗ hổng cấu trúc" (structural hole).
    Một cảng biển nằm ở vị trí bắc cầu (bridge) qua lỗ hổng cấu trúc sẽ nắm giữ hai lợi thế cốt lõi:
    - **Lợi thế tiếp cận thông tin sớm (Information Advantage)**: Nắm bắt trước các biến động về luồng hàng và giá cước vận tải.
    - **Lợi thế kiểm soát & trung chuyển (Control Advantage)**: Quyết định luồng tuyến, thu hút các liên minh hãng tàu quốc tế (ocean alliances).

    **2. Chỉ số Ràng buộc Mạng (Network Constraint $C_i$):**
    $$C_i = \\sum_{j \\in N(i)} c_{ij}, \\quad c_{ij} = \\left( p_{ij} + \\sum_{q \\in N(i) \\setminus \\{i, j\\}} p_{iq} p_{qj} \\right)^2$$
    Trong đó:
    - $p_{ij} = \\frac{w_{ij}}{\\sum_k w_{ik}}$ là tỷ trọng đầu tư nguồn lực/trọng số kết nối từ cảng $i$ tới đối tác $j$.
    - $p_{iq} p_{qj}$ phản ánh mức độ phụ thuộc gián tiếp qua đối tác trung gian $q$.
    - **Ý nghĩa thực tiễn**: $C_i$ càng cao (tiến gần 1.0) thể hiện cảng bị phụ thuộc vào một cụm đối tác khép kín dày đặc (redundant network). Ngược lại, $C_i$ càng thấp thể hiện cảng có mạng lưới mở, ít bị ràng buộc, đóng vai trò hub trung chuyển vượt bậc như Singapore hay Rotterdam.

    **3. Quy mô Hiệu quả (Burt's Effective Size $ES_i$):**
    $$ES_i = \\sum_{j \\in N(i)} \\left( 1 - \\sum_{q} p_{iq} m_{jq} \\right)$$
    Đo lường số lượng đối tác không trùng lặp (non-redundant partners). Nếu tất cả các đối tác của cảng $i$ đều kết nối chéo với nhau, $ES_i \\approx 1$. Nếu các đối tác hoàn toàn độc lập, $ES_i \\approx \\text{degree}(i)$.
    """)

st.caption("Nguồn dữ liệu: IMF PortWatch, UNCTADstat, Global Fishing Watch.")
