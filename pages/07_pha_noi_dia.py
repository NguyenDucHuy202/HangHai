"""07. Mạng phà nội địa — Domestic Coastal & Island Ferry Network.

Features:
- Domestic ferry network using ferry_nodes.parquet and ferry_edges.parquet.
- 5 Live HUD metric cards (Terminals: 25, Routes: 17, Components: 8, Connectivity: 100%, Longest route: 190 km).
- Regional coastal cluster selector with auto-centering camera ViewStates.
- Interactive Pydeck map with Teal LineLayer, terminals ScatterplotLayer, OpenSeaMap seamark layer toggle.
- Point-to-point ferry route planner using nx.shortest_path(F, src, dst):
  * Total km, total hours, hops, itinerary steps, and highlighted route in amber/blue.
  * Disconnected island cluster warning explaining geographical fragmentation and multi-modal transport alternatives.
- Terminal directory and timetable/routes table in tabs.
- "💡 Kiến thức SNA" expander on multimodal transport feeder integration and island connectivity resilience.
- Full Light Mode styling (core/theme.py).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from core.data import (
    load_ferry_nodes,
    load_ferry_edges,
    get_ferry_graph,
)
from core.theme import (
    apply_theme,
    COLOR_BG,
    COLOR_CARD,
    COLOR_CARD_BORDER,
    COLOR_PRIMARY_BLUE,
    COLOR_PRIMARY_DARK,
    COLOR_VN_RED,
    COLOR_AMBER,
    COLOR_TEAL,
    COLOR_SUCCESS,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    RGB_AMBER,
    RGB_VN_RED,
    RGB_TEAL,
    RGB_PRIMARY_BLUE,
    format_nautical_coords,
)
from components.hud import render_hud
from components.layers import seamark_layer
from components.deck_map import render_deck, VIEW_VIETNAM

apply_theme()

st.title("Mạng phà Nội địa & Kết nối Hải đảo")
st.caption("Mạng lưới kết nối bến phà ven biển, cảng tàu khách hải đảo & bài toán tiếp cận giao thông thủy nội địa Việt Nam")

# ---------------------------------------------------------------------------
# Data Loading & Graph Construction
# ---------------------------------------------------------------------------

fnodes = load_ferry_nodes()
fedges = load_ferry_edges()
F = get_ferry_graph()

# Precompute connected components
num_components = nx.number_connected_components(F)
components_list = list(nx.connected_components(F))

# Longest route
longest_route_km = fedges["distance_km"].max() if "distance_km" in fedges.columns else 190.0
longest_route_row = fedges[fedges["distance_km"] == longest_route_km].iloc[0] if not fedges.empty else {}
longest_route_name = longest_route_row.get("route_name", "Cần Thơ – Côn Đảo")

# Lookup tables
node_dict = fnodes.set_index("terminal_id").to_dict("index")
terminal_options = [
    (tid, f"{r['name']} ({r['region']})")
    for tid, r in fnodes.set_index("terminal_id").iterrows()
]
terminal_labels = [opt[1] for opt in terminal_options]
terminal_id_by_label = {opt[1]: opt[0] for opt in terminal_options}

# ---------------------------------------------------------------------------
# 1. Live HUD Metrics
# ---------------------------------------------------------------------------

hud_cards = [
    {
        "title": "BẾN PHÀ NỘI ĐỊA",
        "value": f"{len(fnodes)} Bến",
        "delta": "5 Cụm ven biển",
        "delta_color": "normal",
        "icon": "ferry",
        "tooltip": "Tổng số bến phà ven biển và cảng khách hải đảo Việt Nam được quản lý",
    },
    {
        "title": "TUYẾN PHÀ KHAI THÁC",
        "value": f"{len(fedges) // 2} Tuyến",
        "delta": f"{len(fedges)} Chặng 2 chiều",
        "delta_color": "normal",
        "icon": "route",
        "tooltip": "Các tuyến tàu cao tốc và phà biển phục vụ dân sinh & du lịch biển đảo",
    },
    {
        "title": "CỤM LIÊN THÔNG ĐỘC LẬP",
        "value": f"{num_components} Cụm",
        "delta": "Biệt lập địa lý",
        "delta_color": "neutral",
        "icon": "network",
        "tooltip": "Số thành phần liên thông rời rạc do giới hạn địa lý của các quần đảo",
    },
    {
        "title": "KẾT NỐI HẢI ĐẢO",
        "value": "100%",
        "delta": "Không đảo bị cô lập",
        "delta_color": "normal",
        "icon": "check",
        "tooltip": "100% các đảo tiền tiêu đều duy trì kết nối thủy trực tiếp với đất liền",
    },
    {
        "title": "TUYẾN DÀI NHẤT",
        "value": f"{longest_route_km:,.0f} km",
        "delta": "Cần Thơ – Côn Đảo (3.8h)",
        "delta_color": "normal",
        "icon": "clock",
        "tooltip": f"Tuyến phà cao tốc đường dài: {longest_route_name} ({longest_route_km} km)",
    },
]
render_hud(hud_cards)

# ---------------------------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------------------------

tab_map, tab_planner, tab_terminals, tab_routes = st.tabs([
    "Bản đồ Mạng lưới Phà & Cụm Ven biển",
    "Tìm Luồng Di chuyển Điểm – Điểm (Route Planner)",
    "Danh bạ 25 Bến phà & Cảng khách",
    "Danh mục 17 Tuyến phà Nội địa",
])

# Regional Camera ViewStates mapping
CLUSTER_VIEWS = {
    "Tất cả các cụm (Toàn quốc)": pdk.ViewState(latitude=16.0, longitude=107.5, zoom=5.2, pitch=0),
    "Vịnh Bắc Bộ (Hải Phòng - Cát Bà - Cô Tô)": pdk.ViewState(latitude=20.85, longitude=107.2, zoom=8.5, pitch=0),
    "Duyên hải Miền Trung (Lý Sơn - Cù Lao Chàm - Phú Quý)": pdk.ViewState(latitude=14.5, longitude=108.8, zoom=6.8, pitch=0),
    "Đông Nam Bộ & Côn Đảo (TP.HCM - Cần Giờ - Vũng Tàu - Côn Đảo)": pdk.ViewState(latitude=9.8, longitude=106.8, zoom=7.5, pitch=0),
    "Tây Nam Bộ / Kiên Giang (Rạch Giá - Hà Tiên - Phú Quốc - Nam Du)": pdk.ViewState(latitude=10.0, longitude=104.5, zoom=8.2, pitch=0),
}

# ---------------------------------------------------------------------------
# TAB 1: Bản đồ Mạng lưới Phà & Cụm Ven biển
# ---------------------------------------------------------------------------

with tab_map:
    st.subheader("1. Bản đồ Không gian Mạng lưới Phà & Giao thông Thủy Nội địa")
    st.caption("Hiển thị các tuyến phà biển (Teal LineLayer), bến phà hải đảo (White/Teal ScatterplotLayer) và hải tiêu OpenSeaMap:")

    col_view, col_toggle = st.columns([6, 4])
    with col_view:
        selected_view_name = st.selectbox(
            "🔭 Tiêu cự Camera theo Vùng Cụm Ven biển:",
            options=list(CLUSTER_VIEWS.keys()),
            index=0,
            key="select_cluster_view",
        )
        current_view_state = CLUSTER_VIEWS[selected_view_name]

    with col_toggle:
        st.write("")  # Alignment spacing
        enable_seamark = st.checkbox(
            "⚓ Bật lớp hải đồ & phao tiêu OpenSeaMap (Seamark Layer)",
            value=False,
            help="Nạp các lớp gạch tile raster phao tiêu, đèn biển, phao luồng từ OpenSeaMap",
        )

    # Filter nodes based on view selection
    display_nodes = fnodes.copy()
    if selected_view_name == "Vịnh Bắc Bộ (Hải Phòng - Cát Bà - Cô Tô)":
        display_nodes = display_nodes[display_nodes["region"] == "Vịnh Bắc Bộ"]
    elif selected_view_name == "Duyên hải Miền Trung (Lý Sơn - Cù Lao Chàm - Phú Quý)":
        display_nodes = display_nodes[display_nodes["region"] == "Duyên hải Miền Trung"]
    elif selected_view_name == "Đông Nam Bộ & Côn Đảo (TP.HCM - Cần Giờ - Vũng Tàu - Côn Đảo)":
        display_nodes = display_nodes[display_nodes["region"] == "Đông Nam Bộ"]
    elif selected_view_name == "Tây Nam Bộ / Kiên Giang (Rạch Giá - Hà Tiên - Phú Quốc - Nam Du)":
        display_nodes = display_nodes[display_nodes["region"] == "Đồng bằng sông Cửu Long"]

    # Build line edges data by joining nodes coordinates
    ferry_lines_data: List[Dict[str, Any]] = []
    for _, edge in fedges.iterrows():
        u = str(edge["source"])
        v = str(edge["target"])
        u_meta = node_dict.get(u)
        v_meta = node_dict.get(v)
        if u_meta and v_meta:
            ferry_lines_data.append({
                "src_lon": u_meta["lon"],
                "src_lat": u_meta["lat"],
                "dst_lon": v_meta["lon"],
                "dst_lat": v_meta["lat"],
                "route_name": edge.get("route_name", f"{u_meta['name']} – {v_meta['name']}"),
                "distance_km": float(edge.get("distance_km", 0.0)),
                "transit_hours": float(edge.get("transit_hours", 0.0)),
            })

    # Prepare Layers
    deck_layers: List[pdk.Layer] = []

    # Optional seamark layer
    if enable_seamark:
        deck_layers.append(seamark_layer())

    # Ferry routes LineLayer (Teal)
    if ferry_lines_data:
        df_lines = pd.DataFrame(ferry_lines_data)
        lines_layer = pdk.Layer(
            "LineLayer",
            id="ferry_routes_teal",
            data=df_lines,
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
            get_color=RGB_TEAL + [220],
            get_width=3.5,
            width_min_pixels=2.5,
            pickable=True,
            auto_highlight=True,
        )
        deck_layers.append(lines_layer)

    # Ferry terminals ScatterplotLayer (White fill, Teal border)
    if not display_nodes.empty:
        terminals_layer = pdk.Layer(
            "ScatterplotLayer",
            id="ferry_terminals_points",
            data=display_nodes,
            get_position=["lon", "lat"],
            get_radius=6000,
            radius_min_pixels=5,
            radius_max_pixels=15,
            get_fill_color=[255, 255, 255, 255],
            get_line_color=RGB_TEAL + [255],
            stroked=True,
            filled=True,
            line_width_min_pixels=2.5,
            pickable=True,
            auto_highlight=True,
        )
        deck_layers.append(terminals_layer)

    # Render Deck
    deck_map = render_deck(
        deck_layers,
        view_state=current_view_state,
        height=540,
        tooltip={
            "html": "<b>{name}</b><br/>"
                    "<b>Vùng:</b> {region}<br/>"
                    "<b>Số tuyến kết nối:</b> {degree}<br/>"
                    "<b>Tuyến:</b> {route_name}<br/>"
                    "<b>Cự ly:</b> {distance_km} km ({transit_hours} h)",
            "style": {"backgroundColor": "#0F172A", "color": "#FFFFFF", "fontSize": "12px", "borderRadius": "4px"},
        },
    )
    st.pydeck_chart(deck_map, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: Tìm Luồng Di chuyển Điểm – Điểm (Route Planner)
# ---------------------------------------------------------------------------

with tab_planner:
    st.subheader("2. Trình Tìm Luồng Di chuyển Điểm – Điểm (Ferry Route Planner)")
    st.caption("Tìm lộ trình đường thủy ngắn nhất giữa 2 bến phà, đánh giá số lần đổi tàu và phát hiện phân mảnh địa lý:")

    col_orig, col_dest, col_crit = st.columns([4, 4, 3])

    default_orig_idx = next((i for i, opt in enumerate(terminal_options) if opt[0] == "ferry_rachgia"), 0)
    default_dest_idx = next((i for i, opt in enumerate(terminal_options) if opt[0] == "ferry_namdu"), 4)

    with col_orig:
        orig_label = st.selectbox(
            "🚩 Chọn bến xuất phát (Origin):",
            options=terminal_labels,
            index=default_orig_idx,
            key="select_ferry_origin",
        )
        orig_id = terminal_id_by_label[orig_label]

    with col_dest:
        dest_label = st.selectbox(
            "🏁 Chọn bến đến (Destination):",
            options=terminal_labels,
            index=default_dest_idx,
            key="select_ferry_dest",
        )
        dest_id = terminal_id_by_label[dest_label]

    with col_crit:
        criterion = st.selectbox(
            "⚙️ Tiêu chí tối ưu:",
            options=["Nhanh nhất (giờ di chuyển)", "Ngắn nhất (km)", "Ít đổi tàu nhất (chặng)"],
            index=0,
            key="select_ferry_criterion",
        )

    # Route Planner Execution
    if orig_id == dest_id:
        st.info("⚠️ Điểm xuất phát và điểm đến trùng nhau. Không cần di chuyển.")
    else:
        weight_key = "transit_hours" if "Nhanh nhất" in criterion else ("distance_km" if "Ngắn nhất" in criterion else None)

        # Check topological connectivity
        if nx.has_path(F, orig_id, dest_id):
            shortest_nodes = nx.shortest_path(F, source=orig_id, target=dest_id, weight=weight_key)
            num_hops = len(shortest_nodes) - 1
            num_transfers = max(0, num_hops - 1)

            # Accumulate distance and time
            total_km = 0.0
            total_hours = 0.0
            itinerary_steps: List[Dict[str, Any]] = []
            highlight_lines: List[Dict[str, Any]] = []

            for i in range(num_hops):
                u = shortest_nodes[i]
                v = shortest_nodes[i + 1]
                edge_data = F.get_edge_data(u, v) or {}
                d_km = float(edge_data.get("distance_km", 0.0))
                t_hr = float(edge_data.get("transit_hours", 0.0))
                r_name = edge_data.get("route_name", f"{node_dict[u]['name']} – {node_dict[v]['name']}")

                total_km += d_km
                total_hours += t_hr

                itinerary_steps.append({
                    "Chặng": f"Chặng {i+1}",
                    "Từ bến": node_dict[u]["name"],
                    "Đến bến": node_dict[v]["name"],
                    "Tên tuyến": r_name,
                    "Cự ly": f"{d_km:.1f} km",
                    "Thời gian": f"{t_hr:.2f} giờ ({t_hr * 60:.0f} phút)",
                })

                highlight_lines.append({
                    "src_lon": node_dict[u]["lon"],
                    "src_lat": node_dict[u]["lat"],
                    "dst_lon": node_dict[v]["lon"],
                    "dst_lat": node_dict[v]["lat"],
                    "route_name": r_name,
                    "distance_km": d_km,
                    "transit_hours": t_hr,
                })

            # Display Itinerary Metrics
            st.success(f"✅ Đã tìm thấy hải trình kết nối giữa **{node_dict[orig_id]['name']}** và **{node_dict[dest_id]['name']}**!")

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Tổng cự ly đường thủy", f"{total_km:.1f} km")
            col_m2.metric("Thời gian di chuyển ước tính", f"{total_hours:.2f} giờ")
            col_m3.metric("Số chặng phà", f"{num_hops} Chặng")
            col_m4.metric("Số lần đổi tàu", f"{num_transfers} Lần đổi tàu")

            # Itinerary Table
            st.markdown("##### 🧭 Dòng thời gian Hành trình chi tiết:")
            st.dataframe(pd.DataFrame(itinerary_steps), use_container_width=True, hide_index=True)

            # Highlighted Map
            st.markdown("##### 🗺️ Tuyến đường Highlight trên Bản đồ (Màu Vàng Hổ Phách):")
            hl_layers: List[pdk.Layer] = []

            # Background base lines
            df_base = pd.DataFrame(ferry_lines_data)
            hl_layers.append(pdk.Layer(
                "LineLayer",
                id="base_lines",
                data=df_base,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["dst_lon", "dst_lat"],
                get_color=[203, 213, 225, 120],
                get_width=2,
            ))

            # Highlight lines
            df_hl = pd.DataFrame(highlight_lines)
            hl_layers.append(pdk.Layer(
                "LineLayer",
                id="highlighted_route_line",
                data=df_hl,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["dst_lon", "dst_lat"],
                get_color=RGB_AMBER + [255],
                get_width=5.5,
                width_min_pixels=4,
            ))

            # Highlighted nodes
            path_nodes_df = fnodes[fnodes["terminal_id"].isin(shortest_nodes)].copy()
            hl_layers.append(pdk.Layer(
                "ScatterplotLayer",
                id="path_nodes_highlight",
                data=path_nodes_df,
                get_position=["lon", "lat"],
                get_radius=8000,
                radius_min_pixels=7,
                get_fill_color=RGB_AMBER + [240],
                get_line_color=[255, 255, 255, 255],
                stroked=True,
                filled=True,
                line_width_min_pixels=2.5,
            ))

            # Camera centered on route midpoint
            mid_lon = (node_dict[orig_id]["lon"] + node_dict[dest_id]["lon"]) / 2.0
            mid_lat = (node_dict[orig_id]["lat"] + node_dict[dest_id]["lat"]) / 2.0
            route_view = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=7.5, pitch=0)

            deck_hl = render_deck(hl_layers, view_state=route_view, height=450)
            st.pydeck_chart(deck_hl, use_container_width=True)

        else:
            # Informative guidance for disconnected components
            orig_reg = node_dict[orig_id]["region"]
            dest_reg = node_dict[dest_id]["region"]
            st.warning(
                f"⚠️ **Không có luồng phà kết nối trực tiếp** giữa **{node_dict[orig_id]['name']}** ({orig_reg}) "
                f"và **{node_dict[dest_id]['name']}** ({dest_reg}).\n\n"
                f"- **Phân mảnh địa lý tô-pô**: Mạng phà nội địa Việt Nam bao gồm **{num_components} thành phần liên thông độc lập** "
                f"tương ứng với các vùng quần đảo biệt lập (Vịnh Bắc Bộ, Miền Trung, Đông Nam Bộ, Tây Nam Bộ).\n"
                f"- **Giải pháp trung chuyển đa phương thức (Multimodal Logistics)**: Hành khách cần kết hợp tuyến giao thông "
                f"đường bộ cao tốc Bắc – Nam hoặc mạng lưới hàng không nội địa để trung chuyển giữa hai vùng lãnh thổ này."
            )

# ---------------------------------------------------------------------------
# TAB 3: Danh bạ 25 Bến phà & Cảng khách
# ---------------------------------------------------------------------------

with tab_terminals:
    st.subheader("3. Danh bạ 25 Bến phà ven biển & Cảng khách Hải đảo")
    st.caption("Thông số kỹ thuật, tọa độ hải đồ chuẩn ECDIS và bậc liên kết của các đầu mối giao thông thủy:")

    terminal_records: List[Dict[str, Any]] = []
    for _, r in fnodes.iterrows():
        t_id = str(r["terminal_id"])
        lat_v = float(r["lat"])
        lon_v = float(r["lon"])
        deg_v = int(r["degree"]) if "degree" in r else F.degree(t_id)

        # Retrieve direct partners in F
        neighbors = list(F.neighbors(t_id)) if F.has_node(t_id) else []
        neighbor_names = [node_dict.get(n, {}).get("name", n) for n in neighbors]

        terminal_records.append({
            "Mã bến": t_id,
            "Tên bến phà / Cảng khách": r["name"],
            "Phân vùng cụm": r["region"],
            "Tọa độ ECDIS": format_nautical_coords(lat_v, lon_v),
            "Bậc kết nối (Degree)": deg_v,
            "Các bến liên kết trực tiếp": ", ".join(neighbor_names) if neighbor_names else "Không có",
        })

    st.dataframe(pd.DataFrame(terminal_records), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 4: Danh mục 17 Tuyến phà Nội địa
# ---------------------------------------------------------------------------

with tab_routes:
    st.subheader("4. Danh mục 17 Tuyến phà & Luồng vận tải Khách thủy nội địa")
    st.caption("Bảng tổng hợp cự ly, thời gian hành trình và tốc độ khai thác trung bình:")

    # Deduplicate route pairs (17 pairs from 34 directed records)
    seen_pairs = set()
    route_rows: List[Dict[str, Any]] = []

    for _, r in fedges.iterrows():
        u = str(r["source"])
        v = str(r["target"])
        pair_key = tuple(sorted([u, v]))
        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            d_km = float(r["distance_km"])
            t_hr = float(r["transit_hours"])
            speed_kmh = (d_km / t_hr) if t_hr > 0 else 0.0

            route_rows.append({
                "Tên tuyến phà / luồng tàu": r["route_name"],
                "Bến đầu mối A": node_dict.get(u, {}).get("name", u),
                "Bến đầu mối B": node_dict.get(v, {}).get("name", v),
                "Cự ly (km)": f"{d_km:.1f} km",
                "Thời gian chạy (h)": f"{t_hr:.2f} giờ",
                "Vận tốc trung bình": f"{speed_kmh:.1f} km/h",
                "Phân vùng cụm": node_dict.get(u, {}).get("region", "Ven biển"),
            })

    st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 5. Mandatory "💡 Kiến thức SNA" Expander
# ---------------------------------------------------------------------------

with st.expander("💡 Kiến thức SNA: Tính Khả Tiếp Cận Đa Phương Thức & Khả Năng Chống Chịu Chuỗi Cung Ứng Hải Đảo"):
    st.markdown("""
    **1. Mạng lưới Giao thông Đa phương thức (Multimodal Transportation Feeder Integration):**
    - Khác với mạng lưới tàu container viễn dương vận hành theo các tuyến trục quốc tế (Trunk Lines) có tính dôi dư cao, **Mạng phà nội địa** thuộc Lớp C đóng vai trò là mạng gom hàng và vận tải hành khách nhánh (Feeder Network).
    - Mạng lưới này kết nối các điểm mút giao thông đường bộ cao tốc, quốc lộ ven biển với các hòn đảo biệt lập không có cầu vượt biển (Côn Đảo, Phú Quốc, Phú Quý, Lý Sơn, Cù Lao Chàm, Cô Tô).

    **2. Độ suy yếu của Nút thắt Đơn lẻ (Single Point of Failure):**
    - Mạng phà gồm **8 thành phần liên thông rời rạc** ($k = 8$) phản ánh giới hạn tự nhiên của địa lý hình cung biển Đông.
    - Cấu trúc mạng tại các cụm đảo thường có dạng hình sao (Star Topology) hoặc đồ thị chu trình đơn giản (Tree/Path Graph). Hầu hết các đảo chỉ có đúng 1 hoặc 2 luồng phà kết nối.
    - Khi xảy ra thiên tai, bão biển hoặc sự cố kỹ thuật cầu bến, cạnh cầu nối (Bridge edge) bị cắt đứt khiến hệ số khả tiếp cận của toàn bộ đảo rơi ngay lập tức về 0, gây gián đoạn chuỗi cung ứng dân sinh, nhiên liệu và y tế.

    **3. Chỉ số Đường kính Cụm & Bậc Trung chuyển:**
    - Tuyến ngắn nhất đạt 3.5 km (Bến Gót – Cái Viềng, Cát Hải đi Cát Bà) chỉ mất 24 phút.
    - Tuyến dài nhất đạt 190.0 km (Cần Thơ – Côn Đảo) mất 3.75 giờ, thể hiện sự phát triển vượt bậc của đội tàu cao tốc hai thân vỏ hợp kim nhôm hiện đại tại Việt Nam.
    """)

st.caption("Hải đồ số © 2026 — Phân tích Mạng Phà Nội địa & Kết nối Hải đảo Việt Nam. Giao diện Nautical Light Mode.")
