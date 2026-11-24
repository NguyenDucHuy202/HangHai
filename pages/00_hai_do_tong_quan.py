"""00. Hải đồ tổng quan — Global Maritime Network Overview.

Features:
- 6 Live HUD metric cards (Total Ports, Shipping Arcs, Density 4.03%, LCC 99.9%, Leiden Modularity Q 0.380, VN Ports).
- Sidebar layer controls: ports, glow, stars, arcs, searoutes bathymetry, chokepoints, seamark.
- Dynamic port sizing (PageRank, Betweenness, Degree, Closeness, k-Core) and coloring (Metric, Community, VN vs Global).
- Shipping arc limit slider (500-20,000) and checkbox "Chỉ tuyến chạm Việt Nam".
- Camera presets: Toàn cầu, Việt Nam, Đông Nam Á, 3D Nghiêng 45°.
- Top 20 Global Ports table with row selection navigation to Page 01 (VN ports highlighted in red).
- Interactive TripsLayer tab running `render_trips_animation()`.
- Comprehensive "💡 Kiến thức SNA" expander.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

from core.data import (
    load_nodes,
    load_edges,
    load_edges_geo,
    load_structure_stats,
    load_community_eval,
)
from core.theme import (
    apply_theme,
    CHOKEPOINTS,
    RGB_VN_RED,
    RGB_PRIMARY_BLUE,
    RGB_AMBER,
    RGB_STAR,
)
from components.hud import render_hud
from components.layers import (
    ports_layer,
    ports_glow_layer,
    vn_stars_layer,
    arcs_layer,
    searoute_layer,
    chokepoints_layer,
    seamark_layer,
)
from components.deck_map import render_deck, VIEW_GLOBAL, VIEW_VIETNAM, VIEW_SEA, VIEW_3D
from components.animated_map import render_trips_animation

# Apply Light Mode design system
apply_theme()

st.title("Hải đồ tổng quan")
st.caption("Bản đồ tương tác mạng lưới vận tải hàng hải toàn cầu & Việt Nam — Hệ thống phân tích SNA")

# 1. Load data
nodes_df = load_nodes()
edges_df = load_edges()
vn_nodes = nodes_df[nodes_df.get("is_vn", False) == True]
stats = load_structure_stats()
comm_eval = load_community_eval()

total_ports = len(nodes_df)
total_vn = len(vn_nodes)
total_edges = len(edges_df)

density_val = float(stats.get("density", total_edges / (total_ports * (total_ports - 1))))
lcc_nodes = int(stats.get("lcc_nodes", total_ports))
lcc_pct = (lcc_nodes / total_ports) * 100.0

leiden_q = 0.3802
if comm_eval is not None and len(comm_eval) > 0 and "algorithm" in comm_eval.columns:
    leiden_row = comm_eval[comm_eval["algorithm"].str.lower() == "leiden"]
    if len(leiden_row) > 0:
        leiden_q = float(leiden_row.iloc[0]["modularity"])

# 2. Render 6 HUD Metric Cards (§10.0)
hud_cards = [
    {
        "title": "TỔNG SỐ CẢNG BIỂN",
        "value": f"{total_ports:,}",
        "delta": "Toàn cầu",
        "delta_color": "neutral",
        "icon": "globe",
        "tooltip": "Tổng số lượng cảng biển trong mạng lưới dữ liệu SNA toàn cầu",
    },
    {
        "title": "TUYẾN KẾT NỐI HÀNG HẢI",
        "value": f"{total_edges:,}",
        "delta": "Có hướng",
        "delta_color": "neutral",
        "icon": "git-commit",
        "tooltip": "Tổng số luồng tuyến đường biển trực tiếp kết nối giữa các cặp cảng",
    },
    {
        "title": "MẬT ĐỘ MẠNG LƯỚI",
        "value": f"{density_val * 100:.2f}%",
        "delta": f"{density_val:.4f}",
        "delta_color": "normal",
        "icon": "network",
        "tooltip": "Tỷ lệ giữa số cạnh hiện có so với tổng số cạnh tối đa có thể có: Density = |E| / (|V|*(|V|-1))",
    },
    {
        "title": "THÀNH PHẦN LỚN NHẤT (LCC)",
        "value": f"{lcc_pct:.1f}%",
        "delta": f"{lcc_nodes:,} cảng",
        "delta_color": "normal",
        "icon": "anchor",
        "tooltip": "Tỷ lệ cảng biển nằm trong thành phần liên thông lớn nhất (Giant Component)",
    },
    {
        "title": "MODULARITY Q (LEIDEN)",
        "value": f"{leiden_q:.3f}",
        "delta": "16 cộng đồng",
        "delta_color": "normal",
        "icon": "activity",
        "tooltip": "Chỉ số module hóa của thuật toán phát hiện cộng đồng Leiden",
    },
    {
        "title": "CẢNG BIỂN VIỆT NAM",
        "value": f"{total_vn:,}",
        "delta": f"{total_vn / total_ports * 100:.1f}%",
        "delta_color": "normal",
        "icon": "flag",
        "tooltip": "Số lượng cảng biển và bến chuyên dùng ven biển Việt Nam trong mạng lưới",
    },
]
render_hud(hud_cards)

# 3. Sidebar Filter Controls (§10.0, §9.1, §9.2)
with st.sidebar:
    st.markdown("### ⚙️ Điều khiển Hải đồ")

    st.markdown("#### 1. Bộ lọc Luồng tuyến & Quy mô")
    arc_limit = st.slider(
        "Số lượng tuyến hiển thị (Arcs):",
        min_value=500,
        max_value=20000,
        value=1500,
        step=500,
        help="Số tuyến có trọng số cao nhất được dựng trên bản đồ cầu lớn (Great-Circle)",
    )
    vn_only_arcs = st.checkbox(
        "Chỉ tuyến chạm Việt Nam",
        value=False,
        help="Lọc các tuyến đường biển có điểm xuất phát hoặc điểm đến thuộc Việt Nam",
    )

    st.markdown("#### 2. Kích thước & Màu sắc Cảng")
    size_metric_map = {
        "PageRank (Mức độ uy tín)": "pagerank",
        "Betweenness (Độ trung gian)": "betweenness",
        "Degree (Tổng bậc kết nối)": "degree",
        "Closeness (Tốc độ tiếp cận)": "closeness_time",
        "k-Core (Cấp độ lõi mạng)": "core_number",
    }
    size_choice = st.selectbox("Kích thước cảng theo:", list(size_metric_map.keys()), index=0)
    selected_metric_col = size_metric_map[size_choice]

    color_mode_map = {
        "Thang độ đo (Oceanic Blue)": "metric",
        "Cộng đồng mạng (Leiden)": "community",
        "Phân vùng (Việt Nam vs Quốc tế)": "is_vn",
    }
    color_choice = st.selectbox("Màu sắc cảng theo:", list(color_mode_map.keys()), index=0)
    selected_color_mode = color_mode_map[color_choice]

    st.markdown("#### 3. Bật / Tắt các Lớp Bản đồ")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        show_ports = st.checkbox("Cảng biển", value=True)
        show_glow = st.checkbox("Hào quang Top Hubs", value=True)
        show_vn_stars = st.checkbox("Đánh dấu sao VN", value=True)
        show_arcs = st.checkbox("Luồng tuyến biển (Arcs)", value=True)
    with col_l2:
        show_searoutes = st.checkbox("Tuyến biển thật (Searoute)", value=False)
        show_chokepoints = st.checkbox("Điểm nghẽn (Chokepoints)", value=True)
        show_seamarks = st.checkbox("Hải tiêu OpenSeaMap", value=False)

    st.markdown("#### 4. Phong cách Bản đồ Nền")
    basemap_choice = st.selectbox(
        "Lớp bản đồ nền:",
        [
            "Ảnh Vệ tinh ESRI (ESRI Satellite)",
            "Hải đồ Sống động (Vivid Ocean - Voyager)",
            "Hải đồ Sáng Tối giản (Nautical Light - Positron)",
            "Hải đồ Đêm (Dark Ocean)",
        ],
        index=0,
        help="Ảnh vệ tinh ESRI viễn thám thực tế hoặc hải đồ vector chuẩn Carto không nhãn",
    )
    basemap_key_map = {
        "Ảnh Vệ tinh ESRI (ESRI Satellite)": "satellite",
        "Hải đồ Sống động (Vivid Ocean - Voyager)": "voyager",
        "Hải đồ Sáng Tối giản (Nautical Light - Positron)": "light",
        "Hải đồ Đêm (Dark Ocean)": "dark",
    }
    selected_map_style = basemap_key_map.get(basemap_choice, "satellite")

    st.markdown("#### 5. Phối cảnh Camera")
    camera_choice = st.selectbox(
        "Góc nhìn phối cảnh:",
        ["Việt Nam", "Đông Nam Á", "Toàn cầu", "3D Nghiêng 45°"],
        index=0,
    )

# 4. Main visualization tabs
tab_map, tab_trips = st.tabs(["Hải đồ mạng lưới", "Hoạt ảnh luồng tàu (TripsLayer)"])

with tab_map:
    st.info("🇻🇳 **Khẳng định Chủ quyền Biển Đảo Quốc gia**: Bản đồ thể hiện đầy đủ và nhất quán chủ quyền toàn vẹn lãnh thổ của Việt Nam đối với **Quần đảo Hoàng Sa** (TP. Đà Nẵng), **Quần đảo Trường Sa** (Tỉnh Khánh Hòa) và toàn bộ vùng biển **Biển Đông (Việt Nam)** theo luật pháp quốc tế và UNCLOS 1982.")

    # Assemble Pydeck layers dynamically
    active_layers = []

    # 1. Seamark TileLayer
    if show_seamarks:
        active_layers.append(seamark_layer())

    # 2. Realistic Searoutes PathLayer
    if show_searoutes:
        edges_geo = load_edges_geo()
        active_layers.append(searoute_layer(edges_geo, color=[0, 119, 182, 160], width=2, top_n=3000))

    # 3. Arcs Layer
    if show_arcs:
        active_layers.append(
            arcs_layer(edges_df, limit=arc_limit, vn_only=vn_only_arcs, nodes_df=nodes_df)
        )

    # 4. Ports Halo / Glow for top hubs
    if show_glow:
        active_layers.append(ports_glow_layer(nodes_df, top_n=30))

    # 5. Ports ScatterplotLayer
    if show_ports:
        active_layers.append(
            ports_layer(nodes_df, metric_col=selected_metric_col, color_by=selected_color_mode)
        )

    # 6. Vietnam Stars Layer
    if show_vn_stars:
        active_layers.append(vn_stars_layer(nodes_df))

    # 7. Strategic Chokepoints
    if show_chokepoints:
        active_layers.extend(chokepoints_layer())

    # Resolve camera view
    view_states = {
        "Toàn cầu": pdk.ViewState(latitude=15.0, longitude=108.0, zoom=2.2, pitch=0, bearing=0),
        "Việt Nam": pdk.ViewState(latitude=16.0, longitude=108.0, zoom=5.0, pitch=0, bearing=0),
        "Đông Nam Á": pdk.ViewState(latitude=5.0, longitude=105.0, zoom=3.8, pitch=0, bearing=0),
        "3D Nghiêng 45°": pdk.ViewState(latitude=15.0, longitude=108.0, zoom=3.2, pitch=45, bearing=-15),
    }
    view_state = view_states.get(camera_choice, view_states["Toàn cầu"])

    deck = render_deck(active_layers, view_state=view_state, map_style=selected_map_style, height=640)
    st.pydeck_chart(deck, use_container_width=True)

    # Top 20 Global Ports Ranking Table with interactive navigation
    st.markdown("### 🏆 Bảng xếp hạng Top 20 Cảng biển Toàn cầu")
    st.caption("Cảng biển Việt Nam được tô sáng nổi bật màu đỏ. Bạn có thể chọn cảng để xem hồ sơ phân tích chi tiết.")

    top_cols = ["node_id", "name", "country", "iso3", "pagerank", "betweenness", "degree", "core_number", "is_vn"]
    avail_cols = [c for c in top_cols if c in nodes_df.columns]
    top20 = nodes_df[avail_cols].sort_values(by="pagerank", ascending=False).head(20).copy().reset_index(drop=True)
    top20.insert(0, "rank", range(1, len(top20) + 1))

    # Format columns for display
    display_df = top20.rename(columns={
        "rank": "Hạng",
        "node_id": "Mã cảng",
        "name": "Tên cảng biển",
        "country": "Quốc gia",
        "iso3": "ISO3",
        "pagerank": "PageRank",
        "betweenness": "Betweenness",
        "degree": "Bậc kết nối",
        "core_number": "k-Core",
        "is_vn": "Việt Nam",
    })

    # Highlight Vietnamese ports in soft red
    def highlight_vn_rows(row):
        if row.get("Việt Nam") is True or row.get("Quốc gia") == "Vietnam":
            return ["background-color: #FEF2F2; font-weight: bold; color: #DC2626"] * len(row)
        return [""] * len(row)

    styled_df = display_df.style.apply(highlight_vn_rows, axis=1).format({
        "PageRank": "{:.5f}",
        "Betweenness": "{:.5f}",
        "Bậc kết nối": "{:,}",
    })

    # Interactive single-row selection
    event = st.dataframe(
        styled_df,
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        hide_index=True,
    )

    # Navigation trigger
    selected_port_from_table = None
    if event and hasattr(event, "selection") and event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        selected_port_from_table = top20.iloc[row_idx]["node_id"]

    c_nav1, c_nav2 = st.columns([3, 1])
    with c_nav1:
        default_nav_idx = 0
        if selected_port_from_table and selected_port_from_table in top20["node_id"].values:
            matched_indices = top20[top20["node_id"] == selected_port_from_table].index
            if len(matched_indices) > 0:
                default_nav_idx = int(matched_indices[0])

        chosen_port_id = st.selectbox(
            "Chọn cảng để tra cứu chuyên sâu hồ sơ kết nối:",
            options=top20["node_id"].tolist(),
            index=default_nav_idx,
            format_func=lambda x: f"#{top20.loc[top20['node_id']==x, 'rank'].values[0]} {top20.loc[top20['node_id']==x, 'name'].values[0]} ({top20.loc[top20['node_id']==x, 'country'].values[0]}) — {x}",
        )
    with c_nav2:
        st.write("")
        st.write("")
        if st.button("🔎 Mở Hồ sơ cảng biển", type="primary", use_container_width=True):
            st.query_params["port"] = chosen_port_id
            st.switch_page("pages/01_ho_so_cang.py")

with tab_trips:
    st.markdown("#### 🚢 Hoạt ảnh Luồng tàu Di chuyển qua WebGL TripsLayer")
    st.info("Mô phỏng 400 hành trình tàu hàng hải toàn cầu với thanh điều khiển tua thời gian, tốc độ (1x–10x) và đồng hồ UTC.")
    render_trips_animation(height=650)

# 5. Mandatory SNA Knowledge Expander
with st.expander("💡 Kiến thức SNA: Cấu trúc Tô-pô & Độ đo Trung tâm Mạng Hàng hải"):
    st.markdown("""
    **1. Biểu diễn Mạng Vận tải Hàng hải dưới dạng Đồ thị:**
    Mạng lưới hàng hải được mô hình hóa thành đồ thị có hướng và trọng số $G = (V, E, W)$, trong đó:
    - Tập đỉnh $V$: Danh sách 2.424 cảng biển toàn cầu với tọa độ địa lý $(\\text{lat}, \\text{lon})$.
    - Tập cạnh $E$: 169.888 luồng di chuyển đường biển trực tiếp giữa các cặp cảng.
    - Trọng số $W$: Tần suất chuyến đi (vessel voyages), khoảng cách thực tế ($km$) và thời gian hành trình (transit days).

    **2. Mật độ Mạng lưới (Network Density):**
    $$Density = \\frac{|E|}{|V|(|V|-1)} = \\frac{169.888}{2.424 \\times 2.423} \\approx 2.89\\% \\quad (\\text{hoặc } 4.03\\% \\text{ trên đồ thị vô hướng đa cạnh})$$
    Phản ánh mức độ liên kết thực tế so với đồ thị đầy đủ. Mật độ thấp là đặc trưng tiêu biểu của mạng vận tải không gian (spatial networks) do chi phí mở tuyến vượt đại dương lớn.

    **3. Thành phần Liên thông Lớn nhất (Largest Connected Component - LCC):**
    Chiếm $99.9\\%$ (2.422 / 2.424 cảng), cho thấy mạng lưới hàng hải toàn cầu tích hợp cao, hầu như mọi cảng biển thương mại đều có thể tiếp cận lẫn nhau thông qua hệ thống trung chuyển.

    **4. Thuật toán PageRank trong Mạng Cảng biển:**
    $$PR(u) = \\frac{1-d}{|V|} + d \\sum_{v \\in In(u)} \\frac{PR(v)}{Out(v)}$$
    Với hệ số suy giảm $d = 0.85$. PageRank đo lường tầm quan trọng tích lũy của một cảng biển dựa trên chất lượng và uy tín của các cảng đối tác kết nối tới nó, giúp nhận diện chính xác các siêu cảng trung chuyển (transshipment hubs) như Singapore, Rotterdam hay Thượng Hải.

    **5. Độ trung gian Betweenness Centrality:**
    $$C_B(v) = \\sum_{s \\neq v \\neq t} \\frac{\\sigma_{st}(v)}{\\sigma_{st}}$$
    Đo lường tần suất một cảng nằm trên đường đi ngắn nhất giữa tất cả các cặp cảng khác trong mạng lưới toàn cầu, phản ánh vai trò "người gác cổng" (gatekeeper) của các eo biển và cảng trung chuyển chiến lược.
    """)

st.caption("Nguồn dữ liệu: IMF PortWatch, UNCTADstat, Global Fishing Watch, OpenSeaMap. Khẳng định chủ quyền: Hoàng Sa & Trường Sa là của Việt Nam | Biển Đông (Việt Nam).")
