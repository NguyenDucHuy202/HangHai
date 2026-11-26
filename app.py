"""Hải đồ số — Maritime Transport Social Network Analysis (SNA) Web Application.

Main Streamlit Application Shell:
- Light Mode Design System with Nautical Palette.
- Modern Multi-Page Navigation API (`st.navigation`, `st.Page`).
- Sidebar branding with live dataset statistics from manifest.json.
- Academic citations and data lineage attributions (§16).
"""

from __future__ import annotations

import streamlit as st
from core.data import load_manifest
from core.theme import apply_theme

# 1. Global Streamlit Page Configuration (must be called first)
st.set_page_config(
    page_title="Hải đồ số - Phân tích mạng vận tải hàng hải",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Nautical Light Mode Theme & Fonts
apply_theme()

# 3. Sidebar Branding & Dataset KPI Summary
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="width: 36px; height: 36px; border-radius: 8px; background: linear-gradient(135deg, #0077B6 0%, #023E8A 100%); display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 6px rgba(0, 119, 182, 0.25);">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="5" r="3"/>
                    <line x1="12" y1="22" x2="12" y2="8"/>
                    <path d="M5 12H2a10 10 0 0 0 20 0h-3"/>
                </svg>
            </div>
            <div>
                <div class="sidebar-brand-title">HẢI ĐỒ SỐ</div>
                <div style="font-size: 11px; font-weight: 600; color: #0284C7; letter-spacing: 0.05em; text-transform: uppercase;">Maritime SNA Platform</div>
            </div>
        </div>
        <div class="sidebar-brand-subtitle" style="margin-top: 8px;">
            Hệ thống Phân tích Mạng Vận tải Hàng hải Toàn cầu & Việt Nam
        </div>
        <div class="sidebar-badge">v2.0 · Light Mode</div>
    </div>
    """, unsafe_allow_html=True)

    # Load and display manifest summary
    try:
        manifest = load_manifest()
        stats = manifest.get("dataset_stats", {})
        node_cnt = stats.get("nodes", 2424)
        edge_cnt = stats.get("edges", 169888)
        vn_cnt = stats.get("vn_nodes", 261)
    except Exception:
        node_cnt, edge_cnt, vn_cnt = 2424, 169888, 261

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px;">
        <div style="font-size: 11px; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 6px;">QUY MÔ DỮ LIỆU ĐỒ THỊ</div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
            <span style="color: #475569;">Cảng biển (|V|):</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0077B6;">{node_cnt:,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 3px;">
            <span style="color: #475569;">Tuyến kết nối (|E|):</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0077B6;">{edge_cnt:,}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 12px;">
            <span style="color: #475569;">Cảng Việt Nam:</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #DA251D;">{vn_cnt:,}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 4. Declare Multi-Page Navigation Structure with Professional Material Icons
pages = {
    "Tổng quan & Hồ sơ": [
        st.Page("pages/00_hai_do_tong_quan.py", title="Hải đồ tổng quan", icon=":material/explore:", default=True),
        st.Page("pages/01_ho_so_cang.py", title="Hồ sơ cảng biển", icon=":material/dock:"),
    ],
    "Điều hướng & Tuyến hàng hải": [
        st.Page("pages/02_tim_tuyen.py", title="Tìm tuyến đường biển", icon=":material/route:"),
        st.Page("pages/05_du_doan_tuyen_moi.py", title="Dự đoán tuyến mới (AI)", icon=":material/auto_awesome:"),
    ],
    "Cấu trúc & Phân tích mạng": [
        st.Page("pages/03_cong_dong.py", title="Phát hiện cộng đồng", icon=":material/hub:"),
        st.Page("pages/08_cau_truc_mang.py", title="Cấu trúc mạng toàn cầu", icon=":material/analytics:"),
    ],
    "An ninh & Mô phỏng sự cố": [
        st.Page("pages/04_mo_phong_su_co.py", title="Mô phỏng sự cố & gián đoạn", icon=":material/warning:"),
    ],
    "Thời gian & Nội địa": [
        st.Page("pages/06_viet_nam_toan_cau.py", title="Việt Nam toàn cầu", icon=":material/public:"),
        st.Page("pages/07_pha_noi_dia.py", title="Mạng phà nội địa", icon=":material/directions_boat:"),
    ],
    "Phương pháp & Dữ liệu": [
        st.Page("pages/09_phuong_phap.py", title="Phương pháp & Dữ liệu", icon=":material/science:"),
    ],
}

# 5. Initialize and Run Multi-Page Navigation
nav = st.navigation(pages)

# Sidebar Footer Attributions
with st.sidebar:
    st.markdown("---")
    st.caption("© 2026 Hải đồ số · Đồ án Phân tích Mạng Xã hội")
    st.caption("Dữ liệu: IMF PortWatch, UNCTADstat, Global Fishing Watch, OpenStreetMap, OpenSeaMap.")

nav.run()
