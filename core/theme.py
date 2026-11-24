"""Nautical Light Mode Design System, Color Tokens, and UI Styling for Hải đồ số.

Provides:
- Semantic color tokens (Hex and RGB) conforming to Nautical Light Mode.
- 20-color high-contrast categorical community palette.
- Strategic maritime chokepoint coordinates and descriptions.
- Camera ViewState presets for Pydeck / Deck.gl.
- Nautical coordinate formatter (ECDIS DD°MM.M′N/S format).
- Global CSS injection for Google Fonts, clean cards, HUD banners, and scrollbars.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st

# ===========================================================================
# 1. Semantic Color Tokens (Light Mode Nautical Palette)
# ===========================================================================

COLOR_BG = "#FFFFFF"                  # Pure white canvas background
COLOR_BG_SECONDARY = "#F8FAFC"        # Slate-50 secondary background
COLOR_CARD = "#F8FAFC"                # Slate-50 card surface
COLOR_CARD_BORDER = "#E2E8F0"         # Slate-200 card border
COLOR_BORDER = "#E2E8F0"              # Alias for general border
COLOR_PRIMARY_BLUE = "#0077B6"        # Oceanic blue (primary accents & selected routes)
COLOR_PRIMARY_DARK = "#03045E"        # Deep maritime navy for headers
COLOR_VN_RED = "#DA251D"              # Vietnam national red for VN ports and links
COLOR_AMBER = "#F59E0B"               # Warning / link prediction recommendations
COLOR_TEAL = "#0D9488"                # PortWatch connections / ferry routes
COLOR_STAR = "#EAB308"                # Golden yellow star markers for Vietnam
COLOR_TEXT = "#0F172A"                # Slate-900 primary typography
COLOR_TEXT_PRIMARY = "#0F172A"        # Slate-900 alias
COLOR_TEXT_SECONDARY = "#475569"      # Slate-600 secondary typography
COLOR_MUTED = "#64748B"               # Slate-500 muted text
COLOR_GRID = "#CBD5E1"                # Slate-300 graticule / divider lines
COLOR_DISRUPTED = "#EF4444"           # Red alert for severed / disrupted ports
COLOR_SUCCESS = "#059669"             # Emerald green for positive metrics / normal delta

# ===========================================================================
# 2. RGB Equivalents for Pydeck / Deck.gl (0-255 lists)
# ===========================================================================

def hex_to_rgb(hex_str: str, alpha: int = 255) -> List[int]:
    """Converts a hex color string ('#RRGGBB') to an [R, G, B, A] list."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join([c * 2 for c in hex_str])
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    if alpha < 255:
        return [r, g, b, alpha]
    return [r, g, b]


RGB_BG = [255, 255, 255]
RGB_PRIMARY_BLUE = [0, 119, 182]
RGB_PRIMARY_DARK = [3, 4, 94]
RGB_VN_RED = [218, 37, 29]
RGB_AMBER = [245, 158, 11]
RGB_TEAL = [13, 148, 136]
RGB_STAR = [234, 179, 8]
RGB_DISRUPTED = [239, 68, 68]
RGB_MUTED = [100, 116, 139]
RGB_GRID = [203, 213, 225]

# ===========================================================================
# 3. 20-Color High-Contrast Categorical Community Palette
# ===========================================================================

COMMUNITY_PALETTE_20: List[str] = [
    "#0077B6",  # 0: Oceanic Blue
    "#0D9488",  # 1: Deep Teal
    "#D97706",  # 2: Amber Orange
    "#DC2626",  # 3: Crimson Red
    "#7C3AED",  # 4: Royal Violet
    "#2563EB",  # 5: Cobalt Blue
    "#059669",  # 6: Emerald Green
    "#EA580C",  # 7: Rust Orange
    "#9333EA",  # 8: Purple
    "#0891B2",  # 9: Cyan Sea
    "#4F46E5",  # 10: Indigo
    "#16A34A",  # 11: Sea Green
    "#CA8A04",  # 12: Mustard
    "#C026D3",  # 13: Fuchsia
    "#0284C7",  # 14: Sky Blue
    "#65A30D",  # 15: Lime Olive
    "#B45309",  # 16: Bronze
    "#475569",  # 17: Slate Gray
    "#BE123C",  # 18: Rose Red
    "#334155",  # 19: Deep Slate
]

COMMUNITY_PALETTE_RGB: List[List[int]] = [hex_to_rgb(c) for c in COMMUNITY_PALETTE_20]


def get_community_color(comm_id: int, as_rgb: bool = False, alpha: int = 255) -> Union[str, List[int]]:
    """Returns the categorical color for a given community ID (modulo 20)."""
    idx = int(comm_id) % len(COMMUNITY_PALETTE_20)
    if as_rgb:
        rgb = hex_to_rgb(COMMUNITY_PALETTE_20[idx])
        if alpha < 255:
            return rgb + [alpha] if len(rgb) == 3 else rgb[:3] + [alpha]
        return rgb
    return COMMUNITY_PALETTE_20[idx]


# ===========================================================================
# 4. Strategic Maritime Chokepoints Catalog
# ===========================================================================

CHOKEPOINTS: List[Dict[str, Any]] = [
    {
        "id": "malacca",
        "name": "Eo Malacca",
        "en_name": "Strait of Malacca",
        "lon": 101.35,
        "lat": 2.50,
        "traffic_share": "25% thương mại toàn cầu",
        "desc": "Tuyến hàng hải nhộn nhịp nhất thế giới kết nối Ấn Độ Dương và Biển Đông / Thái Bình Dương.",
    },
    {
        "id": "suez",
        "name": "Kênh Suez",
        "en_name": "Suez Canal",
        "lon": 32.34,
        "lat": 30.00,
        "traffic_share": "12% thương mại toàn cầu",
        "desc": "Hành lang hàng hải Á - Âu rút ngắn 7.000 km so với hải trình vòng qua Mũi Hảo Vọng.",
    },
    {
        "id": "panama",
        "name": "Kênh Panama",
        "en_name": "Panama Canal",
        "lon": -79.91,
        "lat": 9.10,
        "traffic_share": "5% thương mại toàn cầu",
        "desc": "Kênh đào kết nối Thái Bình Dương và Đại Tây Dương, khống chế luồng vận tải Bờ Đông - Châu Á.",
    },
    {
        "id": "hormuz",
        "name": "Eo Hormuz",
        "en_name": "Strait of Hormuz",
        "lon": 56.40,
        "lat": 26.56,
        "traffic_share": "20% sản lượng dầu mỏ thế giới",
        "desc": "Huyết mạch vận tải dầu thô và khí hóa lỏng từ Vịnh Ba Tư ra thị trường quốc tế.",
    },
    {
        "id": "bab_el_mandeb",
        "name": "Eo Bab-el-Mandeb",
        "en_name": "Bab-el-Mandeb Strait",
        "lon": 43.33,
        "lat": 12.58,
        "traffic_share": "Cửa ngõ Biển Đỏ",
        "desc": "Nút thắt chiến lược giữa Vịnh Aden và Biển Đỏ dẫn vào Kênh đào Suez.",
    },
    {
        "id": "gibraltar",
        "name": "Eo Gibraltar",
        "en_name": "Strait of Gibraltar",
        "lon": -5.60,
        "lat": 35.98,
        "traffic_share": "Cửa ngõ Địa Trung Hải",
        "desc": "Lối vào Địa Trung Hải từ Đại Tây Dương với hơn 100.000 lượt tàu qua lại mỗi năm.",
    },
    {
        "id": "bosphorus",
        "name": "Eo Bosphorus",
        "en_name": "Bosphorus Strait",
        "lon": 29.08,
        "lat": 41.12,
        "traffic_share": "Cửa ngõ Biển Đen",
        "desc": "Tuyến đường thủy duy nhất kết nối Biển Đen với Địa Trung Hải qua Biển Marmara.",
    },
]

# ===========================================================================
# 5. Pydeck Camera ViewState Presets
# ===========================================================================

VIEW_PRESETS: Dict[str, Dict[str, Any]] = {
    "vietnam": {"latitude": 16.0, "longitude": 108.0, "zoom": 5.0, "pitch": 0, "bearing": 0},
    "sea": {"latitude": 5.0, "longitude": 105.0, "zoom": 3.8, "pitch": 0, "bearing": 0},
    "global": {"latitude": 12.0, "longitude": 112.0, "zoom": 2.2, "pitch": 0, "bearing": 0},
    "3d": {"latitude": 12.0, "longitude": 112.0, "zoom": 3.0, "pitch": 45, "bearing": -15},
}

# ===========================================================================
# 6. Nautical Coordinate Formatter
# ===========================================================================

def format_nautical_coords(lat: float, lon: float) -> str:
    """Formats decimal latitude and longitude into ECDIS nautical format.
    
    Example:
        format_nautical_coords(20.85, 106.70) -> "20°51.0′N 106°42.0′E"
    """
    lat_val = float(lat)
    lon_val = float(lon)

    lat_hemi = "N" if lat_val >= 0 else "S"
    lat_abs = abs(lat_val)
    lat_deg = int(lat_abs)
    lat_min = (lat_abs - lat_deg) * 60.0

    lon_hemi = "E" if lon_val >= 0 else "W"
    lon_abs = abs(lon_val)
    lon_deg = int(lon_abs)
    lon_min = (lon_abs - lon_deg) * 60.0

    return f"{lat_deg:02d}°{lat_min:04.1f}′{lat_hemi} {lon_deg:03d}°{lon_min:04.1f}′{lon_hemi}"

# ===========================================================================
# 7. Global CSS Injection & Light Mode Theme Applier
# ===========================================================================

def get_global_css() -> str:
    """Generates global CSS stylesheet for Nautical Light Mode theme.
    
    Includes:
    - Google Fonts imports ('Be Vietnam Pro' and 'JetBrains Mono').
    - Clean card styling, border radiuses, and light shadows.
    - Responsive HUD metric badge classes.
    - Custom scrollbars and refined Streamlit UI overrides.
    """
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global typography */
    html, body, [class*="css"], .stMarkdown, p, span, label, div {{
        font-family: 'Be Vietnam Pro', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Main canvas light styling */
    .stApp {{
        background-color: {COLOR_BG};
        color: {COLOR_TEXT};
    }}

    /* Monospace for numbers, metrics, code, timestamps, coordinates */
    .font-mono, [data-testid="stMetricValue"], .nautical-coords, .mono-val {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Clean Light Mode Card */
    .haido-card {{
        background: {COLOR_CARD};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }}

    /* HUD Metrics Banner */
    .haido-hud-container {{
        display: flex;
        flex-direction: row;
        gap: 12px;
        flex-wrap: wrap;
        width: 100%;
        margin-top: 4px;
        margin-bottom: 18px;
    }}

    .haido-hud-card {{
        flex: 1 1 180px;
        min-width: 160px;
        background: {COLOR_CARD};
        border: 1px solid {COLOR_CARD_BORDER};
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .haido-hud-card:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }}

    .haido-hud-title {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: {COLOR_MUTED};
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    .haido-hud-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: {COLOR_PRIMARY_BLUE};
        line-height: 1.15;
    }}

    .haido-hud-delta {{
        font-size: 11px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 2px;
        padding: 2px 6px;
        border-radius: 4px;
        margin-top: 4px;
        align-self: flex-start;
    }}

    .haido-hud-delta.delta-normal {{
        background: #ECFDF5;
        color: #059669;
    }}

    .haido-hud-delta.delta-inverse {{
        background: #FEF2F2;
        color: #DC2626;
    }}

    .haido-hud-delta.delta-neutral {{
        background: #F1F5F9;
        color: #64748B;
    }}

    /* SNA Knowledge Expander */
    .stExpander {{
        border: 1px solid #BAE6FD !important;
        border-radius: 8px !important;
        background-color: #F0F9FF !important;
        margin-top: 14px;
        margin-bottom: 14px;
    }}

    .stExpander summary {{
        font-weight: 600;
        color: #0369A1 !important;
    }}

    /* Refined custom scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: #F1F5F9;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #CBD5E1;
        border-radius: 3px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #94A3B8;
    }}

    /* Sidebar nautical styling */
    [data-testid="stSidebar"] {{
        background-color: {COLOR_CARD};
        border-right: 1px solid {COLOR_CARD_BORDER};
    }}

    .sidebar-brand-title {{
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: {COLOR_PRIMARY_DARK};
        margin-bottom: 2px;
    }}

    .sidebar-brand-subtitle {{
        font-size: 12px;
        color: {COLOR_MUTED};
        margin-bottom: 12px;
    }}

    .sidebar-badge {{
        display: inline-block;
        padding: 2px 8px;
        background: #E0F2FE;
        color: #0369A1;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        margin-bottom: 10px;
    }}
    </style>
    """


def apply_theme() -> None:
    """Injects Nautical Light Mode CSS into the current Streamlit page."""
    st.markdown(get_global_css(), unsafe_allow_html=True)
