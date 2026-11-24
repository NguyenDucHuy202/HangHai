"""Responsive Light Mode HUD Metric Card Banner Component for Hải đồ số.

Renders high-visibility metric summary cards with:
- JetBrains Mono 24px bold values in Oceanic Blue (#0077B6).
- Uppercase slate-500 labels.
- Delta indicator pill badges (green for normal, red for inverse, slate for neutral).
- Semantic icons and optional tooltips.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional
import streamlit as st

def _svg(paths: str, stroke: str = "#0284C7", fill: str = "none") -> str:
    """Generates a crisp Lucide-grade vector SVG icon for HUD metrics."""
    return (
        f'<svg width="15" height="15" viewBox="0 0 24 24" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display: inline-block; vertical-align: -2px; margin-right: 5px;">{paths}</svg>'
    )


ICON_MAP: Dict[str, str] = {
    "anchor": _svg('<circle cx="12" cy="5" r="3"/><line x1="12" y1="22" x2="12" y2="8"/><path d="M5 12H2a10 10 0 0 0 20 0h-3"/>'),
    "flag": _svg('<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/>', stroke="#DC2626"),
    "git-commit": _svg('<circle cx="12" cy="12" r="4"/><line x1="1.05" y1="12" x2="7" y2="12"/><line x1="17.01" y1="12" x2="22.96" y2="12"/>'),
    "ship": _svg('<path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76"/><path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/><line x1="12" y1="1" x2="12" y2="5"/>'),
    "ferry": _svg('<path d="M2 21c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1 .6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M19.38 20A11.6 11.6 0 0 0 21 14l-9-4-9 4c0 2.9.94 5.34 2.81 7.76"/><path d="M19 13V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6"/><line x1="12" y1="1" x2="12" y2="5"/>'),
    "compass": _svg('<circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76" fill="#0284C7"/>'),
    "activity": _svg('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>'),
    "network": _svg('<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>'),
    "alert-triangle": _svg('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', stroke="#D97706"),
    "alert": _svg('<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', stroke="#D97706"),
    "clock": _svg('<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'),
    "database": _svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'),
    "layers": _svg('<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>'),
    "route": _svg('<circle cx="6" cy="19" r="3"/><path d="M9 19h8.5a3.5 3.5 0 0 0 0-7h-11a3.5 3.5 0 0 1 0-7H15"/><circle cx="18" cy="5" r="3"/>'),
    "zap": _svg('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="#F59E0B" stroke="#F59E0B"/>'),
    "check": _svg('<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>', stroke="#059669"),
    "globe": _svg('<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>'),
    "star": _svg('<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" fill="#F59E0B" stroke="#F59E0B"/>'),
    "brain": _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
    "ai": _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>'),
    "target": _svg('<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2" fill="#0284C7"/>'),
    "port": _svg('<path d="M2 20h20"/><path d="M5 20V8l7-4 7 4v12"/><path d="M9 20v-6h6v6"/>'),
    "science": _svg('<path d="M10 2v7.31L4.1 20.45A1.5 1.5 0 0 0 5.38 22.5h13.24a1.5 1.5 0 0 0 1.28-2.05L14 9.31V2"/><line x1="8" y1="2" x2="16" y2="2"/>'),
}


def render_hud(cards: List[Dict[str, Any]]) -> None:
    """Renders a responsive Light Mode HUD banner with JetBrains Mono numbers.

    Parameters
    ----------
    cards : list[dict]
        List of card dictionaries. Each dictionary can contain:
        - title : str (Label, e.g. 'TỔNG SỐ CẢNG')
        - value : str | int | float (Main metric, e.g. '2,425')
        - delta : Optional[str] (Change or ratio, e.g. '+12', '10.8%')
        - delta_color : Optional[str] ('normal', 'inverse', 'neutral', 'off')
        - icon : Optional[str] (Icon key or unicode emoji)
        - tooltip : Optional[str] (Tooltip hint)
    """
    if not cards:
        return

    card_items_html: List[str] = []

    for card in cards:
        raw_title = str(card.get("title", ""))
        raw_val = str(card.get("value", ""))
        raw_delta = card.get("delta")
        delta_color = str(card.get("delta_color", "normal")).lower()
        icon_key = card.get("icon")
        tooltip = card.get("tooltip")

        # Resolve icon (Supports vector SVG and string fallbacks)
        icon_html = ""
        if icon_key:
            resolved_icon = ICON_MAP.get(str(icon_key).lower())
            if resolved_icon and resolved_icon.startswith("<svg"):
                icon_html = resolved_icon
            elif resolved_icon:
                icon_html = f"<span style='font-size: 14px;'>{resolved_icon}</span> "
            else:
                icon_html = f"<span style='font-size: 14px;'>{html.escape(str(icon_key))}</span> "

        # Resolve tooltip
        title_attr = f" title='{html.escape(str(tooltip))}'" if tooltip else ""

        # Delta badge
        delta_html = ""
        if raw_delta is not None and str(raw_delta).strip():
            delta_str = str(raw_delta).strip()
            if delta_color == "inverse":
                bg_color = "#FEF2F2"
                text_color = "#DC2626"
                delta_class = "delta-inverse"
                arrow = "▼ " if not delta_str.startswith(("-", "▼")) else ""
            elif delta_color in ("neutral", "off", "gray"):
                bg_color = "#F1F5F9"
                text_color = "#64748B"
                delta_class = "delta-neutral"
                arrow = ""
            else:  # normal (positive / emerald)
                bg_color = "#ECFDF5"
                text_color = "#059669"
                delta_class = "delta-normal"
                arrow = "▲ " if not delta_str.startswith(("+", "▲", "%")) else ""

            delta_html = (
                f'<div class="haido-hud-delta {delta_class}" style="margin-top: 6px; display: inline-flex; '
                f'align-items: center; background: {bg_color}; color: {text_color}; font-size: 11px; '
                f'font-weight: 600; padding: 2px 8px; border-radius: 4px; align-self: flex-start;">'
                f'{arrow}{html.escape(delta_str)}</div>'
            )

        card_html = (
            f'<div class="haido-hud-card" style="flex: 1 1 180px; min-width: 160px; background: #F8FAFC; '
            f'border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; '
            f'box-shadow: 0 1px 3px rgba(0,0,0,0.04); display: flex; flex-direction: column; '
            f'justify-content: space-between;"{title_attr}>\n'
            f'    <div class="haido-hud-title" style="font-family: \'Be Vietnam Pro\', sans-serif; font-size: 11px; '
            f'font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.05em; '
            f'margin-bottom: 4px; display: flex; align-items: center; gap: 4px;">\n'
            f'        {icon_html}<span>{html.escape(raw_title)}</span>\n'
            f'    </div>\n'
            f'    <div class="haido-hud-value" style="font-family: \'JetBrains Mono\', monospace; font-size: 24px; '
            f'font-weight: 700; color: #0077B6; line-height: 1.15;">\n'
            f'        {html.escape(raw_val)}\n'
            f'    </div>\n'
            f'    {delta_html}\n'
            f'</div>'
        )
        card_items_html.append(card_html)

    full_hud_html = (
        '<div class="haido-hud-container" style="display: flex; gap: 12px; flex-wrap: wrap; width: 100%; margin-bottom: 16px;">\n'
        + "\n".join(card_items_html)
        + "\n</div>"
    )

    st.markdown(full_hud_html, unsafe_allow_html=True)
