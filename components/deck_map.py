"""Pydeck Map Wrapper with Carto Positron Light Basemap and HTML Tooltips.

Provides:
- Camera ViewState presets (Vietnam, Southeast Asia, Global, 3D Perspective).
- Clean Light Mode HTML tooltip templates for ports, arcs, and chokepoints.
- render_deck / make_deck wrapper returning fully configured pdk.Deck instances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import json
import urllib.parse
import pydeck as pdk

# Carto Vector Basemaps (token-free vector tiles, nolabels style to maintain Vietnamese sovereignty)
CARTO_POSITRON: str = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"
CARTO_POSITRON_NO_LABELS: str = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json"
CARTO_VOYAGER_NO_LABELS: str = "https://basemaps.cartocdn.com/gl/voyager-nolabels-gl-style/style.json"
CARTO_DARK_NO_LABELS: str = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json"

# ESRI World Imagery Photorealistic Satellite Style as standalone MapLibre style Data URI
# Provides full photorealistic natural terrain (deep blue sea, green vegetation, desert sands, no foreign labels)
_ESRI_SATELLITE_STYLE_DICT = {
    "version": 8,
    "name": "EsriWorldImagery",
    "sources": {
        "esri-satellite": {
            "type": "raster",
            "tiles": [
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            ],
            "tileSize": 256,
            "maxzoom": 19,
        }
    },
    "layers": [
        {
            "id": "esri-satellite-tiles",
            "type": "raster",
            "source": "esri-satellite",
            "minzoom": 0,
            "maxzoom": 22,
        }
    ],
}
ESRI_SATELLITE_STYLE_URL: str = (
    "data:application/json;charset=utf-8,"
    + urllib.parse.quote(json.dumps(_ESRI_SATELLITE_STYLE_DICT))
)

# Camera ViewState Presets
VIEW_VIETNAM = pdk.ViewState(latitude=16.0, longitude=108.0, zoom=5.0, pitch=0, bearing=0)
VIEW_SEA = pdk.ViewState(latitude=5.0, longitude=105.0, zoom=3.8, pitch=0, bearing=0)
VIEW_GLOBAL = pdk.ViewState(latitude=15.0, longitude=108.0, zoom=2.5, pitch=0, bearing=0)
VIEW_3D = pdk.ViewState(latitude=15.0, longitude=108.0, zoom=3.5, pitch=45, bearing=-15)

DEFAULT_VIEW = pdk.ViewState(latitude=15.0, longitude=108.0, zoom=3.5, pitch=0, bearing=0)

# Default Light Mode Tooltip
DEFAULT_TOOLTIP: Dict[str, Any] = {
    "html": (
        "<div style=\"font-family: 'Be Vietnam Pro', sans-serif; font-size: 12px; color: #0F172A; min-width: 160px; max-width: 320px;\">"
        "<div style=\"font-weight: 700; font-size: 13px; color: #0077B6; margin-bottom: 2px;\">{name}</div>"
        "<div style=\"color: #64748B; font-size: 11px; margin-bottom: 4px;\">{country} {iso3}</div>"
        "<div style=\"border-top: 1px solid #E2E8F0; padding-top: 4px; font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.45;\">"
        "<div>{pagerank}</div>"
        "<div style=\"color: #475569; margin-top: 2px;\">{degree}</div>"
        "</div>"
        "</div>"
    ),
    "style": {
        "backgroundColor": "#FFFFFF",
        "color": "#0F172A",
        "border": "1px solid #E2E8F0",
        "borderRadius": "8px",
        "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)",
        "padding": "8px 12px",
    },
}

DEFAULT_ARC_TOOLTIP: Dict[str, Any] = {
    "html": (
        "<div style=\"font-family: 'Be Vietnam Pro', sans-serif; font-size: 12px; color: #0F172A;\">"
        "<div style=\"font-weight: 700; color: #0077B6;\">{source} &rarr; {target}</div>"
        "<div style=\"font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-top: 4px;\">"
        "Trọng số: {weight}<br/>"
        "Khoảng cách: {distance_km} km<br/>"
        "Thời gian: {transit_days} ngày"
        "</div>"
        "</div>"
    ),
    "style": {
        "backgroundColor": "#FFFFFF",
        "color": "#0F172A",
        "border": "1px solid #E2E8F0",
        "borderRadius": "8px",
        "padding": "8px 12px",
    },
}


def render_deck(
    layers: Union[pdk.Layer, List[pdk.Layer], List[Any]],
    view_state: Optional[pdk.ViewState] = None,
    tooltip: Optional[Union[Dict[str, Any], bool, str]] = None,
    map_style: str = "satellite",
    map_provider: str = "carto",
    height: int = 650,
    include_sovereignty: bool = True,
) -> pdk.Deck:
    """Wraps Pydeck layers in a standardized Deck instance with Carto Positron basemap.

    Parameters
    ----------
    layers : list[pdk.Layer] | pdk.Layer
        Pydeck layers to render. Nested lists are automatically flattened.
    view_state : Optional[pdk.ViewState]
        Camera ViewState. Defaults to DEFAULT_VIEW (centered on Indo-Pacific / Vietnam).
    tooltip : Optional[dict | bool | str]
        Tooltip configuration. Defaults to DEFAULT_TOOLTIP if None.
    map_style : str
        Basemap style. Defaults to 'light' (or Carto Positron vector style).
    map_provider : str
        Tile provider ('carto', 'mapbox', etc.). Defaults to 'carto'.
    height : int
        Canvas height in pixels.

    Returns
    -------
    pdk.Deck
        Configured Pydeck Deck instance ready for st.pydeck_chart.
    """
    # Flatten nested layers if any (e.g. from chokepoints_layer or selected_route_layer)
    flat_layers: List[pdk.Layer] = []
    if isinstance(layers, pdk.Layer):
        flat_layers = [layers]
    elif isinstance(layers, (list, tuple)):
        for item in layers:
            if isinstance(item, (list, tuple)):
                flat_layers.extend([sub for sub in item if isinstance(sub, pdk.Layer)])
            elif isinstance(item, pdk.Layer):
                flat_layers.append(item)

    # Resolve view state
    active_view_state = view_state or DEFAULT_VIEW

    # Resolve tooltip
    if tooltip is None or tooltip is True:
        active_tooltip = DEFAULT_TOOLTIP
    elif isinstance(tooltip, str):
        active_tooltip = {"text": tooltip}
    elif isinstance(tooltip, dict):
        active_tooltip = tooltip
    else:
        active_tooltip = None

    # Resolve basemap style
    # Supports:
    # - "satellite": ESRI World Imagery photorealistic satellite tiles (full color natural imagery)
    # - "voyager": Carto Voyager vibrant colored basemap (vivid blue oceans & green terrain, no labels)
    # - "dark": Carto Dark Matter (stealth night mode, no labels)
    # - "light": Carto Positron (clean minimalist nautical light, no labels)
    s = str(map_style).lower() if map_style is not None else "satellite"
    if s in ("satellite", "vệ tinh", "imagery", "sat"):
        chosen_style = ESRI_SATELLITE_STYLE_URL
    elif s in ("voyager", "sống động", "vivid", "color", "màu"):
        chosen_style = CARTO_VOYAGER_NO_LABELS
    elif s in ("dark", "tối", "đêm"):
        chosen_style = CARTO_DARK_NO_LABELS
    elif s in ("light", "positron", "sáng"):
        chosen_style = CARTO_POSITRON_NO_LABELS
    else:
        chosen_style = map_style

    # Automatically overlay Vietnam maritime sovereignty layers (Hoàng Sa, Trường Sa & Biển Đông)
    if include_sovereignty:
        try:
            from components.layers import vietnam_sovereignty_layers
            sov_layers = vietnam_sovereignty_layers()
            flat_layers.extend(sov_layers)
        except Exception:
            pass

    return pdk.Deck(
        layers=flat_layers,
        initial_view_state=active_view_state,
        map_style=chosen_style,
        map_provider=map_provider,
        tooltip=active_tooltip,
        height=height,
    )


# Alias for spec compatibility
make_deck = render_deck
