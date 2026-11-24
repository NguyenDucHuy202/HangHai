"""Streamlit Component Bridge for Deck.gl TripsLayer Vessel Movement Animation.

Renders high-frequency client-side 60 FPS WebGL vessel trajectories using
MapLibre GL and Deck.gl TripsLayer with in-canvas Light Mode Glassmorphism controls:
- Play/Pause toggle
- Speed multipliers (1x, 2x, 5x, 10x)
- Interactive time scrubber slider
- UTC simulation clock (YYYY-MM-DD HH:mm UTC)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import streamlit as st
import streamlit.components.v1 as components

from core.data import load_trips


def render_trips_animation(
    trips_data: Optional[Union[str, Path, List[Dict[str, Any]]]] = None,
    height: int = 720,
) -> None:
    """Renders the standalone TripsLayer HTML component inside Streamlit.

    Parameters
    ----------
    trips_data : Optional[str | Path | list[dict]]
        Vessel trips trajectory data. If None, automatically loads from artifacts/trips.json.
    height : int
        Canvas height in pixels (default 720px).
    """
    # 1. Resolve trips payload
    if trips_data is None:
        trips_list = load_trips()
    elif isinstance(trips_data, (str, Path)):
        p = Path(trips_data)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                trips_list = json.load(f)
        else:
            trips_list = load_trips()
    elif isinstance(trips_data, list):
        trips_list = trips_data
    else:
        trips_list = load_trips()

    # 2. Read HTML template
    curr_dir = Path(__file__).resolve().parent
    template_path = curr_dir / "animated_map.html"

    if not template_path.exists():
        st.error(f"Không tìm thấy template TripsLayer tại {template_path}")
        return

    html_content = template_path.read_text(encoding="utf-8")

    # 3. Serialize and inject JSON payload into placeholder
    trips_json = json.dumps(trips_list, ensure_ascii=False)
    injected_html = html_content.replace("__TRIPS_DATA__", trips_json)

    # 4. Render embedded HTML iframe
    components.html(injected_html, height=height, scrolling=False)
