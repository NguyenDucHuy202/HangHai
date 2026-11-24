"""Maritime Transport SNA UI Components Package for Hải đồ số.

Exports:
- HUD metrics banner: render_hud
- Pydeck layer factories: ports_layer, arcs_layer, searoute_layer, chokepoints_layer, seamark_layer, etc.
- Pydeck map wrapper: render_deck, make_deck
- TripsLayer WebGL animation: render_trips_animation
"""

from components.hud import render_hud
from components.layers import (
    ports_layer,
    ports_glow_layer,
    vn_stars_layer,
    arcs_layer,
    searoute_layer,
    chokepoints_layer,
    seamark_layer,
    satellite_layer,
    suggested_routes_layer,
    disrupted_layer,
    vietnam_sovereignty_layers,
)
from components.deck_map import render_deck, make_deck
from components.animated_map import render_trips_animation

__all__ = [
    "render_hud",
    "ports_layer",
    "ports_glow_layer",
    "vn_stars_layer",
    "arcs_layer",
    "searoute_layer",
    "chokepoints_layer",
    "seamark_layer",
    "satellite_layer",
    "suggested_routes_layer",
    "disrupted_layer",
    "vietnam_sovereignty_layers",
    "render_deck",
    "make_deck",
    "render_trips_animation",
]
