"""Maritime geometry engine and vessel trajectory synthesizer.

Wraps searoute with LRU caching, great-circle slerp fallback, batch symmetry optimization,
and timestamp interpolation for Deck.gl TripsLayer animation.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two [lon, lat] points in km."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def great_circle_fallback(
    origin_lonlat: List[float] | Tuple[float, float],
    dest_lonlat: List[float] | Tuple[float, float],
    num_points: int = 15,
) -> List[List[float]]:
    """Generates intermediate waypoints along a great-circle arc using spherical slerp."""
    lon1, lat1 = origin_lonlat[0], origin_lonlat[1]
    lon2, lat2 = dest_lonlat[0], dest_lonlat[1]

    # Convert to 3D Cartesian coordinates on unit sphere
    phi1, lambda1 = math.radians(lat1), math.radians(lon1)
    phi2, lambda2 = math.radians(lat2), math.radians(lon2)

    v1 = np.array([math.cos(phi1) * math.cos(lambda1),
                   math.cos(phi1) * math.sin(lambda1),
                   math.sin(phi1)])
    v2 = np.array([math.cos(phi2) * math.cos(lambda2),
                   math.cos(phi2) * math.sin(lambda2),
                   math.sin(phi2)])

    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    omega = math.acos(dot)

    if omega < 1e-6:
        # Source and destination are virtually identical
        return [[round(lon1, 4), round(lat1, 4)], [round(lon2, 4), round(lat2, 4)]]

    sin_omega = math.sin(omega)
    points = []
    for step in range(num_points):
        t = step / float(num_points - 1)
        vt = (math.sin((1.0 - t) * omega) / sin_omega) * v1 + (math.sin(t * omega) / sin_omega) * v2
        vt = vt / np.linalg.norm(vt)

        lat_t = math.degrees(math.asin(vt[2]))
        lon_t = math.degrees(math.atan2(vt[1], vt[0]))
        points.append([round(lon_t, 4), round(lat_t, 4)])

    return points


# In-memory session cache for searoute calls
_SEAROUTE_MEMO_CACHE: Dict[Tuple[float, float, float, float], Dict[str, Any]] = {}


def safe_searoute(
    origin_lonlat: List[float] | Tuple[float, float],
    dest_lonlat: List[float] | Tuple[float, float],
) -> Dict[str, Any]:
    """Computes realistic maritime routing avoiding landmasses with great-circle fallback.
    
    Returns a dictionary:
    - 'path': List of [lon, lat] coordinates
    - 'length_km': Route nautical distance in km
    - 'is_searoute': True if searoute succeeded, False if great-circle fallback
    """
    lon1, lat1 = round(float(origin_lonlat[0]), 4), round(float(origin_lonlat[1]), 4)
    lon2, lat2 = round(float(dest_lonlat[0]), 4), round(float(dest_lonlat[1]), 4)
    key = (lon1, lat1, lon2, lat2)

    if key in _SEAROUTE_MEMO_CACHE:
        return _SEAROUTE_MEMO_CACHE[key]

    # Check for identical coordinates
    if lon1 == lon2 and lat1 == lat2:
        res = {
            "path": [[lon1, lat1], [lon2, lat2]],
            "length_km": 1.0,
            "is_searoute": False,
        }
        _SEAROUTE_MEMO_CACHE[key] = res
        return res

    try:
        import searoute as sr
        route_geojson = sr.searoute([lon1, lat1], [lon2, lat2])
        raw_coords = route_geojson["geometry"]["coordinates"]
        raw_length = float(route_geojson["properties"].get("length", 0.0))

        if len(raw_coords) >= 2:
            clean_coords = [[round(pt[0], 4), round(pt[1], 4)] for pt in raw_coords]
            dist_km = raw_length if raw_length > 0 else haversine_distance_km(lon1, lat1, lon2, lat2)
            res = {
                "path": clean_coords,
                "length_km": round(dist_km, 2),
                "is_searoute": True,
            }
            _SEAROUTE_MEMO_CACHE[key] = res
            return res
    except Exception:
        pass

    # Fallback to great-circle arc
    gc_coords = great_circle_fallback([lon1, lat1], [lon2, lat2], num_points=15)
    dist_km = haversine_distance_km(lon1, lat1, lon2, lat2)
    res = {
        "path": gc_coords,
        "length_km": round(max(dist_km, 1.0), 2),
        "is_searoute": False,
    }
    _SEAROUTE_MEMO_CACHE[key] = res
    return res


def interpolate_route_timestamps(
    coordinates: List[List[float]],
    start_time: float,
    duration_sec: float,
    min_dt: float = 0.05,
) -> Tuple[List[List[float]], List[float]]:
    """Interpolates timestamps along coordinates proportionally to segment distances.
    
    Guarantees strictly monotonic non-decreasing timestamps for Deck.gl TripsLayer.
    """
    if len(coordinates) < 2:
        return coordinates, [start_time] * len(coordinates)

    # Compute cumulative distances
    dists = [0.0]
    for i in range(len(coordinates) - 1):
        p1, p2 = coordinates[i], coordinates[i + 1]
        d = haversine_distance_km(p1[0], p1[1], p2[0], p2[1])
        dists.append(dists[-1] + max(d, 0.001))

    total_dist = dists[-1]
    timestamps = []
    for i, d in enumerate(dists):
        fraction = d / total_dist if total_dist > 0 else (i / (len(dists) - 1))
        t = start_time + fraction * duration_sec
        # Enforce monotonic increase
        if timestamps and t < timestamps[-1] + min_dt:
            t = timestamps[-1] + min_dt
        timestamps.append(round(t, 2))

    return coordinates, timestamps
