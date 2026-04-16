from typing import Dict, Any, Callable, Optional, Awaitable
import requests
import time
from datetime import datetime
import ee
import numpy as np
import math
import asyncio
import random
import threading

# -----------------------------
# Helpers: safe rounding & JSON sanitization=
# ------------------------------

def _is_num(x):
    return isinstance(x, (int, float, np.number))

def _round_safe(x, nd=4):
    """Round numbers safely; pass None through; convert NaN/inf to None."""
    if x is None:
        return None
    try:
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return round(xf, nd)
    except Exception:
        return None

def _clean_numbers(obj):
    """
    Recursively replace NaN/inf with None so JSON is valid.
    Also ensures nested dicts/lists are cleaned.
    """
    if isinstance(obj, dict):
        return {k: _clean_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_numbers(v) for v in obj]
    if _is_num(obj):
        try:
            xf = float(obj)
            return xf if math.isfinite(xf) else None
        except Exception:
            return None
    return obj

def strip_z(coords):
    """Strip Z coordinate from coordinates, handling different coordinate formats"""
    if not isinstance(coords, list) or not coords:
        return coords

    if isinstance(coords[0], list):
        if isinstance(coords[0][0], list):
            return [strip_z(c) for c in coords]
        else:
            return [[pt[0], pt[1]] for pt in coords if len(pt) >= 2]
    else:
        if len(coords) >= 2:
            return [coords[0], coords[1]]
        else:
            return coords


# ------------------------------
# PlotSyncService for Django API
# ------------------------------

class PlotSyncService:
    """
    Service to fetch plot data from Django /plots/ API
    """

    def __init__(self, django_api_url: str = "https://cropeye-backendd.up.railway.app"):
        self.django_api_url = django_api_url
        self.plots_cache = {}
        self.last_sync = None
        # Keep cache short so apps can refresh every 3-5 seconds safely.
        self.cache_duration = 4
        self._cache_lock = threading.Lock()
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=200, pool_maxsize=200, max_retries=2)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self.last_error: Optional[str] = None
        self.last_successful_sync: Optional[datetime] = None
        self.refresh_attempts = 0
        self.refresh_successes = 0

    def fetch_plots_from_api(self) -> Dict[str, Any]:
        """Fetch all plots from Django API with retries (Django on Render may cold-start)"""
        timeout = 30  # Render cold start can take 30+ seconds
        max_retries = 3
        retry_delay = 8

        for attempt in range(max_retries):
            try:
                response = self._session.get(
                    f"{self.django_api_url}/api/plots/public/",
                    headers={'Content-Type': 'application/json'},
                    timeout=timeout
                )

                if response.status_code == 200:
                    plots_data = response.json()
                    return self._process_plots_response(plots_data)
                else:
                    print(f"Warning: Django API returned status {response.status_code}. Using empty plot list.")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                    else:
                        return {}

            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not connect to Django API: {str(e)}. Using empty plot list.")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    return {}

        return {}

    def _process_plots_response(self, plots_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Process the Django API response and convert to plot dictionary format"""
        plot_dict = {}

        plots_list = plots_data.get('results') if isinstance(plots_data, dict) else plots_data

        for plot in plots_list or []:
            plot_id = plot.get('id')
            gat_number = plot.get('gat_number', '')
            plot_number = plot.get('plot_number', '')

            address = plot.get('address', {})
            village = address.get('village', '')
            field_officer = plot.get('field_officer', {})
            field_officer_id = field_officer.get('id') if isinstance(field_officer, dict) else None

            farms = plot.get('farms', [])
            plantation_date = None
            plantation_type = None
            crop_type_name = None

            if farms:
                crop_type_name = farms[0].get('crop_type_name')

            if crop_type_name is None and isinstance(plot.get('crop_type'), dict):
                crop_type_name = plot.get('crop_type', {}).get('name')
            foundation_pruning_date = None
            fruit_pruning_date = None

            if farms:
                plantation_date = farms[0].get('plantation_date')
                plantation_type = farms[0].get('plantation_type')
                foundation_pruning_date = farms[0].get('foundation_pruning_date')
                fruit_pruning_date = farms[0].get('fruit_pruning_date')

            plot_name = f"{gat_number}_{plot_number}" if gat_number and plot_number else gat_number or f"plot_{plot_id}"

            geometry, geom_type, coords = None, None, None
            boundary = plot.get('boundary')

            if isinstance(boundary, dict) and 'coordinates' in boundary and boundary['coordinates']:
                coords = strip_z(boundary['coordinates'])
                geom_type = 'Polygon'
                try:
                    geometry = ee.Geometry.Polygon(coords)
                except Exception as e:
                    print(f"Error creating Polygon geometry for plot {plot_id}: {e}")
                    continue
            elif plot.get('location') and plot['location'].get('coordinates'):
                location = plot['location']['coordinates']
                lat, lng = location[1], location[0]
                offset = 0.001
                polygon_coords = [[
                    [lng - offset, lat - offset], [lng + offset, lat - offset],
                    [lng + offset, lat + offset], [lng - offset, lat + offset],
                    [lng - offset, lat - offset]
                ]]
                coords = strip_z(polygon_coords)
                geom_type = 'Polygon'
                geometry = ee.Geometry.Polygon(coords)
            else:
                print(f"Warning: Plot {plot_id} has no valid geometry, skipping.")
                continue

            plot_dict[plot_name] = {
                "geometry": geometry,
                "geom_type": geom_type,
                "original_coords": coords,
                "properties": {
                    'gat_number': gat_number,
                    'plot_number': plot_number,
                    'village': village,
                    'taluka': address.get('taluka', ''),
                    'district': address.get('district', ''),
                    'state': address.get('state', ''),
                    'country': address.get('country', ''),
                    'pin_code': address.get('pin_code', ''),
                    'django_id': plot_id,
                    'plantation_date': plantation_date,
                    'plantation_type': plantation_type,
                    'foundation_pruning_date': foundation_pruning_date,
                    'fruit_pruning_date': fruit_pruning_date,
                    'crop_type_name': crop_type_name,
                    'field_officer_id': field_officer_id,
                }
            }

        return plot_dict

    def get_plots_dict(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """Get plots dictionary, with caching"""
        current_time = datetime.now()
        with self._cache_lock:
            if (
                not force_refresh and
                self.last_sync and
                (current_time - self.last_sync).total_seconds() < self.cache_duration and
                self.plots_cache
            ):
                return self.plots_cache

            self.refresh_attempts += 1
            plots_data = self.fetch_plots_from_api()
            if plots_data:
                self.plots_cache = plots_data
                self.last_sync = current_time
                self.last_successful_sync = current_time
                self.last_error = None
                self.refresh_successes += 1
            else:
                self.last_error = "Empty or failed sync from Django API"
            return self.plots_cache

    def health_ping(self) -> bool:
        """Cheap keep-alive ping to reduce Django cold-start chance."""
        try:
            resp = self._session.get(
                f"{self.django_api_url}/api/plots/public/?page_size=1",
                timeout=10
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_refresh_stats(self) -> Dict[str, Any]:
        return {
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "last_successful_sync": self.last_successful_sync.isoformat() if self.last_successful_sync else None,
            "last_error": self.last_error,
            "refresh_attempts": self.refresh_attempts,
            "refresh_successes": self.refresh_successes,
            "cache_duration_seconds": self.cache_duration,
            "cached_plots": len(self.plots_cache),
        }


class PlotKeepAliveManager:
    """Background refresh loop for high-availability plot sync."""

    def __init__(
        self,
        plot_service: PlotSyncService,
        apply_callback: Callable[[Dict[str, Dict]], Optional[Awaitable[None]]],
        min_interval_seconds: float = 3.0,
        max_interval_seconds: float = 5.0,
    ):
        self.plot_service = plot_service
        self.apply_callback = apply_callback
        self.min_interval_seconds = min_interval_seconds
        self.max_interval_seconds = max_interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_cycle: Optional[datetime] = None

    async def _apply(self, plots: Dict[str, Dict]) -> None:
        maybe = self.apply_callback(plots)
        if asyncio.iscoroutine(maybe):
            await maybe

    async def refresh_now(self, force: bool = True) -> Dict[str, Dict]:
        plots = await asyncio.to_thread(self.plot_service.get_plots_dict, force)
        if plots:
            await self._apply(plots)
        self._last_cycle = datetime.now()
        return plots

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.to_thread(self.plot_service.health_ping)
                await self.refresh_now(force=True)
            except Exception as e:
                self.plot_service.last_error = str(e)
            await asyncio.sleep(random.uniform(self.min_interval_seconds, self.max_interval_seconds))

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._task and not self._task.done()),
            "last_cycle": self._last_cycle.isoformat() if self._last_cycle else None,
            "interval_seconds": [self.min_interval_seconds, self.max_interval_seconds],
            "plot_sync": self.plot_service.get_refresh_stats(),
        }
