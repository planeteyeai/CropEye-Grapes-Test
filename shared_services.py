from typing import Dict, Any
import requests
import time
from datetime import datetime
import ee
import numpy as np
import math

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
        self.cache_duration = 300  # 5 minute cache - reduce upstream load and egress
        self.min_force_refresh_interval = 300  # guard forced refreshes from running too frequently
        self.last_force_refresh_monotonic = 0.0
        self.public_plots_etag = None

    def fetch_plots_from_api(self) -> Dict[str, Any]:
        """Fetch public plots from Django API with retries, pagination, and ETag support."""
        timeout = 30  # Render cold start can take 30+ seconds
        max_retries = 3
        retry_delay = 8

        for attempt in range(max_retries):
            try:
                base_url = f"{self.django_api_url}/api/plots/public/"
                headers = {'Content-Type': 'application/json'}
                if self.public_plots_etag:
                    headers["If-None-Match"] = self.public_plots_etag

                response = requests.get(
                    base_url,
                    params={"page_size": 100},
                    headers=headers,
                    timeout=timeout,
                )

                if response.status_code == 304:
                    return self.plots_cache

                if response.status_code != 200:
                    print(f"Warning: Django API returned status {response.status_code}.")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(retry_delay)
                        continue
                    return self.plots_cache

                self.public_plots_etag = response.headers.get("ETag") or self.public_plots_etag
                first_page = response.json()

                if not isinstance(first_page, dict) or "results" not in first_page:
                    return self._process_plots_response(first_page)

                all_results = list(first_page.get("results") or [])
                next_url = first_page.get("next")
                while next_url:
                    next_response = requests.get(
                        next_url,
                        headers={'Content-Type': 'application/json'},
                        timeout=timeout,
                    )
                    if next_response.status_code != 200:
                        break
                    next_data = next_response.json()
                    all_results.extend(next_data.get("results") or [])
                    next_url = next_data.get("next")

                return self._process_plots_response({"results": all_results})

            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not connect to Django API: {str(e)}.")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(retry_delay)
                else:
                    return self.plots_cache

        return self.plots_cache

    def _process_plots_response(self, plots_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Process the Django API response and convert to plot dictionary format"""
        plot_dict = {}

        for plot in plots_data.get('results', []):
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
            crop_type_name = plot.get('crop_type_name')
            if crop_type_name is None and isinstance(plot.get('crop_type'), dict):
                crop_type_name = plot.get('crop_type', {}).get('name')
            if farms:
                plantation_date = farms[0].get('plantation_date')
                plantation_type = farms[0].get('plantation_type')
                foundation_pruning_date = farms[0].get('foundation_pruning_date')
                fruit_pruning_date = farms[0].get('fruit_pruning_date')
                if crop_type_name is None:
                    crop_type_name = farms[0].get('crop_type_name')
                if crop_type_name is None and isinstance(farms[0].get('crop_type'), dict):
                    crop_type_name = farms[0].get('crop_type', {}).get('name')

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
        now_monotonic = time.monotonic()

        if (
            not force_refresh
            and self.last_sync
            and (current_time - self.last_sync).seconds < self.cache_duration
            and self.plots_cache
        ):
            return self.plots_cache

        if (
            force_refresh
            and self.plots_cache
            and (now_monotonic - self.last_force_refresh_monotonic) < self.min_force_refresh_interval
        ):
            return self.plots_cache

        if force_refresh:
            self.last_force_refresh_monotonic = now_monotonic

        plots_data = self.fetch_plots_from_api()
        if isinstance(plots_data, dict):
            self.plots_cache = plots_data
            self.last_sync = current_time
            return plots_data
        return self.plots_cache
    



    
