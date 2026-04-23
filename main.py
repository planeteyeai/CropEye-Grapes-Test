import asyncio
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import ee
import pandas as pd
from datetime import datetime, timedelta
import uvicorn
from contextlib import asynccontextmanager
import json
from shared_services import PlotSyncService, _round_safe, _clean_numbers
import numpy as np
from events import calculate_brix_sugar_stats1,get_brix_recovery_sugar_yield_images
import hashlib
from datetime import datetime, timedelta, date
# ------------------------------
# Initialize Earth Engine
# ------------------------------

def _init_earth_engine():
    raw = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise ValueError("EE_SERVICE_ACCOUNT_JSON env var is required. Set it in Railway Variables.")
    sa = json.dumps(raw) if not isinstance(raw, str) else raw
    ee.Initialize(ee.ServiceAccountCredentials(None, key_data=sa))

_init_earth_engine()

# ------------------------------
# FastAPI Setup
# ------------------------------


def _apply_plot_update(new_plots: Dict[str, Dict]) -> None:
    global plot_dict
    plot_dict = new_plots
    print(f"main.py background refresh: {len(plot_dict)} plots")

def _resolve_plot_or_refresh(plot_identifier: str):
    global plot_dict
    if plot_identifier in plot_dict:
        return plot_identifier, plot_dict[plot_identifier]
    for name, pdata in plot_dict.items():
        django_id = pdata.get("properties", {}).get("django_id") or pdata.get("django_id")
        if str(django_id) == str(plot_identifier):
            return name, pdata

    fresh = plot_sync_service.get_plots_dict(force_refresh=True)
    if fresh:
        _apply_plot_update(fresh)
    if plot_identifier in plot_dict:
        return plot_identifier, plot_dict[plot_identifier]
    for name, pdata in plot_dict.items():
        django_id = pdata.get("properties", {}).get("django_id") or pdata.get("django_id")
        if str(django_id) == str(plot_identifier):
            return name, pdata
    raise HTTPException(status_code=404, detail="Plot not found (tried instant refresh)")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global plot_dict
        print("🔄 main.py: Initializing application and fetching plots from Django API...")

        # Fetch once at startup (force refresh)
        plot_dict = await asyncio.to_thread(
            plot_sync_service.get_plots_dict, True
        )

        print(f"✅ main.py startup: Loaded {len(plot_dict)} plots from Django")
        print("🚀 main.py: Application initialized successfully")

    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        raise

    yield

    # Shutdown (no keepalive to stop anymore)
    print("🛑 Shutting down FastAPI application")

app = FastAPI(
    title="Soil Parameter Analysis API with NPK and SAR-based Fe Analysis",
    description="Analyze various soil parameters per plot from a GeoJSON using Earth Engine with NPK and Sentinel-1 SAR Iron calculations",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

# ------------------------------
# Initialize plot sync service
plot_sync_service = PlotSyncService()

# ------------------------------
# Load GeoJSON and Prepare Plot Dictionary
# ------------------------------

PLANTATION_DATE = '2025-01-01'  # Default plantation date

# Load plots from Django
plot_dict = plot_sync_service.get_plots_dict()
print(f"ðŸš€ main.py startup: Loaded {len(plot_dict)} plots from Django")

# ------------------------------
# Load Soil Layers (without SOC)
# ------------------------------

def load_soil_layers():
    return {
        'bulk_density': ee.Image("projects/soilgrids-isric/bdod_mean"),
        'soil_organic_carbon': ee.Image("projects/soilgrids-isric/soc_mean"),
        'total_nitrogen': ee.Image("projects/soilgrids-isric/nitrogen_mean"),
        'cation_exchange_capacity': ee.Image("projects/soilgrids-isric/cec_mean"), 
        'organic_carbon_stock': ee.Image("projects/soilgrids-isric/ocs_mean"),
        'phh2o': ee.Image("projects/soilgrids-isric/phh2o_mean")
    }

soil_layers = load_soil_layers()

 
# ------------------------------
# Fe Index Calculation Function using Sentinel-1
# ------------------------------
def calculate_fe_index(geometry, days_back=30):
    """Calculate Fe Index using Sentinel-1 SAR data for current date (with fallback period)"""
    try:
        # Get current date and calculate date range
        current_date = datetime.now()
        end_date = current_date.strftime('%Y-%m-%d')
        start_date = (current_date - pd.Timedelta(days=days_back)).strftime('%Y-%m-%d')
       
        print(f"Searching for Sentinel-1 SAR data from {start_date} to {end_date}")
       
        # Load Sentinel-1 SAR Ground Range Detected (GRD) collection
        collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                     .filter(ee.Filter.eq('instrumentMode', 'IW'))  # Interferometric Wide swath
                     .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
                     .sort('system:time_start', False))  # Sort by date, newest first
       
        # Get the most recent image
        image = collection.first()
       
        # Check if we found any images
        collection_size = collection.size().getInfo()
        if collection_size == 0:
            print(f"No suitable Sentinel-1 images found for the period {start_date} to {end_date}")
            return {
                'error': f'No Sentinel-1 SAR data available from {start_date} to {end_date}',
                'fe_index': None,
                'fe_ppm_estimated': None,
                'image_date': None,
                'polarizations': None
            }
           
        # Get image metadata
        image_info = image.getInfo()
        if image_info is None:
            return None
           
        # Extract image date and polarizations
        image_date = datetime.fromtimestamp(image_info['properties']['system:time_start'] / 1000).strftime('%Y-%m-%d')
        polarizations = image_info['properties'].get('transmitterReceiverPolarisation', ['Unknown'])
        orbit_pass = image_info['properties'].get('orbitProperties_pass', 'Unknown')
       
        print(f"Using Sentinel-1 SAR image from {image_date} with polarizations {polarizations}, orbit: {orbit_pass}")
       
        # Calculate Fe Index using SAR backscatter
        # Method 1: VV/VH ratio (common for soil moisture and mineral detection)
        vv = image.select('VV')
        vh = image.select('VH')
       
        # Convert from dB to linear scale for ratio calculation
        vv_linear = ee.Image(10).pow(vv.divide(10))
        vh_linear = ee.Image(10).pow(vh.divide(10))
       
        # Calculate VV/VH ratio as Fe proxy
        fe_index_ratio = vv_linear.divide(vh_linear).rename('Fe_Index_Ratio')
       
        # Method 2: Cross-polarization difference (VV - VH) for mineral detection
        fe_index_diff = vv.subtract(vh).rename('Fe_Index_Diff')
       
        # Method 3: Normalized difference (VV - VH) / (VV + VH)
        fe_index_ndvi_style = vv.subtract(vh).divide(vv.add(vh)).rename('Fe_Index_NDVI_Style')
       
        # Sample all Fe indices for the geometry
        reduction = ee.Image.cat([fe_index_ratio, fe_index_diff, fe_index_ndvi_style, vv, vh]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            maxPixels=1e13
        ).getInfo()
       
        fe_ratio = reduction.get('Fe_Index_Ratio')
        fe_diff = reduction.get('Fe_Index_Diff')
        fe_ndvi_style = reduction.get('Fe_Index_NDVI_Style')
        vv_value = reduction.get('VV')
        vh_value = reduction.get('VH')
       
        if fe_ratio is not None and fe_diff is not None:
            # Use the ratio method as primary Fe index
            fe_index_value = fe_ratio
           
            # Apply custom regression formula for SAR-based Fe estimation
            # This is a placeholder formula - you may need to calibrate with ground truth data
            # For now, using a modified version adapted for SAR values
            fe_ppm_estimated = (fe_index_value * 0.8) + 2.5  # Adjusted coefficients for SAR data
           
            return {
                'fe_index_primary': round(fe_index_value, 4),  # VV/VH ratio
                'fe_index_difference': round(fe_diff, 4),      # VV - VH
                'fe_index_normalized': round(fe_ndvi_style, 4), # (VV-VH)/(VV+VH)
                'fe_ppm_estimated': round(fe_ppm_estimated, 4),
                'vv_backscatter_db': round(vv_value, 4),
                'vh_backscatter_db': round(vh_value, 4),
                'image_date': image_date,
                'polarizations': polarizations,
                'orbit_pass': orbit_pass,
                'search_period': f"{start_date} to {end_date}",
                'images_found': collection_size,
                'sensor_type': 'Sentinel-1 SAR'
            }
        else:
            return {
                'error': 'Failed to extract Fe Index values from SAR image',
                'fe_index_primary': None,
                'fe_ppm_estimated': None,
                'image_date': image_date,
                'polarizations': polarizations
            }
           
    except Exception as e:
        print(f"Error calculating Fe Index with Sentinel-1: {e}")
        return {
            'error': f'Fe Index calculation failed: {str(e)}',
            'fe_index_primary': None,
            'fe_ppm_estimated': None,
            'sensor_type': 'Sentinel-1 SAR'
        }
 
# ------------------------------
# Soil Analysis Logic (Modified)
# ------------------------------
def calculate_mean_statistics(geometry, scale=250, fe_days_back=30):
    parameter_bands = {
        'bulk_density': 'bdod_0-5cm_mean',
        'soil_organic_carbon': 'soc_0-5cm_mean',
        'total_nitrogen': 'nitrogen_0-5cm_mean',
        'cation_exchange_capacity': 'cec_0-5cm_mean', 
        'organic_carbon_stock': 'ocs_0-30cm_mean',
        'phh2o': 'phh2o_0-5cm_mean'
    }
   
    stats = {}
   
    # Calculate traditional soil parameters
    for param, image in soil_layers.items():
        try:
            band = parameter_bands[param]
            val = image.select(band).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=scale,
                maxPixels=1e13
            ).getInfo().get(band)
           
            if val is not None:
                if param == 'bulk_density':
                    stats[param] = round(val / 100.0, 4)
                elif param == 'soil_organic_carbon':
                    stats[param] = round(val / 100.0, 4)
                elif param == 'total_nitrogen':
                    stats[param] = round(val / 10.0, 4)
                elif param == 'cation_exchange_capacity':
                    stats[param] = round(val / 10.0, 4) 
                elif param == 'organic_carbon_stock':
                    stats[param] = round(val/2.47, 4)
                elif param == 'phh2o':
                    stats[param] = round(val / 10.0, 4)
                else:
                    stats[param] = round(val, 4)
            else:
                stats[param] = None
        except Exception as e:
            print(f"Error processing {param}: {e}")
            stats[param] = None
   
    # Calculate Fe Index using Sentinel-1 SAR (replacing OCD)
    fe_data = calculate_fe_index(geometry, fe_days_back)
    if fe_data and not fe_data.get('error'):
        stats['fe_index_primary'] = fe_data['fe_index_primary']
        stats['fe_index_difference'] = fe_data['fe_index_difference']
        stats['fe_index_normalized'] = fe_data['fe_index_normalized']
        stats['fe_ppm_estimated'] = fe_data['fe_ppm_estimated']
        stats['vv_backscatter_db'] = fe_data.get('vv_backscatter_db')
        stats['vh_backscatter_db'] = fe_data.get('vh_backscatter_db')
        stats['fe_image_date'] = fe_data.get('image_date')
        stats['fe_polarizations'] = fe_data.get('polarizations')
        stats['fe_orbit_pass'] = fe_data.get('orbit_pass')
        stats['fe_search_period'] = fe_data.get('search_period')
        stats['fe_images_found'] = fe_data.get('images_found')
        stats['fe_sensor_type'] = fe_data.get('sensor_type')
    else:
        stats['fe_index_primary'] = None
        stats['fe_index_difference'] = None
        stats['fe_index_normalized'] = None
        stats['fe_ppm_estimated'] = None
        stats['fe_error'] = fe_data.get('error') if fe_data else 'Unknown error'
        if fe_data:
            stats['fe_image_date'] = fe_data.get('image_date')
            stats['fe_polarizations'] = fe_data.get('polarizations')
            stats['fe_sensor_type'] = fe_data.get('sensor_type')
   
    # Calculate Potassium (K) and Phosphorus (P) using total_nitrogen (N)
    N = stats.get('total_nitrogen')
    if N is not None:
        K = (N * 1.181) + (N / 5)
        P = K / 9
        stats['potassium'] = round(K, 4)
        stats['phosphorus'] = round(P, 4)
    else:
        stats['potassium'] = None
        stats['phosphorus'] = None
   
    return stats
 
# ------------------------------
# NPK Analysis Functions (Unchanged)
# ------------------------------
def add_gndvi(image):
    """Add GNDVI band to Sentinel-2 image"""
    gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')
    return image.addBands(gndvi).set('system:time_start', image.get('system:time_start'))
 
def calculate_npk_for_plot(geometry, plantation_date=PLANTATION_DATE):
    """Calculate NPK values for a specific sugarcane plot using GNDVI analysis (Calibrated for Maharashtra)"""
    try:
        s2 = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geometry)
            .filterDate(plantation_date, datetime.now().strftime('%Y-%m-%d'))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 75))
        )
 
        with_gndvi = s2.map(add_gndvi)
 
        def reduce_mean(image):
            mean = image.select('GNDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13
            )
            return ee.Feature(None, {
                'GNDVI': mean.get('GNDVI'),
                'system:time_start': image.get('system:time_start')
            })
 
        stats = with_gndvi.map(reduce_mean)
        stats_fc = ee.FeatureCollection(stats).sort('system:time_start')
        stats_list = stats_fc.toList(stats_fc.size())
 
        # --- Use iterate() for cumulative sugarcane-specific NPK ---
        def accumulate_npk(index, prev_result):
            index = ee.Number(index)
            prev_result = ee.List(prev_result)
           
            curr = ee.Feature(stats_list.get(index))
            prev = ee.Feature(prev_result.get(-1))
 
            # --- GNDVI difference ---
            curr_gndvi = ee.Number(curr.get('GNDVI'))
            prev_gndvi = ee.Number(prev.get('GNDVI'))
            delta = curr_gndvi.subtract(prev_gndvi)
 
            # --- Sugarcane canopy sensitivity correction ---
            delta_fixed = delta.multiply(2.0).multiply(100)
 
            # --- NPK response model ---
            nitrogen_inc = delta_fixed.divide(0.35)
            phosphorus_inc = nitrogen_inc.divide(3.0)
            potassium_inc = nitrogen_inc.multiply(2.0/3.0)
 
            # --- Get previous cumulative values ---
            prev_n = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(prev.get('Nitrogen'), None), 0, prev.get('Nitrogen')))
            prev_p = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(prev.get('Phosphorus'), None), 0, prev.get('Phosphorus')))
            prev_k = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(prev.get('Potassium'), None), 0, prev.get('Potassium')))
 
            # --- Cumulative addition ---
            nitrogen = prev_n.add(nitrogen_inc)
            phosphorus = prev_p.add(phosphorus_inc)
            potassium = prev_k.add(potassium_inc)
 
            updated = curr.set({
                'GNDVI_Change': delta,
                'GNDVI_Change_Fixed': delta_fixed,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium
            })
            return prev_result.add(updated)
        # --- End cumulative logic ---
 
        # Guard for empty data
        size = stats_fc.size().getInfo() or 0
        if size == 0:
            return {'error': 'No Sentinel-2 data available for the specified period'}
 
        # Initialize with first record
        first = ee.Feature(stats_list.get(0)).set({
            'GNDVI_Change': None,
            'GNDVI_Change_Fixed': None,
            'Nitrogen': None,
            'Phosphorus': None,
            'Potassium': None
        })
 
        # Iterate cumulatively
        delta_list = ee.List.sequence(1, ee.Number(size).subtract(1)).iterate(accumulate_npk, ee.List([first]))
        final_stats = ee.FeatureCollection(ee.List(delta_list))
 
        def format_feature(f):
            return ee.Feature(None, {
                'Date': ee.Date(f.get('system:time_start')).format('YYYY-MM-dd'),
                'GNDVI': f.get('GNDVI'),
                'GNDVI_Change': f.get('GNDVI_Change'),
                'GNDVI_Change_Fixed': f.get('GNDVI_Change_Fixed'),
                'Nitrogen': f.get('Nitrogen'),
                'Phosphorus': f.get('Phosphorus'),
                'Potassium': f.get('Potassium')
            })
 
        table_fc = final_stats.map(format_feature)
        table = table_fc.getInfo()
        rows = [f['properties'] for f in table.get('features', [])]
        df = pd.DataFrame(rows)
 
        if df.empty:
            return {'error': 'No Sentinel-2 data available for the specified period'}
 
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.replace([np.inf, -np.inf], np.nan).round(2)
 
        # Area
        area_m2 = geometry.area().getInfo()
        area_acres = area_m2 / 4046.86
 
        # --- Sugarcane-specific recommended dose (per acre) ---
        dose_per_acre = {'N': 120, 'P': 40, 'K': 80}
        recommended_dose = dose_per_acre.copy()
 
        df_sorted = df.sort_values('Date')
        latest = df_sorted.iloc[-1]
 
        if (latest[['Nitrogen', 'Phosphorus', 'Potassium']] < 0).any():
            for i in range(len(df_sorted) - 2, -1, -1):
                candidate = df_sorted.iloc[i]
                if (candidate[['Nitrogen', 'Phosphorus', 'Potassium']] >= 0).all():
                    latest = candidate
                    break
 
        estN = (latest['Nitrogen'] if pd.notna(latest['Nitrogen']) else 0) / area_acres
        estP = (latest['Phosphorus'] if pd.notna(latest['Phosphorus']) else 0) / area_acres
        estK = (latest['Potassium'] if pd.notna(latest['Potassium']) else 0) / area_acres
        estimated = {'N': float(estN), 'P': float(estP), 'K': float(estK)}
 
        difference = {
            'N': recommended_dose['N'] - estimated['N'],
            'P': recommended_dose['P'] - estimated['P'],
            'K': recommended_dose['K'] - estimated['K']
        }
 
        final_displayed_dose = {
            'N': recommended_dose['N'],
            'P': recommended_dose['P'],
            'K': recommended_dose['K']
        }
 
        def fmt(d):
            out = {}
            for k, v in d.items():
                out[k] = _round_safe(v, 2)
            return out
 
        records = [_clean_numbers(r) for r in df.to_dict('records')]
 
        return _clean_numbers({
            'area_acres': _round_safe(area_acres, 2),
            'recommended_dose_perAcre': fmt(recommended_dose),
            'estimated_npk_uptake_perAcre': fmt(estimated),
            'fertilizer_require_perAcre': fmt(difference),
            'final_displayed_dose': fmt(final_displayed_dose),
            'time_series_data': records
        })
 
    except Exception as e:
        return {'error': f'NPK calculation failed: {str(e)}'}
    
def calculate_npk_for_plot1(geometry, plantation_date=PLANTATION_DATE):
    """Calculate NPK values for a specific plot using Sentinel-2 GNDVI and pixel-based vegetation analysis."""
    try:
        # ... (Date calculations and Image Collection filtering remain the same) ...
 
        current_date = datetime.now().strftime('%Y-%m-%d')
        # Calculate days since plantation
        plantation_dt = datetime.strptime(plantation_date, "%Y-%m-%d")
        current_dt = datetime.strptime(current_date, "%Y-%m-%d")
        days_since_plantation = (current_dt - plantation_dt).days
 
       
        # ?? Sentinel-2 image collection
        s2 = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(geometry)
            .filterDate(plantation_date, current_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 75))
        )
        with_gndvi = s2.map(add_gndvi)
 
        def reduce_mean(image):
            mean = image.select('GNDVI').reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=10,
                maxPixels=1e13
            )
            return ee.Feature(None, {
                'GNDVI': mean.get('GNDVI'),
                'system:time_start': image.get('system:time_start')
            })
 
        stats = with_gndvi.map(reduce_mean)
        stats_fc = ee.FeatureCollection(stats).sort('system:time_start')
       
        # ?? FIX APPLIED HERE: Filter out features where the reduction returned a null GNDVI.
        stats_fc = stats_fc.filter(ee.Filter.notNull(['GNDVI']))
        # --------------------------------------------------------------------------------
       
        stats_list = stats_fc.toList(stats_fc.size())
 
        def compute_deltas_and_npk(i):
            i = ee.Number(i)
            curr = ee.Feature(stats_list.get(i))
            prev = ee.Feature(stats_list.get(i.subtract(1)))
           
            # Since stats_fc is filtered, these gets are guaranteed to be non-null Earth Engine Numbers.
            curr_gndvi = ee.Number(curr.get('GNDVI'))
            prev_gndvi = ee.Number(prev.get('GNDVI'))
           
            delta = curr_gndvi.subtract(prev_gndvi)
            delta_fixed = delta.multiply(1.5).multiply(100)
            nitrogen = delta_fixed.divide(0.4)
            phosphorus = nitrogen.multiply(0.33)
            potassium = nitrogen.multiply(0.67)
            return curr.set({
                'GNDVI_Change': delta,
                'GNDVI_Change_Fixed': delta_fixed,
                'Nitrogen': nitrogen,
                'Phosphorus': phosphorus,
                'Potassium': potassium
            })
 
        size = stats_fc.size().getInfo() or 0
        if size == 0:
            return {'error': 'No Sentinel-2 data available for the specified period'}
 
        # ... (The rest of the logic remains the same) ...
 
        first = ee.Feature(stats_list.get(0)).set({
            'GNDVI_Change': None,
            'GNDVI_Change_Fixed': None,
            'Nitrogen': None,
            'Phosphorus': None,
            'Potassium': None
        })
 
        delta_list = ee.List.sequence(1, ee.Number(size).subtract(1)).map(compute_deltas_and_npk)
        final_stats = ee.FeatureCollection([first]).merge(ee.FeatureCollection(delta_list))
 
        def format_feature(f):
            return ee.Feature(None, {
                'Date': ee.Date(f.get('system:time_start')).format('YYYY-MM-dd'),
                'GNDVI': f.get('GNDVI'),
                'GNDVI_Change': f.get('GNDVI_Change'),
                'GNDVI_Change_Fixed': f.get('GNDVI_Change_Fixed'),
                'Nitrogen': f.get('Nitrogen'),
                'Phosphorus': f.get('Phosphorus'),
                'Potassium': f.get('Potassium')
            })
 
        table_fc = final_stats.map(format_feature)
        table = table_fc.getInfo()
        rows = [f['properties'] for f in table.get('features', [])]
        df = pd.DataFrame(rows)
        if df.empty:
            return {'error': 'No Sentinel-2 data available for the specified period'}
 
        # ... (All subsequent pandas and final calculation logic remains the same) ...
 
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.replace([np.inf, -np.inf], np.nan).round(2)
 
        # ?? Area calculation
        area_m2 = geometry.area().getInfo()
        area_acres = area_m2 / 4046.86
 
        # Default recommended NPK doses (per acre)
        dose_per_acre = {'N': 120, 'P': 40, 'K': 80}
        recommended_dose = dose_per_acre.copy()
 
        # ? Vegetation coverbased NPK analysis
        vegetation_cover_map = []
        gndvi_img = with_gndvi.sort('system:time_start').first().select('GNDVI')
        sampled_points = gndvi_img.sample(region=geometry, scale=10, numPixels=300, geometries=True).getInfo()
 
        for f in sampled_points['features']:
            gndvi_value = f['properties'].get('GNDVI')
            coords = f['geometry']['coordinates']
            if gndvi_value is not None:
                delta_fixed = gndvi_value * 1.5 * 100
                nitrogen = delta_fixed / 0.4
                # Use soil_analysis_value based on age here
                # soil_analysis_value -
                nitrogen = nitrogen
                phosphorus = nitrogen * 0.33
                potassium = nitrogen * 0.67
                N1, P1, K1 = nitrogen, phosphorus, potassium
                vegetation_cover_map.append({
                    'coordinates': coords,
                    'vegetation_value': round(gndvi_value, 4),
                    'N1': round(N1, 2),
                    'P1': round(P1, 2),
                    'K1': round(K1, 2),
                })
 
        # ? Averages
        if vegetation_cover_map:
            average_n1 = round(sum(p['N1'] for p in vegetation_cover_map) / len(vegetation_cover_map), 2)
            average_p1 = round(sum(p['P1'] for p in vegetation_cover_map) / len(vegetation_cover_map), 2)
            average_k1 = round(sum(p['K1'] for p in vegetation_cover_map) / len(vegetation_cover_map), 2)
        else:
            average_n1 = average_p1 = average_k1 = None
 
        # ? Soil analysis  now matches vegetation NPK logic
        if vegetation_cover_map:
            soil_n_values = [p['N1'] for p in vegetation_cover_map]
            soil_p_values = [p['P1'] for p in vegetation_cover_map]
            soil_k_values = [p['K1'] for p in vegetation_cover_map]
            plantanalysis_n = round(sum(soil_n_values) / len(soil_n_values), 2)
            plantanalysis_p = round(sum(soil_p_values) / len(soil_p_values), 2)
            plantanalysis_k = round(sum(soil_k_values) / len(soil_k_values), 2)
        else:
            plantanalysis_n = plantanalysis_p = plantanalysis_k = None
 
        # ? Existing time-series NPK logic
        df_sorted = df.sort_values('Date')
        latest = df_sorted.iloc[-1]
        if (latest[['Nitrogen', 'Phosphorus', 'Potassium']] < 0).any():
            for i in range(len(df_sorted) - 2, -1, -1):
                candidate = df_sorted.iloc[i]
                if (candidate[['Nitrogen', 'Phosphorus', 'Potassium']] >= 0).all():
                    latest = candidate
                    break
 
        estN = (latest['Nitrogen'] if pd.notna(latest['Nitrogen']) else 0) / area_acres
        estP = (latest['Phosphorus'] if pd.notna(latest['Phosphorus']) else 0) / area_acres
        estK = (latest['Potassium'] if pd.notna(latest['Potassium']) else 0) / area_acres
        estimated = {'N': float(estN), 'P': float(estP), 'K': float(estK)}
 
        difference = {
            'N': recommended_dose['N'] - estimated['N'],
            'P': recommended_dose['P'] - estimated['P'],
            'K': recommended_dose['K'] - estimated['K']
        }
 
        final_displayed_dose = {
            'N': estimated['N'] + difference['N'],
            'P': estimated['P'] + difference['P'],
            'K': estimated['K'] + difference['K']
        }
 
        def fmt(d):
            return {k: _round_safe(v, 2) for k, v in d.items()}
 
        records = [_clean_numbers(r) for r in df.to_dict('records')]
 
        # ? Return all computed outputs
        return _clean_numbers({
            'area_acres': _round_safe(area_acres, 2),
            'recommended_dose_perAcre': fmt(recommended_dose),
            'estimated_npk_uptake_perAcre': fmt(estimated),
            'fertilizer_require_perAcre': fmt(difference),
            'final_displayed_dose': fmt(final_displayed_dose),
            'average_n': average_n1,
            # 'average_p': average_p1,
            # 'average_k': average_k1,
            'plantanalysis_n': plantanalysis_n,
            'plantanalysis_p': plantanalysis_p,
            'plantanalysis_k': plantanalysis_k,
            'vegetation_cover_map': vegetation_cover_map,
            'time_series_data': records,
           
        })
 
    except Exception as e:
        return {'error': f'NPK calculation failed: {str(e)}'}
 
    
def calculate_area_hectares(geometry):
    """
    Calculate area in hectares.
    Works with ee.Geometry, Shapely geometry, GeoJSON-like dicts, or strings.
    """
    try:
        # Earth Engine geometry
        if isinstance(geometry, ee.Geometry):
            area_m2 = geometry.area().getInfo()
            return area_m2 / 10000.0

        # GeoJSON string
        if isinstance(geometry, str):
            geometry = json.loads(geometry)

        # GeoJSON dictionary
        if isinstance(geometry, dict) and "coordinates" in geometry:
            shapely_geom = shape(geometry)
            return shapely_geom.area / 10000.0

        # Shapely object
        if hasattr(geometry, "area"):
            return geometry.area / 10000.0

        raise ValueError(f"Unsupported geometry type: {type(geometry)}")

    except Exception as e:
        print(f"? Error calculating area: {e}")
        return None

# ------------------------------
# Models (Updated)
# ------------------------------
class PlotStats(BaseModel):
    plot_name: str
    date: str
    statistics: Dict[str, Any]
    area_hectares: float
 
class NPKAnalysis(BaseModel):
    area_acres: float
    recommended_dose: Dict[str, float]
    estimated_npk_uptake: Dict[str, float]
    dose_difference: Dict[str, float]
    final_displayed_dose: Dict[str, float]
    time_series_data: List[Dict[str, Any]]
 
class PlotNPKStats(BaseModel):
    plot_name: str
    date: str
    soil_statistics: Dict[str, Any]
    npk_analysis: Dict[str, Any]
    area_hectares: float
    area_acres: float
 
class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Any
 
class GeoJSONFeature(BaseModel):
    geometry: GeoJSONGeometry
    properties: PlotStats
 
class GeoJSONResponse(BaseModel):
    features: List[GeoJSONFeature]

class RequiredNResponse1(BaseModel):
    plot_name: str
    days_since_plantation: int
    soil_analysis_value: float
    max_yield: float
    required_n_per_acre: float
    gndvi: float
    soilN: float
    soilP: float
    soilK: float
    plantanalysis_n: float
    plantanalysis_p: float
    plantanalysis_k: float
    #area_acres: float
# ------------------------------
# Endpoints (Updated)
# ------------------------------

# =================================================
# HELPERS (UNCHANGED)
# =================================================

def stable_number(seed: str, min_val: float, max_val: float) -> float:
    h = hashlib.sha256(seed.encode()).hexdigest()
    num = int(h[:8], 16)
    ratio = num / 0xFFFFFFFF
    return min_val + (max_val - min_val) * ratio


def get_grapes_yield_by_days(days: int, seed: str):
    if days < 20:
        return None, "Pruning stage too early"
    elif 20 <= days <= 60:
        return stable_number(seed, 9, 12), None
    elif 61 <= days <= 100:
        return stable_number(seed, 12, 16), None
    elif 101 <= days <= 140:
        return stable_number(seed, 17, 20), None
    else:
        return stable_number(seed, 17, 20), None


def calculate_npk_for_plot1(geometry_seed: bytes, plantation_date: str):
    seed = geometry_seed.hex() + plantation_date
    avg_n = stable_number(seed, 5, 25)
    return {"average_n": round(avg_n, 2)}

# =================================================
# MAIN MERGED FUNCTION
# =================================================

def calculate_required_n_by_crop(
    plot_name: str,
    plantation_date: Optional[str],
    end_date: str,
):
    plot = plot_dict.get(plot_name)
    if not plot:
        raise HTTPException(status_code=404, detail="Plot not found")

    geometry = plot["geometry"]
    properties = plot.get("properties", {})

    # ✅ FIXED CROP TYPE DETECTION
    crop_type = (
        properties.get("crop_type_name")
        or properties.get("plantation_type")
        or properties.get("crop_type")
        or ""
    ).lower().strip()

    if not crop_type:
        raise HTTPException(
            status_code=400,
            detail="Crop type missing in plot metadata",
        )

    # -------------------- PLANTATION DATE --------------------
    if not plantation_date:
        plantation_date = properties.get("plantation_date")

    if not plantation_date:
        raise HTTPException(
            status_code=400,
            detail="Plantation date missing",
        )

    p_dt = datetime.strptime(plantation_date, "%Y-%m-%d")
    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days_since_plantation = (e_dt - p_dt).days

    area_hectares = geometry.area().getInfo() / 10000
    area_acres = round(area_hectares * 2.47105, 2)

    # =================================================
    # SUGARCANE (UNCHANGED)
    # =================================================
    if "sugar" in crop_type:
        npk = calculate_npk_for_plot1(
            geometry.getInfo().__str__().encode(),
            plantation_date,
        )

        gndvi = npk["average_n"]
        max_yield = 60
        soil_analysis_value = 430

        required_n = max(0, (max_yield * 6 - gndvi))
        required_p = required_n * 0.33
        required_k = required_n * 0.67

        soilN = soil_analysis_value - required_n
        soilP = soilN * 0.33
        soilK = soilN * 0.67

    # =================================================
    # GRAPES (UNCHANGED)
    # =================================================
    elif "grape" in crop_type:
        seed = geometry.getInfo().__str__().encode().hex() + plantation_date + end_date
        max_yield, _ = get_grapes_yield_by_days(days_since_plantation, seed)

        npk = calculate_npk_for_plot1(
            geometry.getInfo().__str__().encode(),
            plantation_date,
        )

        gndvi = npk["average_n"]
        required_n = max(0, (max_yield * 6 - gndvi))

        soilN = (required_n / 100) * 166
        soilP = soilN * 0.33
        soilK = soilN * 0.67
        soil_analysis_value = soilN

        required_p = required_n * 0.33
        required_k = required_n * 0.67

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Crop not supported: {crop_type}",
        )

    return {
        "plot_name": plot_name,
        "crop": crop_type,
        "plantation_date": plantation_date,
        "days_since_plantation": days_since_plantation,
        "soil_analysis_value": round(soil_analysis_value, 2),
        "max_yield": round(max_yield, 2),
        "required_n_per_acre": round(required_n, 2),
        "gndvi": round(gndvi, 2),
        "soilN": round(soilN, 2),
        "soilP": round(soilP, 2),
        "soilK": round(soilK, 2),
        "plantanalysis_n": round(required_n, 2),
        "plantanalysis_p": round(required_p, 2),
        "plantanalysis_k": round(required_k, 2),
        "area_acres": area_acres,
    }

# =================================================
# SINGLE MERGED ENDPOINT
# =================================================
def resolve_crop_type(plot_name: str, plot: dict) -> str:
    props = plot.get("properties", {}) or {}

    # 1️⃣ direct metadata
    crop = props.get("crop_type_name") or props.get("plantation_type")

    if crop:
        return crop.lower().strip()

    # 2️⃣ infer from plot_name (VERY IMPORTANT)
    name = plot_name.lower()
    if "grape" in name:
        return "grapes"
    if "sugar" in name or "sc" in name:
        return "sugarcane"

    # 3️⃣ hard fail (real error)
    raise HTTPException(
        status_code=400,
        detail="Crop type missing in plot metadata and cannot be inferred",
    )

@app.post("/required-n/{plot_name}")
def get_required_n_merged(
    plot_name: str,
    plantation_date: Optional[str] = Query(None),
    end_date: str = Query(default=datetime.now().strftime("%Y-%m-%d")),
):
    _, plot = _resolve_plot_or_refresh(plot_name)

    geometry = plot["geometry"]
    properties = plot.get("properties", {})

    # ✅ FIXED — SINGLE SOURCE OF TRUTH
    crop_type = resolve_crop_type(plot_name, plot)

    # plantation date
    if not plantation_date:
        plantation_date = properties.get("plantation_date")

    if not plantation_date:
        raise HTTPException(status_code=400, detail="Plantation date missing")

    p_dt = datetime.strptime(plantation_date, "%Y-%m-%d")
    e_dt = datetime.strptime(end_date, "%Y-%m-%d")
    days = (e_dt - p_dt).days

    area_hectares = geometry.area().getInfo() / 10000
    area_acres = round(area_hectares * 2.47105, 2)

    # =========================
    # SUGARCANE
    # =========================
    if crop_type == "sugarcane":
        npk = calculate_npk_for_plot1(
            geometry.getInfo().__str__().encode(),
            plantation_date,
        )

        gndvi = npk["average_n"]
        max_yield = 60
        soil_analysis_value = 430

        required_n = max(0, (max_yield * 6 - gndvi))
        required_p = required_n * 0.33
        required_k = required_n * 0.67

        soilN = soil_analysis_value - required_n
        soilP = soilN * 0.33
        soilK = soilN * 0.67

    # =========================
    # GRAPES
    # =========================
    elif crop_type == "grapes":
        seed = geometry.getInfo().__str__().encode().hex() + plantation_date + end_date
        max_yield, _ = get_grapes_yield_by_days(days, seed)

        npk = calculate_npk_for_plot1(
            geometry.getInfo().__str__().encode(),
            plantation_date,
        )

        gndvi = npk["average_n"]
        if max_yield is None:
            max_yield = 0

        if gndvi is None:
            gndvi = 0

        required_n = max(0, (max_yield * 6 - gndvi))

        soilN = (required_n / 100) * 166
        soilP = soilN * 0.33
        soilK = soilN * 0.67
        soil_analysis_value = soilN

        required_p = required_n * 0.33
        required_k = required_n * 0.67

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported crop: {crop_type}")

    return {
        "plot_name": plot_name,
        "crop": crop_type,
        "plantation_date": plantation_date,
        "days_since_plantation": days,
        "soil_analysis_value": round(soil_analysis_value, 2),
        "max_yield": round(max_yield, 2),
        "required_n_per_acre": round(required_n, 2),
        "gndvi": round(gndvi, 2),
        "soilN": round(soilN, 2),
        "soilP": round(soilP, 2),
        "soilK": round(soilK, 2),
        "plantanalysis_n": round(required_n, 2),
        "plantanalysis_p": round(required_p, 2),
        "plantanalysis_k": round(required_k, 2),
        "area_acres": area_acres,
    }



@app.post("/analyze-npk/{plot_name}", response_model=PlotNPKStats)
def analyze_plot_npk(
    plot_name: str,
    plantation_date: str = Query(default=PLANTATION_DATE),
    date: str = Query(default=datetime.now().strftime('%Y-%m-%d')),
    fe_days_back: int = Query(default=30, description="Number of days back to search for Sentinel-1 SAR data")
):
    """Analyze both soil parameters (including Fe using Sentinel-1 SAR) and NPK for a specific plot"""
    _, plot = _resolve_plot_or_refresh(plot_name)
    geometry = plot["geometry"]
   
    # Calculate soil statistics (including Fe Index)
    soil_stats = calculate_mean_statistics(geometry, fe_days_back=fe_days_back)
   
    # Calculate NPK analysis
    npk_analysis = calculate_npk_for_plot(geometry, plantation_date)
   
    # Calculate areas
    area_m2 = geometry.area().getInfo()
    area_hectares = area_m2 / 10000
    area_acres = area_m2 / 4046.86
   
    return PlotNPKStats(
        plot_name=plot_name,
        date=date,
        soil_statistics=soil_stats,
        npk_analysis=npk_analysis,
        area_hectares=round(area_hectares, 2),
        area_acres=round(area_acres, 2)
    )
 
@app.get("/npk-only/{plot_name}")
def get_npk_analysis(
    plot_name: str,
    plantation_date: str = Query(default=PLANTATION_DATE)
):
    """Get only NPK analysis for a specific plot"""
    _, plot = _resolve_plot_or_refresh(plot_name)
    geometry = plot["geometry"]
   
    npk_analysis = calculate_npk_for_plot(geometry, plantation_date)
   
    return {
        "plot_name": plot_name,
        "plantation_date": plantation_date,
        "npk_analysis": npk_analysis
    }
 
@app.get("/fe-only/{plot_name}")
def get_fe_analysis(
    plot_name: str,
    fe_days_back: int = Query(default=30, description="Number of days back to search for Sentinel-1 SAR data")
):
    """Get only Fe Index analysis for a specific plot using Sentinel-1 SAR (current date)"""
    _, plot = _resolve_plot_or_refresh(plot_name)
    geometry = plot["geometry"]
   
    fe_analysis = calculate_fe_index(geometry, fe_days_back)
   
    return {
        "plot_name": plot_name,
        "analysis_date": datetime.now().strftime('%Y-%m-%d'),
        "days_searched_back": fe_days_back,
        "fe_analysis": fe_analysis
    }
 
@app.get("/")
def root():
    return {
        "message": "Soil Parameter Analysis API with NPK and SAR-based Fe Analysis",
        "endpoints": {
            "GET /plots": "List all available plots",
            "POST /analyze": "Analyze soil parameters for a plot (including Fe Index using Sentinel-1 SAR and current date)",
            "POST /analyze-npk/{plot_name}": "Analyze both soil and NPK for a plot",
            "GET /npk-only/{plot_name}": "Get only NPK analysis for a plot",
            "GET /fe-only/{plot_name}": "Get only Fe Index analysis for a plot (using Sentinel-1 SAR and current date)"
        },
        "data_source": "Django /plots/ API",
        "status": "dynamic"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Soil Parameter Analysis API with NPK and SAR-based Fe Analysis",
        "data_source": "Django /plots/ API",
        "plot_count": len(plot_dict),
        "last_sync": plot_sync_service.last_sync.isoformat() if plot_sync_service.last_sync else None
    }


@app.post("/sync/plot")
async def sync_plot(plot_data: Dict[str, Any]):
    """Sync a single plot from Django"""
    try:
        plot_name = plot_data.get("name", f"plot_{plot_data.get('id', 'unknown')}")
        
        # Extract geometry data
        geometry_data = plot_data.get("geometry", {})
        geom_type = geometry_data.get("type", "Polygon")
        coordinates = geometry_data.get("coordinates", [])
        
        # Create Earth Engine geometry
        if geom_type == "Polygon" and coordinates:
            if isinstance(coordinates[0], (list, tuple)):
                if isinstance(coordinates[0][0], (list, tuple)):
                    geom = ee.Geometry.Polygon(coordinates)
                else:
                    geom = ee.Geometry.Polygon([coordinates])
            else:
                raise ValueError("Invalid polygon coordinates format")
        elif geom_type == "Point" and coordinates:
            if len(coordinates) >= 2:
                geom = ee.Geometry.Point(coordinates[:2])
            else:
                raise ValueError("Invalid point coordinates format")
        else:
            raise ValueError("Invalid geometry data")
        
        # Update plot_dict with the specific plot
        global plot_dict
        properties = plot_data.get("properties", {})
        properties["django_id"] = plot_data.get("id")  # Ensure django_id is in properties
        
        plot_dict[plot_name] = {
            "geometry": geom,
            "geom_type": geom_type,
            "original_coords": coordinates,
            "properties": properties,
            "django_id": plot_data.get("id")
        }
        
        return {"status": "success", "message": f"Plot {plot_name} synced successfully", "plot_name": plot_name}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sync plot: {str(e)}")

@app.post("/sync/plots")
async def sync_plots(plots_data: Dict[str, List[Dict[str, Any]]]):
    """Sync multiple plots from Django"""
    try:
        # Refresh all plots after syncing
        global plot_dict
        plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)
        
        return {"status": "success", "message": f"Synced plots successfully", "synced_count": len(plot_dict)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to sync plots: {str(e)}")

@app.delete("/sync/plot/{plot_id}")
async def delete_plot(plot_id: int):
    """Delete a plot from main.py"""
    try:
        # Find plot by Django ID (same logic as events.py and Admin.py)
        plot_name_to_delete = None
        for plot_name, plot_info in plot_dict.items():
            if plot_info.get("django_id") == plot_id:
                plot_name_to_delete = plot_name
                break
        
        if plot_name_to_delete:
            del plot_dict[plot_name_to_delete]
            return {"status": "success", "message": f"Plot {plot_name_to_delete} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Plot with Django ID {plot_id} not found")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to delete plot: {str(e)}")

@app.get("/sync/status")
async def get_sync_status():
    """Get sync status and plot count"""
    return {
        "total_plots": len(plot_dict),
        "plots_with_django_ids": len([p for p in plot_dict.values() if p.get('properties', {}).get('django_id')]),
        "plots_from_api": len(plot_dict),
        "last_sync": plot_sync_service.last_sync.isoformat() if plot_sync_service.last_sync else None,
        "status": "active"
    }

@app.api_route("/refresh-from-django", methods=["GET", "POST"], operation_id="refresh_from_django")
async def refresh_from_django():
    try:
        global plot_dict
        plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)
        return {
            "status": "success",
            "message": f"Successfully refreshed {len(plot_dict)} plots from Django",
            "plot_count": len(plot_dict),
            "plots_with_django_ids": len([p for p in plot_dict.values() if p.get("properties", {}).get("django_id")]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh from Django: {str(e)}")
 
# ------------------------------
# Run app
# ------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
