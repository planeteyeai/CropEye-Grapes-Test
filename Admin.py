from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any, Union
import ee
import json
import requests
from datetime import datetime, timedelta, date
import uvicorn
from contextlib import asynccontextmanager
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import numpy as np
import os
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from pydantic import BaseModel
from shapely.geometry import shape, Point, Polygon
from geopy.distance import geodesic
from shared_services import PlotSyncService, PlotKeepAliveManager
from events import generate_brix_time_series, compute_brix_image

# Initialize Earth Engine - move this to the top
raw = os.environ["EE_SERVICE_ACCOUNT_JSON"]
service_account = raw if isinstance(raw, str) else json.dumps(raw)
ee.Initialize(ee.ServiceAccountCredentials(None, key_data=service_account))
    
app =FastAPI(tittle="Admin API for SAR Index Mapping with Pest Detection")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response  
class PlotInfo(BaseModel):
    name: str
    geometry_type: str
    area_hectares: Optional[float] = None
    recent_dates: List[str]

class IndexClassification(BaseModel):
    class_name: str
    value_range: str
    pixel_count: int
    percentage: float

class IndexAnalysis(BaseModel):
    index_name: str
    classifications: List[IndexClassification]
    total_pixels: int

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]

# Pydantic model for input
class FactoryLocation(BaseModel):
    lat: float
    lon: float

# Pydantic model for output
class FarmerDistance(BaseModel):
    name: str
    lat: float
    lon: float
    distance_km: float
class PlotBoundaryProperties(BaseModel):
    plot_name: str
    start_date: str
    end_date: str
    center_coordinates: List[float]
    area_hectares: Optional[float] = None
    feature_type: str = "plot_boundary"
    indices_analysis: List[IndexAnalysis]
    tile_urls: Dict[str, Optional[str]] = {}
    image_count: int = 0
    last_updated: str = ""
    satellite_update_date: Optional[str] = None

class PixelProperties(BaseModel):
    Name: str
    plot_id: str
    pixel_id: int
    VV: Optional[float] = None
    VH: Optional[float] = None
    VV_VH_ratio: Optional[float] = None
    feature_type: str = "pixel_data"
    analysis_date: str

# New Pydantic models for pest detection
class PestDetectionProperties(BaseModel):
    plot_name: str
    start_date: str
    end_date: str
    center_coordinates: List[float]
    area_hectares: Optional[float] = None
    feature_type: str = "pest_detection"
    pest_threshold: float
    nir_threshold: float
    ndwi_threshold: float
    tile_url: Optional[str] = None
    image_count: int = 0
    last_updated: str = ""
    satellite_update_date: Optional[str] = None

class PestDetectionStats(BaseModel):
    total_pixels: int
    pest_affected_pixels: int
    pest_percentage: float
    healthy_pixels: int
    healthy_percentage: float

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: Union[PlotBoundaryProperties, PixelProperties, PestDetectionProperties]

class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]

class DateRange(BaseModel):
    start_date: str
    end_date: str

# Update indexVisParams for Sentinel-1 (assuming this is correct for Admin.py)
indexVisParams = {
    'VV': {
        'min': -25,
        'max': 0,
        'palette': ['#00FFFF','#008000', '#FFFF00', '#FF0000'] 
    },
    'VH': {
        'min': -30,
        'max': 0,
        'palette': ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']
    },
    'VV_VH_ratio': {
        'min': 0,
        'max': 10,
        'palette': ['#FFFFFF', '#00FF00', '#FF0000']
    },
    'SWI': {
        'min': -1,
        'max': 1,
        'palette': ['#8B0000','#FFFF00',"#00BFFF",'#0000FF','#000080','#000080','#000080']

 
    },
    'RVI': {
        'min': 0,
        'max': 1,
        'palette': ['#8B0000', '#FFA500', '#FFFF00', '#00FFFF', '#0000FF']
    }
}


# Initialize plot sync service
plot_sync_service = PlotSyncService()
keepalive_manager: Optional[PlotKeepAliveManager] = None
plot_dict: Dict[str, Dict[str, Any]] = {}

def _apply_plot_update(new_plots: Dict[str, Dict[str, Any]]) -> None:
    global plot_dict
    plot_dict = new_plots
    print(f"Admin.py background refresh: {len(plot_dict)} plots")

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

def get_latest_satellite_update(collection) -> str:
    """Get the latest satellite update date from the collection"""
    try:
        size = collection.size().getInfo()
        if size == 0:
            return "no_data"

        # Get the most recent image timestamp
        latest_image = collection.sort('system:time_start', False).first()
        latest_timestamp = latest_image.get('system:time_start').getInfo()  

        # Convert to readable date
        latest_date = datetime.fromtimestamp(latest_timestamp / 1000)
        return latest_date.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error getting latest satellite update: {e}")
        return "unknown"

def get_tile_url(image, vis_params, name):
    """Generate tile URL for the given image and visualization parameters"""
    try:
        map_id_dict = ee.Image(image).getMapId(vis_params)
        tile_url = map_id_dict['tile_fetcher'].url_format
        print(f"\n?? {name} Tile URL:")
        print(f"   {tile_url}")
        return tile_url
    except Exception as e:
        print(f"Error generating tile URL for {name}: {str(e)}")
        return None

# Initialize Earth Engine and load data
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        global plot_dict, keepalive_manager
        print("?? Admin.py: Initializing application and fetching plots from Django API...")
        keepalive_manager = PlotKeepAliveManager(plot_sync_service, _apply_plot_update, 3.0, 5.0)
        plot_dict = await keepalive_manager.refresh_now(force=True)
        await keepalive_manager.start()
        print(f"? Admin.py startup: Loaded {len(plot_dict)} plots from Django")
        print("? Admin.py: Application initialized successfully")
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        raise
    yield
    # Shutdown
    if keepalive_manager:
        await keepalive_manager.stop()
    print("Shutting down FastAPI application")

# Create FastAPI app
app = FastAPI(
    title="SAR Index Mapping API with Pest Detection",
    description="API for SAR index analysis and pest detection using Sentinel-1 and Sentinel-2 data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)

def get_recent_dates():
    """Get list of recent dates for date selection"""
    today = datetime.today()
    last_7_dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    last_7_dates.reverse()
    return last_7_dates

def filter_s1(collection, start, end, aoi):
    return (collection
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        .sort('system:time_start')
    )

def filter_s2(collection, start, end, aoi, cloud_threshold=30):
    """Filter Sentinel-2 collection"""
    return (collection
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
    )

def addIndices(image):
    vv = image.select('VV')
    vh = image.select('VH')
    vv_vh_ratio = vv.divide(vh).rename('VV_VH_ratio')
    swi = vv.subtract(vh).divide(vv.add(vh)).rename('SWI')
    rvi = vh.multiply(4).divide(vv.add(vh)).rename('RVI')
    return image.addBands([vv, vh, vv_vh_ratio, swi, rvi])
 
def safe_median(collection):
    """Safely compute median of collection with fallback"""
    size = collection.size()
    return ee.Image(ee.Algorithms.If(
        size.gt(0),
        collection.median(),
        None
    ))
 
def get_alternative_image(aoi, target_date):
    """Get alternative image when no images found in date range"""
    try:
        target_date_obj = datetime.strptime(target_date, '%Y-%m-%d')
       
        # Try 60 days back
        extended_start = (target_date_obj - timedelta(days=60)).strftime('%Y-%m-%d')
        print(f"Trying extended date range: {extended_start} to {target_date}")
       
        extended_coll = filter_s1(
            ee.ImageCollection('COPERNICUS/S1_GRD'),
            extended_start, target_date, aoi
        ).map(addIndices)
       
        extended_size = extended_coll.size().getInfo()
        print(f"Extended collection size: {extended_size}")
       
        if extended_size > 0:
            return extended_coll.sort('system:time_start', False).first()
       
        # If still no images, try even broader search (6 months)
        very_extended_start = (target_date_obj - timedelta(days=180)).strftime('%Y-%m-%d')
        very_extended_coll = filter_s1(
            ee.ImageCollection('COPERNICUS/S1_GRD'),
            very_extended_start, target_date, aoi
        ).map(addIndices)
       
        very_extended_size = very_extended_coll.size().getInfo()
        print(f"Very extended collection size: {very_extended_size}")
       
        if very_extended_size > 0:
            return very_extended_coll.sort('system:time_start', False).first()
           
        return None
    except Exception as e:
        print(f"Error getting alternative image: {e}")
        return None
 
def classify_index(index_name: str, image, aoi) -> IndexAnalysis:
    """Classify index values"""
    if index_name == 'VV':
        # Apply 20-class vegetation bins for VV
        classified = (
            image.select('VV')
            .where(image.select('VV').gte(-9), 1) 
            .where(image.select('VV').gte(-10).And(image.select('VV').lt(-9)), 2)
            .where(image.select('VV').gte(-11).And(image.select('VV').lt(-10)), 3)
            .where(image.select('VV').gte(-12).And(image.select('VV').lt(-11)), 4)
            .where(image.select('VV').gte(-13).And(image.select('VV').lt(-12)), 5)
            .where(image.select('VV').lt(-13), 6)
        )
        
        labels = {
            1: 'Healthy >= -9 ', 2: 'Healthy -10 to -9', 3: 'Moderate -11 to -10 ',
            4: 'Stress -12 to -11', 5: 'Weak -13 to -12',6: 'Weak < -13'
        }

        scale = 10      
    elif index_name == 'VH':
        classified = (
            image.select('VH')
            .where(image.select('VH').gte(-5), 1)
            .where(image.select('VH').gte(-10).And(image.select('VH').lt(-5)), 2)
            .where(image.select('VH').gte(-15).And(image.select('VH').lt(-10)), 3)
            .where(image.select('VH').lt(-15), 4)
        )
       
        labels = {
            1: f'High VH (>= -5 dB)',
            2: f'Medium VH (-10 to -5 dB)',
            3: f'Low VH (-15 to -10 dB)',
            4: f'Very Low VH (< -15 dB)'
        }
        scale = 10
    elif index_name == 'VV_VH_ratio':
        classified = (
            image.select('VV_VH_ratio')
            .where(image.select('VV_VH_ratio').gte(6), 1)
            .where(image.select('VV_VH_ratio').gte(3).And(image.select('VV_VH_ratio').lt(6)), 2)
            .where(image.select('VV_VH_ratio').gte(1).And(image.select('VV_VH_ratio').lt(3)), 3)
            .where(image.select('VV_VH_ratio').lt(1), 4)
        )
       
        labels = {
            1: 'Very High Ratio (>= 6)', 2: 'High Ratio (3-6)',
            3: 'Medium Ratio (1-3)', 4: 'Low Ratio (< 1)'
        }
        scale = 10
    elif index_name == 'SWI':
        classified = (
            image.select('SWI')
            .where(image.select('SWI').gte(0.5), 1)
            .where(image.select('SWI').gte(0.2).And(image.select('SWI').lt(0.5)), 2)
            .where(image.select('SWI').gte(0).And(image.select('SWI').lt(0.2)), 3)
            .where(image.select('SWI').gte(-0.1).And(image.select('SWI').lt(0)), 4)
            .where(image.select('SWI').gte(-0.15).And(image.select('SWI').lt(-0.1)), 5)
            .where(image.select('SWI').gte(-0.25).And(image.select('SWI').lt(-0.15)), 6)
            .where(image.select('SWI').gte(-0.3).And(image.select('SWI').lt(-0.25)), 7)
            .where(image.select('SWI').gte(-0.4).And(image.select('SWI').lt(-0.3)), 8)
            .where(image.select('SWI').gte(-0.5).And(image.select('SWI').lt(-0.4)), 9)
            .where(image.select('SWI').gte(-0.6).And(image.select('SWI').lt(-0.5)), 10)
            .where(image.select('SWI').gte(-0.65).And(image.select('SWI').lt(-0.6)), 11)
            .where(image.select('SWI').gte(-0.7).And(image.select('SWI').lt(-0.65)), 12)
            .where(image.select('SWI').gte(-0.75).And(image.select('SWI').lt(-0.7)), 13)
            .where(image.select('SWI').lt(-0.75), 14)
        )

        labels = {
            1: 'Water Bodies 0.5-0.6', 2: 'Water Bodies 0.2-0.5', 3: 'Water Bodies 0-0.2',
            4: 'Shallow Water -0.1-0', 5: 'Moist Ground -0.15--0.1', 6: 'Moist Ground -0.25--0.15',
            7: 'Moist Ground -0.3--0.25', 8: 'Water Stress -0.4--0.3', 9: 'Water Stress -0.5--0.4',
            10: 'Dry -0.6--0.5', 11: 'Dry -0.65--0.6', 12: 'Dry -0.7--0.65',
            13: 'Dry -0.75--0.7', 14: 'Dry < -0.75'
        }
        scale = 10
    elif index_name == 'RVI':
        classified = (
            image.select('RVI')
            .where(image.select('RVI').gte(0.90), 1)
            .where(image.select('RVI').gte(0.80).And(image.select('RVI').lt(0.90)), 2)
            .where(image.select('RVI').gte(0.70).And(image.select('RVI').lt(0.80)), 3)
            .where(image.select('RVI').gte(0.60).And(image.select('RVI').lt(0.70)), 4)
            .where(image.select('RVI').gte(0.50).And(image.select('RVI').lt(0.60)), 5)
            .where(image.select('RVI').gte(0.40).And(image.select('RVI').lt(0.50)), 6)
            .where(image.select('RVI').gte(0.30).And(image.select('RVI').lt(0.40)), 7)
            .where(image.select('RVI').gte(0.20).And(image.select('RVI').lt(0.30)), 8)
            .where(image.select('RVI').gte(0.10).And(image.select('RVI').lt(0.20)), 9)
            .where(image.select('RVI').gte(0.00).And(image.select('RVI').lt(0.10)), 10)
            .where(image.select('RVI').gte(-0.10).And(image.select('RVI').lt(0.00)), 11)
            .where(image.select('RVI').gte(-0.20).And(image.select('RVI').lt(-0.10)), 12)
            .where(image.select('RVI').gte(-0.30).And(image.select('RVI').lt(-0.20)), 13)
            .where(image.select('RVI').gte(-0.40).And(image.select('RVI').lt(-0.30)), 14)
            .where(image.select('RVI').gte(-0.50).And(image.select('RVI').lt(-0.40)), 15)
            .where(image.select('RVI').gte(-0.60).And(image.select('RVI').lt(-0.50)), 16)
            .where(image.select('RVI').gte(-0.70).And(image.select('RVI').lt(-0.60)), 17)
            .where(image.select('RVI').gte(-0.80).And(image.select('RVI').lt(-0.70)), 18)
            .where(image.select('RVI').gte(-0.90).And(image.select('RVI').lt(-0.80)), 19)
            .where(image.select('RVI').lt(-0.90), 20)
        )
       
        labels = {
            1: 'Excess 0.90-1.00', 2: 'Excess 0.80-0.90', 3: 'ADEQUATE 0.70-0.80',
            4: 'ADEQUATE 0.60-0.70', 5: 'ADEQUATE 0.50-0.60', 6: 'ADEQUATE 0.40-0.50',
            7: 'Sufficient Uptake 0.30-0.40', 8: 'Sufficient Uptake 0.20-0.30', 9: 'Less uptake 0.10-0.20',
            10: 'Less uptake 0.00-0.10', 11: 'Less uptake -0.10-0.00', 12: 'Less uptake -0.20--0.10',
            13: 'Dry -0.30--0.20', 14: 'Dry -0.40--0.30', 15: 'Dry -0.50--0.40',
            16: 'Dry -0.60--0.50', 17: 'Dry -0.70--0.60', 18: 'Dry -0.80--0.70',
            19: 'Dry -0.90--0.80', 20: 'Dry -1.00--0.90'
        }
        scale = 10
    else:
        return IndexAnalysis(
            index_name=index_name,
            classifications=[],
            total_pixels=0
        )
   
    try:
        counts = classified.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=scale,
            maxPixels=1e9
        ).getInfo()
    except Exception as e:
        print(f"Error in reduceRegion: {e}")
        return IndexAnalysis(
            index_name=index_name,
            classifications=[],
            total_pixels=0
        )

    classifications = []
    total_pixels = 0

    for key, val in counts.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                total_pixels += sub_val

    for key, val in counts.items():
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                class_id = int(float(sub_key))
                label = labels.get(class_id, f"Class {class_id}")
                sub_val_int = int(sub_val)
                percentage = (sub_val_int / total_pixels) * 100 if total_pixels > 0 else 0

                classifications.append(IndexClassification(
                    class_name=label,
                    value_range=label.split()[-1] if '(' in label else label,
                    pixel_count=sub_val_int,
                    percentage=round(percentage, 2)
                ))

    return IndexAnalysis(
        index_name=index_name,
        classifications=classifications,
        total_pixels=int(round(total_pixels)) if total_pixels else 0
    )

def calculate_center_coordinates(coords, geom_type):
    """Calculate center coordinates from geometry"""
    if geom_type == "Polygon":
        ring = coords[0]
        lons = [point[0] for point in ring]
        lats = [point[1] for point in ring]
        return [sum(lats) / len(lats), sum(lons) / len(lons)]
    elif geom_type == "MultiPolygon":
        ring = coords[0][0]
        lons = [point[0] for point in ring]
        lats = [point[1] for point in ring]
        return [sum(lats) / len(lats), sum(lons) / len(lons)]
    return [0, 0]

def calculate_area_hectares(geometry):
    """Calculate area in hectares"""
    try:
        area_m2 = geometry.area().getInfo()
        return round(area_m2 / 10000, 2)
    except Exception as e:
        print(f"Error calculating area: {e}")
        return None

def generate_pixel_features(image, aoi, plot_name, analysis_date, num_pixels=100):
    """Generate pixel-level features for the plot"""
    try:
        sample_points = image.select(['VV', 'VH', 'VV_VH_ratio', 'SWI', 'RVI']).sample(
            region=aoi,
            scale=10,
            numPixels=num_pixels,
            geometries=True
        )

        samples = sample_points.getInfo()
        pixel_features = []
    
        for i, sample in enumerate(samples.get('features', [])):
            coordinates = sample['geometry']['coordinates']
            properties = sample['properties']
    
            pixel_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": coordinates
                },
                "properties": {
                    "Name": f"{plot_name}_pixel_{i}",
                    "plot_id": plot_name,
                    "pixel_id": i,
                    "VV": round(properties.get('VV', 0), 4) if properties.get('VV') is not None else None,
                    "VH": round(properties.get('VH', 0), 4) if properties.get('VH') is not None else None,
                    "VV_VH_ratio": round(properties.get('VV_VH_ratio', 0), 4) if properties.get('VV_VH_ratio') is not None else None,
                    "SWI": round(properties.get('SWI', 0), 4) if properties.get('SWI') is not None else None,
                    "RVI": round(properties.get('RVI', 0), 4) if properties.get('RVI') is not None else None,
                    "feature_type": "pixel_data",
                    "analysis_date": analysis_date
                }
            }
            pixel_features.append(pixel_feature)
       
        return pixel_features
    except Exception as e:
        print(f"Error generating pixel features: {e}")
        return []
 
def detect_pest(s2_image, aoi, pest_threshold=0.3, nir_threshold=0.15, ndwi_threshold=0.4):
    """Perform pest detection analysis using Sentinel-2 data"""
    try:
        # Calculate indices
        false_color = s2_image.select(['B8', 'B4', 'B3']).rename(['nir', 'red', 'green'])
        false_color_norm = false_color.divide(10000).clamp(0, 1)
       
        ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('ndwi')
        ndwi_norm = ndwi.add(1).divide(2).clamp(0, 1)
        ndwi_rgb = ee.Image.cat([ndwi_norm, ndwi_norm, ndwi_norm]).rename(['nir', 'red', 'green'])
       
        averaged_image = false_color_norm.add(ndwi_rgb).divide(2)
       
        ndvi = s2_image.normalizedDifference(['B8', 'B4']).rename('ndvi')
        red_edge_ndvi = s2_image.normalizedDifference(['B8', 'B5']).rename('red_edge_ndvi')
       
        # PEST Detection Logic
        low_ndvi = ndvi.lt(pest_threshold)
        low_nir = false_color_norm.select('nir').lt(nir_threshold)
        water_stress = ndwi_norm.lt(ndwi_threshold)
        pest_mask = low_ndvi.And(low_nir).Or(water_stress.And(low_ndvi))
       
        # Visualization Image
        image_8bit = averaged_image.multiply(255).uint8()
        avg_band = image_8bit.select(['nir', 'red', 'green']).reduce(ee.Reducer.mean())
        red_channel = avg_band.expression('255 - b()').uint8()
        green_channel = avg_band.uint8()
        blue_channel = ee.Image.constant(0).uint8()
       
        base_rgb = ee.Image.cat([red_channel, green_channel, blue_channel]).rename(['red', 'green', 'blue'])
        black_rgb = ee.Image.constant([0, 0, 0]).uint8()
        final_image = base_rgb.where(pest_mask, black_rgb)
       
        # Calculate pest statistics
        pest_stats = pest_mask.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9
        ).getInfo()
       
        total_pixels = 0
        pest_affected_pixels = 0
       
        for key, val in pest_stats.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    if float(sub_key) == 1:  # Pest affected pixels
                        pest_affected_pixels = int(round(sub_val))
                    total_pixels += int(round(sub_val))
       
        healthy_pixels = total_pixels - pest_affected_pixels
        pest_percentage = (pest_affected_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        healthy_percentage = (healthy_pixels / total_pixels) * 100 if total_pixels > 0 else 0
       
        stats = PestDetectionStats(
            total_pixels=int(total_pixels),
            pest_affected_pixels=int(pest_affected_pixels),
            pest_percentage=round(pest_percentage, 2),
            healthy_pixels=int(healthy_pixels),
            healthy_percentage=round(healthy_percentage, 2)
        )
       
        return final_image, pest_mask, stats
       
    except Exception as e:
        print(f"Error in pest detection: {e}")
        raise e
 
def default_start_date(end_date: str = None):
    if end_date is None:
        end_date_obj = date.today()
    else:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d').date()
        except Exception:
            end_date_obj = date.today()
    start_date_obj = end_date_obj - timedelta(days=15)
    return start_date_obj.strftime('%Y-%m-%d')
 
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "SAR Index Mapping API with Pest Detection", "version": "1.0.0"}
 
@app.get("/plots", response_model=List[str])
async def get_plots():
    """Get list of available plots"""
    return list(plot_dict.keys())
 
@app.get("/plots/{plot_name}/info", response_model=PlotInfo)
async def get_plot_info_with_dates(plot_name: str):
    """Get information about a specific plot including recent dates"""
    _, plot = _resolve_plot_or_refresh(plot_name)
    geom = plot['geometry']
    try:
        area = geom.area().divide(10000).getInfo()
    except Exception as e:
        print(f"Error calculating area: {e}")
        area = None
   
    recent_dates = get_recent_dates()
   
    return PlotInfo(
        name=plot_name,
        geometry_type=geom.type().getInfo(),
        area_acre=round(area*2.47105, 2) if area else None,
        recent_dates=recent_dates
    )


@app.post("/analyze_Growth", response_model=Dict[str, Any])
async def analyze_plot_combined(
    plot_name: str = Query(..., description="Name of the plot to analyze"),
   end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        description="End date for analysis (defaults to tomorrow's date)"
    ),
    start_date: str = Depends(default_start_date)
):
    """
    Analyze plot growth using Sentinel-1 VH or Sentinel-2 NDVI.
    Sentinel-2 is preferred if both datasets are available.
    """
    try:
        _, plot_data = _resolve_plot_or_refresh(plot_name)
        geometry = plot_data["geometry"]
        area_hectares = geometry.area().divide(10000).getInfo()

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)

        # -------------------- SENTINEL-2 --------------------
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
            .sort("system:time_start", False)
        )

        s2_count = s2_collection.size().getInfo()
        latest_s2_image = None
        latest_s2_date = None
        if s2_count > 0:
            latest_s2_image = ee.Image(s2_collection.first())
            latest_s2_date = ee.Date(latest_s2_image.get("system:time_start"))

        # -------------------- SENTINEL-1 --------------------
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .map(lambda img: img.clip(geometry))
            .sort("system:time_start", False)
        )

        s1_count = s1_collection.size().getInfo()
        latest_s1_image = None
        previous_s1_image = None
        latest_s1_date = None
        if s1_count >= 2:
            latest_s1_image = ee.Image(s1_collection.toList(2).get(0))
            previous_s1_image = ee.Image(s1_collection.toList(2).get(1))
            latest_s1_date = ee.Date(latest_s1_image.get("system:time_start"))
        elif s1_count == 1:
            latest_s1_image = ee.Image(s1_collection.first())
            latest_s1_date = ee.Date(latest_s1_image.get("system:time_start"))

        # -------------------- DECIDE WHICH DATA TO USE --------------------
        use_s2 = False
        use_s1 = False

        if latest_s2_date and latest_s1_date:
            # Prefer Sentinel-2 if both available
            use_s2 = latest_s2_date.millis().getInfo() >= latest_s1_date.millis().getInfo()
            use_s1 = not use_s2
        elif latest_s2_date:
            use_s2 = True
        elif latest_s1_date:
            use_s1 = True
        else:
            raise HTTPException(status_code=404, detail="No Sentinel-1 or Sentinel-2 images found for the given date range.")

        # -------------------- ANALYSIS LOGIC --------------------
        if use_s2:
            analysis = latest_s2_image.normalizedDifference(["B8", "B4"]).rename("NDVI").clip(geometry)
            ndvi = analysis.select("NDVI")
            weak_mask = ndvi.gte(0.2).And(ndvi.lt(0.4))
            stress_mask = ndvi.gte(0.0).And(ndvi.lt(0.2))
            moderate_mask = ndvi.gte(0.4).And(ndvi.lt(0.6))
            healthy_mask = ndvi.gte(0.6)
            latest_image_date = latest_s2_date.format("YYYY-MM-dd").getInfo()
            data_source = "Sentinel-2 NDVI"
        else:
            analysis = latest_s1_image.select("VH").clip(geometry)
            vh = analysis.select("VH")
            weak_mask = vh.gte(-11)
            stress_mask = vh.lt(-11).And(vh.gt(-13))
            moderate_mask = vh.lte(-13).And(vh.gt(-15))
            healthy_mask = vh.lte(-15)
            latest_image_date = latest_s1_date.format("YYYY-MM-dd").getInfo()
            data_source = "Sentinel-1 VH"

        # -------------------- COMBINED VISUALIZATION --------------------
        combined_class = (
            ee.Image(0)
            .where(weak_mask, 1)
            .where(stress_mask, 2)
            .where(moderate_mask, 3)
            .where(healthy_mask, 4)
            .clip(geometry)
        )

        combined_smooth = combined_class.focal_mean(radius=10, units="meters")
        combined_vis_params = {
            "min": 0,
            "max": 4,
            "palette": ["#bc1e29", "#58cf54", "#28ae31", "#056c3e"]
        }
        combined_smooth_vis = combined_smooth.visualize(**combined_vis_params).clip(geometry)
        tile_url = combined_smooth_vis.getMapId()["tile_fetcher"].url_format

        # -------------------- PIXEL COUNTS --------------------
        count_image = ee.Image.constant(1)

        def get_pixel_count(mask):
            return count_image.updateMask(mask).reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant")

        healthy_count = get_pixel_count(healthy_mask).getInfo() or 0
        moderate_count = get_pixel_count(moderate_mask).getInfo() or 0
        weak_count = get_pixel_count(weak_mask).getInfo() or 0
        stress_count = get_pixel_count(stress_mask).getInfo() or 0
        total_pixel_count = get_pixel_count(count_image).getInfo() or 0

        # -------------------- PIXEL COORDINATES --------------------
        def mask_to_coords(mask):
            points = mask.selfMask().addBands(ee.Image.pixelLonLat()).sample(
                region=geometry, scale=10, geometries=True, tileScale=4
            ).getInfo()
            coords = [f["geometry"]["coordinates"] for f in points.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]

        healthy_coords = mask_to_coords(healthy_mask)
        moderate_coords = mask_to_coords(moderate_mask)
        weak_coords = mask_to_coords(weak_mask)
        stress_coords = mask_to_coords(stress_mask)

        # -------------------- RESPONSE --------------------
        feature = {
            "type": "Feature",
            "geometry": {
                "type": plot_data["geom_type"],
                "coordinates": plot_data["original_coords"]
            },
            "properties": {
                "plot_name": plot_name,
                "area_acres": round(area_hectares * 2.471, 2),
                "start_date": start_date,
                "end_date": end_date,
                "image_count": 1,
                "tile_url": tile_url,
                "data_source": data_source,
                "latest_image_date": latest_image_date,
                "last_updated": datetime.now().isoformat()
            }
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,

                "healthy_pixel_count": healthy_count,
                "healthy_pixel_percentage": (healthy_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "healthy_pixel_coordinates": healthy_coords,

                "moderate_pixel_count": moderate_count,
                "moderate_pixel_percentage": (moderate_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "moderate_pixel_coordinates": moderate_coords,

                "weak_pixel_count": weak_count,
                "weak_pixel_percentage": (weak_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "weak_pixel_coordinates": weak_coords,

                "stress_pixel_count": stress_count,
                "stress_pixel_percentage": (stress_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "stress_pixel_coordinates": stress_coords,

                "analysis_start_date": analysis_start.format("YYYY-MM-dd").getInfo(),
                "analysis_end_date": analysis_end.format("YYYY-MM-dd").getInfo(),
                "latest_image_date": latest_image_date
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Growth analysis failed: {str(e)}")



    

def Water(end_date: str):
    """Return 30 days before the given end_date"""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=25)
    return start_dt.strftime("%Y-%m-%d")


@app.post("/wateruptake", response_model=Dict[str, Any])
async def analyze_water_uptake(
    plot_name: str,
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        description="End date for analysis (defaults to tomorrow's date)"
    ),
    start_date: str = Depends(
        lambda end_date=Query(default=date.today().strftime("%Y-%m-%d")): default_start_date(end_date)
    )
):
    """Analyze water uptake using Sentinel-1 ?VH or Sentinel-2 NDMI (choose latest)"""
 
    try:
        _, plot_data = _resolve_plot_or_refresh(plot_name)
        geometry = plot_data["geometry"]
 
        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)
 
        # --- Sentinel-2 NDMI ---
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .map(lambda img: img.clip(geometry))
            .map(lambda img: img.addBands(
                img.normalizedDifference(['B8A', 'B11']).rename('NDMI')
            ))
            .select(['NDMI'])
            .sort("system:time_start", False)
        )
        s2_count = s2_collection.size().getInfo()
 
        latest_s2_date = None
        if s2_count > 0:
            latest_s2_image = ee.Image(s2_collection.first())
            latest_s2_date = ee.Date(latest_s2_image.get("system:time_start"))
 
        # --- Sentinel-1 ?VH ---
        s1_collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(analysis_start, analysis_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VH"])
            .map(lambda img: img.clip(geometry))
            .sort("system:time_start", False)
        )
        s1_count = s1_collection.size().getInfo()
        latest_s1_date = None
        if s1_count >= 2:
            latest_image = ee.Image(s1_collection.toList(2).get(0))
            previous_image = ee.Image(s1_collection.toList(2).get(1))
            latest_s1_date = ee.Date(latest_image.get("system:time_start"))
 
        # --- Decide which dataset to use ---
        sensor = None
        use_s2 = False
        use_s1 = False
 
        if latest_s2_date and latest_s1_date:
            use_s2 = latest_s2_date.millis().getInfo() >= latest_s1_date.millis().getInfo()
            use_s1 = not use_s2
        elif latest_s2_date:
            use_s2 = True
        elif latest_s1_date:
            use_s1 = True
        else:
            raise HTTPException(status_code=404, detail="No Sentinel-1 or Sentinel-2 images in the selected period")
 
        if use_s2:
            sensor = "s2"
            ndmi_image = s2_collection.median().clip(geometry)
 
            # NDMI classification
            deficient = ndmi_image.lt(-0.21)
            less = ndmi_image.gte(-0.21).And(ndmi_image.lt(-0.031))
            adequat = ndmi_image.gte(-0.031).And(ndmi_image.lt(0.142))
            excellent = ndmi_image.gte(0.14).And(ndmi_image.lt(0.22))
            excess = ndmi_image.gte(0.22)
 
        if use_s1:
            sensor = "s1"
            delta_vh = latest_image.subtract(previous_image).rename("deltaVH").clip(geometry)
 
            # ?VH classification
            excess = delta_vh.gte(6)
            excellent = delta_vh.gte(4.0).And(delta_vh.lt(6))
            adequat = delta_vh.gt(-0.1).And(delta_vh.lt(4))
            less = delta_vh.gte(-3.13).And(delta_vh.lte(-0.1))
            deficient = delta_vh.lt(-3.13)
 
        # --- Combined classification & visualization ---
        combined_class = (
            ee.Image(0)
            .where(deficient, 1)
            .where(less, 2)
            .where(adequat, 3)
            .where(excellent, 4)
            .where(excess, 5)
            .clip(geometry)
        )
 
        smoothed_class = combined_class.focal_mean(radius=7, units="meters")
        vis_params = {
            "min": 0,
            "max": 5,
            "palette": [
                "#EBFF34",  # Very deficient
                "#CC8213AF",  # Low uptake
                "#1348E88E",  # Adequat
                "#2E199ABD",  # Adequate
                "#0602178F",  # Excess
            ],}
        smoothed_vis = smoothed_class.visualize(**vis_params).clip(geometry)
        tile_url = smoothed_vis.getMapId()["tile_fetcher"].url_format
 
        # --- Pixel analysis ---
        count_image = ee.Image.constant(1)
 
        def get_pixel_count(mask):
            return count_image.updateMask(mask).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=10,
                bestEffort=True
            ).get("constant").getInfo() or 0
 
        def mask_to_coords(mask):
            sampled = mask.selfMask().addBands(ee.Image.pixelLonLat()).sample(
                region=geometry, scale=10, geometries=True, tileScale=4
            ).getInfo()
            coords = [f["geometry"]["coordinates"] for f in sampled.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]
 
        deficient_count = get_pixel_count(combined_class.eq(1))
        less_count = get_pixel_count(combined_class.eq(2))
        adequat_count = get_pixel_count(combined_class.eq(3))
        excellent_count = get_pixel_count(combined_class.eq(4))
        excess_count = get_pixel_count(combined_class.eq(5))
        total_pixel_count = get_pixel_count(count_image)
 
        deficient_coords = mask_to_coords(combined_class.eq(1))
        less_coords = mask_to_coords(combined_class.eq(2))
        adequat_coords = mask_to_coords(combined_class.eq(3))
        excellent_coords = mask_to_coords(combined_class.eq(4))
        excess_coords = mask_to_coords(combined_class.eq(5))
 
        # --- Build feature ---
        feature = {
            "type": "Feature",
            "geometry": {
                "type": plot_data["geom_type"],
                "coordinates": plot_data["original_coords"],
            },
            "properties": {
                "plot_name": plot_name,
                "tile_url": tile_url,
                "sensor": sensor,  # added sensor info
                "image_count_in_range": s2_count if use_s2 else s1_count,
                "analysis_dates": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "latest_image_date": latest_s2_date.format("YYYY-MM-dd").getInfo() if use_s2 else
                                         ee.Date(latest_image.get("system:time_start")).format("YYYY-MM-dd").getInfo(),
                    "previous_image_date": None if use_s2 else
                                           ee.Date(previous_image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
                },
                "last_updated": datetime.now().isoformat(),
            },
        }
 
        # --- Return final response ---
        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {
                "total_pixel_count": total_pixel_count,
                "deficient_pixel_count": deficient_count,
                "deficient_pixel_percentage": (deficient_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "deficient_pixel_coordinates": deficient_coords,
                "less_pixel_count": less_count,
                "less_pixel_percentage": (less_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "less_pixel_coordinates": less_coords,
                "adequat_pixel_count": adequat_count,
                "adequat_pixel_percentage": (adequat_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "adequat_pixel_coordinates": adequat_coords,
                "excellent_pixel_count": excellent_count,
                "excellent_pixel_percentage": (excellent_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "excellent_pixel_coordinates": excellent_coords,
                "excess_pixel_count": excess_count,
                "excess_pixel_percentage": (excess_count / total_pixel_count) * 100 if total_pixel_count else 0,
                "excess_pixel_coordinates": excess_coords,
                "analysis_start_date": start_date,
                "analysis_end_date": end_date,
            }
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Water uptake analysis failed: {str(e)}")
 
    

@app.post("/SoilMoisture", response_model=Dict[str, Any])
async def analyze_plot_combined(
    plot_name: str,
    end_date: str = Query(
        default_factory=lambda: (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
        description="End date for analysis (defaults to tomorrow's date)"
    ),
    start_date: str = Depends(lambda end_date=Query(default=date.today().strftime('%Y-%m-%d')): default_start_date(end_date)),
):
    """
    Analyze plot using the most recent available satellite data (Sentinel-1 or Sentinel-2).
    Priority: Latest image date, with S2 preferred if both available on same date.
    """
   
    try:
        _, plot_data = _resolve_plot_or_refresh(plot_name)
        geometry = plot_data["geometry"]
 
        # === Check Sentinel-1 Availability ===
        s1_collection = (
            filter_s1(
                ee.ImageCollection('COPERNICUS/S1_GRD'),
                start_date, end_date, geometry
            )
            .map(addIndices)
        )
        s1_size = s1_collection.size().getInfo()
        s1_latest_date = None
        if s1_size > 0:
            s1_sorted = s1_collection.sort("system:time_start", False)
            s1_latest_img = ee.Image(s1_sorted.first())
            s1_latest_date = ee.Date(s1_latest_img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
 
        # === Check Sentinel-2 Availability ===
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(lambda img: img.clip(geometry))
        )
        s2_size = s2_collection.size().getInfo()
        s2_latest_date = None
        if s2_size > 0:
            s2_sorted = s2_collection.sort("system:time_start", False)
            s2_latest_img = ee.Image(s2_sorted.first())
            s2_latest_date = ee.Date(s2_latest_img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
 
        # === Decide Which Sensor to Use ===
        if s1_size == 0 and s2_size == 0:
            raise HTTPException(status_code=404, detail="No satellite images available in given date range")
       
        # Determine which sensor to use
        use_sentinel2 = False
        sensor_used = None
        latest_date = None
       
        if s1_latest_date and s2_latest_date:
            # Both available - compare dates
            if s2_latest_date >= s1_latest_date:
                use_sentinel2 = True
                sensor_used = "Sentinel-2"
                latest_date = s2_latest_date
            else:
                use_sentinel2 = False
                sensor_used = "Sentinel-1"
                latest_date = s1_latest_date
        elif s2_latest_date:
            use_sentinel2 = True
            sensor_used = "Sentinel-2"
            latest_date = s2_latest_date
        else:
            use_sentinel2 = False
            sensor_used = "Sentinel-1"
            latest_date = s1_latest_date
 
        # === Process Based on Selected Sensor ===
        if use_sentinel2:
            # === SENTINEL-2 NDWI Processing ===
            s2_composite = s2_collection.median().clip(geometry)
            ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
           
            classified = (
                ndwi.where(ndwi.lte(-0.4), 1)
                .where(ndwi.gt(-0.4).And(ndwi.lte(-0.3)), 2)
                .where(ndwi.gt(-0.3).And(ndwi.lte(0)), 3)
                .where(ndwi.gt(0).And(ndwi.lte(0.2)), 4)
                .where(ndwi.gt(0.2), 5)
            ).clip(geometry)
           
            image_count = s2_size
           
        else:
            # === SENTINEL-1 VV Processing ===
            vv_composite = safe_median(s1_collection.select(['VV'])).clip(geometry)
            vv_band = vv_composite.select('VV')
           
            classified = (
                vv_band.where(vv_band.gt(-6), 5)
                .where(vv_band.gt(-8).And(vv_band.lte(-6)), 4)
                .where(vv_band.gt(-10).And(vv_band.lte(-8)), 3)
                .where(vv_band.gt(-12).And(vv_band.lte(-10)), 2)
                .where(vv_band.lte(-12), 1)
            ).clip(geometry)
           
            image_count = s1_size
 
        # === Common Classification Labels & Visualization ===
        labels = {
            1: "less",
            2: "adequate",
            3: "excellent",
            4: "excess",
            5: "shallow water",
        }
 
        palette = [
            "#2FC0D3",  # less
            "#4365D4",  # adequate
            "#473CDF",  # excellent
            "#2116BF",  # excess
            "#000475",  # shallow water
        ]
       
        vis_params = {"min": 1, "max": 5, "palette": palette}
 
        # === Smoothing & Visualization ===
        smoothed = classified.focal_mean(radius=20, units="meters")
        visual = smoothed.visualize(**vis_params).clip(geometry)
        tile_url = visual.getMapId()["tile_fetcher"].url_format
 
        # === Pixel Counting ===
        count_image = ee.Image.constant(1)
        total_pixel_count = (
            count_image.reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
            .get("constant")
            .getInfo()
        )
 
        def get_pixel_count(mask):
            return (
                count_image.updateMask(mask)
                .reduceRegion(ee.Reducer.count(), geometry, 10, bestEffort=True)
                .get("constant")
            )
 
        def mask_to_coords(mask, geom):
            points = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=geom, scale=10, geometries=True, tileScale=4)
                .getInfo()
            )
            coords = [f["geometry"]["coordinates"] for f in points.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]
 
        # === Pixel Summary ===
        pixel_summary = {
            "total_pixel_count": total_pixel_count,
            "start_date": start_date,
            "end_date": end_date,
            "latest_image_date": latest_date,
            "sensor_used": sensor_used,
            "s1_available": s1_size > 0,
            "s2_available": s2_size > 0,
            "s1_latest_date": s1_latest_date,
            "s2_latest_date": s2_latest_date,
        }
 
        for class_id in range(1, 6):
            label = labels[class_id]
            mask = classified.eq(class_id)
            count = get_pixel_count(mask).getInfo() or 0
            percent = (count / total_pixel_count) * 100 if total_pixel_count else 0
            coords = mask_to_coords(mask, geometry)
 
            key_base = label.lower().replace(" ", "_")
            pixel_summary[f"{key_base}_pixel_count"] = count
            pixel_summary[f"{key_base}_pixel_percentage"] = round(percent, 2)
            pixel_summary[f"{key_base}_pixel_coordinates"] = coords
 
        # === GeoJSON Feature ===
        feature = {
            "type": "Feature",
            "geometry": {
                "type": plot_data["geom_type"],
                "coordinates": plot_data["original_coords"],
            },
            "properties": {
                "plot_name": plot_name,
                "start_date": start_date,
                "end_date": end_date,
                "sensor_used": sensor_used,
                "latest_image_date": latest_date,
                "image_count": image_count,
                "tile_url": tile_url,
                "last_updated": datetime.now().isoformat(),
            },
        }
 
        # === Final Response ===
        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": pixel_summary
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined moisture analysis failed: {str(e)}")
 

# ============================================================
# COMBINED PEST DETECTION (GRAPES + SUGARCANE)
# ============================================================

def run_pest_detection_analysis_by_plot(
    plot_name: str,
    plot_data: dict,
    start_date: str,
    end_date: str,
):
    try:

        if not isinstance(plot_data, dict):
            return None

        geometry = plot_data.get("geometry")
        geom_type = plot_data.get("geom_type")
        original_coords = plot_data.get("original_coords")

        if geometry is None or geom_type is None or original_coords is None:
            return None

        if not isinstance(geometry, ee.Geometry):
            geometry = ee.Geometry(geometry)

        props = plot_data.get("properties") or {}

        crop_type = (
            props.get("crop_type_name")
            or props.get("crop_type")
            or props.get("crop")
            or ""
        )

        crop_type = str(crop_type).lower().strip()

        if not crop_type:
            return None

        # ==============================
        # DATE
        # ==============================

        analysis_start = ee.Date(start_date)
        analysis_end = ee.Date(end_date)
        baseline_year = analysis_end.get("year").subtract(1)

        # ========================================================
        # GRAPES
        # ========================================================
        if "grape" in crop_type:

            baseline_start = ee.Date.fromYMD(baseline_year, 4, 1)
            baseline_end = ee.Date.fromYMD(baseline_year, 6, 30)

            def db_to_linear(img):
                return ee.Image(10).pow(img.divide(10)).add(1e-6)

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
                .select(["VV", "VH"])
            )

            image_count = int(s1.size().getInfo() or 0)
            if image_count == 0:
                return None

            image_dates = (
                s1.aggregate_array("system:time_start")
                .map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
                .getInfo()
            )

            cur = s1.median().clip(geometry)

            base = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(baseline_start, baseline_end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .select(["VV", "VH"])
                .median()
                .clip(geometry)
            )

            vv_cur = db_to_linear(cur.select("VV"))
            vh_cur = db_to_linear(cur.select("VH"))
            vv_base = db_to_linear(base.select("VV"))
            vh_base = db_to_linear(base.select("VH"))

            rvi_cur = vh_cur.multiply(4).divide(vv_cur.add(vh_cur))
            rvi_base = vh_base.multiply(4).divide(vv_base.add(vh_base))

            vhvv_cur = vh_cur.divide(vv_cur)
            vhvv_base = vh_base.divide(vv_base)

            fungi_mask = (
                rvi_cur.subtract(rvi_base).lt(-0.55)
                .And(vhvv_cur.subtract(vhvv_base).lt(-0.15))
                .And(vh_cur.subtract(vh_base).lt(0))
            )

            sucking_mask = (
                rvi_base.subtract(rvi_cur).gt(0.20)
                .And(vhvv_base.subtract(vhvv_cur).gt(0.10))
                .And(vh_cur.subtract(vh_base).lt(-0.01))
            )

            chewing_mask = (
                vh_base.subtract(vh_cur).gt(0.05)
                .And(rvi_base.subtract(rvi_cur).gt(0.06))
                .And(vhvv_base.subtract(vhvv_cur).gt(0.10))
            )

            wilt_mask = ee.Image.constant(0)
            soilborne_mask = chewing_mask

        # ========================================================
        # SUGARCANE
        # ========================================================
        elif "sugar" in crop_type:

            baseline_start = ee.Date.fromYMD(baseline_year, 6, 1)
            baseline_end = ee.Date.fromYMD(baseline_year, 10, 30)

            s1 = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
            )

            image_count = int(s1.size().getInfo() or 0)
            if image_count == 0:
                return None

            image_dates = (
                s1.aggregate_array("system:time_start")
                .map(lambda d: ee.Date(d).format("YYYY-MM-dd"))
                .getInfo()
            )

            img = s1.median().clip(geometry)

            vv = img.select("VV")
            vh = img.select("VH")

            ratio = vv.divide(vh.add(1e-6))

            chewing_mask = ratio.lte(0.01)
            fungi_mask = chewing_mask
            sucking_mask = ratio.gte(1)
            wilt_mask = ratio.gte(1)
            soilborne_mask = chewing_mask

        else:
            return None

        # =====================================================
        # COMBINED VISUALIZATION WITH PALETTE
        # =====================================================

        combined_class = (
            ee.Image(0)
            .where(chewing_mask, 1)
            .where(fungi_mask, 2)
            .where(sucking_mask, 3)
            .where(wilt_mask, 4)
            .where(soilborne_mask, 5)
            .clip(geometry)
        )

        combined_smooth = combined_class.focal_mean(radius=10, units="meters")

        combined_vis_params = {
            "min": 0,
            "max": 5,
            "palette": [
                "#FF0000",
                "#000000",
                "#000000",
                "#000000",
                "#000000",
                "#000000",
            ],
        }

        combined_smooth_vis = combined_smooth.visualize(**combined_vis_params).clip(geometry)
        tile_url = combined_smooth_vis.getMapId()["tile_fetcher"].url_format

        # =====================================================
        # PIXEL COORDINATES
        # =====================================================

        def mask_to_coords(mask, geom):
            points = (
                mask.selfMask()
                .addBands(ee.Image.pixelLonLat())
                .sample(region=geom, scale=10, geometries=True, tileScale=4)
                .getInfo()
            )

            coords = [f["geometry"]["coordinates"] for f in points.get("features", [])]
            return [list(x) for x in {tuple(c) for c in coords}]

        chewing_coords = mask_to_coords(chewing_mask, geometry)
        fungi_coords = mask_to_coords(fungi_mask, geometry)
        sucking_coords = mask_to_coords(sucking_mask, geometry)
        wilt_coords = mask_to_coords(wilt_mask, geometry)
        soilborne_coords = mask_to_coords(soilborne_mask, geometry)

        # =====================================================
        # PIXEL COUNT
        # =====================================================

        count_image = ee.Image.constant(1)

        def get_pixel_count(mask):
            value = count_image.updateMask(mask).reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant")
            return int(value.getInfo() or 0)

        chewing_pixel_count = get_pixel_count(chewing_mask)
        fungi_pixel_count = get_pixel_count(fungi_mask)
        sucking_pixel_count = get_pixel_count(sucking_mask)
        wilt_pixel_count = get_pixel_count(wilt_mask)
        soilborne_pixel_count = get_pixel_count(soilborne_mask)

        total_pixel_count = int(
            count_image.reduceRegion(
                ee.Reducer.count(), geometry, 10, bestEffort=True
            ).get("constant").getInfo()
        )

        healthy_pixel_count = total_pixel_count - (
            chewing_pixel_count
            + fungi_pixel_count
            + sucking_pixel_count
            + wilt_pixel_count
            + soilborne_pixel_count
        )

        def pct(v):
            return (v / total_pixel_count * 100) if total_pixel_count else 0

        # =====================================================
        # DATE STRINGS
        # =====================================================

        baseline_start_str = baseline_start.format("YYYY-MM-dd").getInfo()
        baseline_end_str = baseline_end.format("YYYY-MM-dd").getInfo()
        analysis_start_str = analysis_start.format("YYYY-MM-dd").getInfo()
        analysis_end_str = analysis_end.format("YYYY-MM-dd").getInfo()

        # =====================================================
        # FEATURE
        # =====================================================

        feature = {
            "type": "Feature",
            "geometry": {
                "type": geom_type,
                "coordinates": original_coords,
            },
            "properties": {
                "plot_name": plot_name,
                "crop_type": crop_type,
                "start_date": start_date,
                "end_date": end_date,
                "image_count": image_count,
                "image_dates": image_dates,
                "tile_url": tile_url,
                "last_updated": datetime.utcnow().isoformat(),
            },
        }

        return {
            "type": "FeatureCollection",
            "features": [feature],
            "pixel_summary": {

                "total_pixel_count": total_pixel_count,
                "healthy_pixel_count": healthy_pixel_count,

                "chewing_affected_pixel_count": chewing_pixel_count,
                "chewing_affected_pixel_percentage": pct(chewing_pixel_count),
                "chewing_affected_pixel_coordinates": chewing_coords,

                "fungi_affected_pixel_count": fungi_pixel_count,
                "fungi_affected_pixel_percentage": pct(fungi_pixel_count),
                "fungi_affected_pixel_coordinates": fungi_coords,

                "sucking_affected_pixel_count": sucking_pixel_count,
                "sucking_affected_pixel_percentage": pct(sucking_pixel_count),
                "sucking_affected_pixel_coordinates": sucking_coords,

                "wilt_affected_pixel_count": wilt_pixel_count,
                "wilt_affected_pixel_percentage": pct(wilt_pixel_count),
                "wilt_affected_pixel_coordinates": wilt_coords,

                "SoilBorn_pixel_count": soilborne_pixel_count,
                "SoilBorn_affected_pixel_percentage": pct(soilborne_pixel_count),
                "SoilBorn_affected_pixel_coordinates": soilborne_coords,

                "baseline_start_date": baseline_start_str,
                "baseline_end_date": baseline_end_str,
                "analysis_start_date": analysis_start_str,
                "analysis_end_date": analysis_end_str,
            },
        }

    except Exception as e:
        print(f"Pest detection failed: {e}")
        return None


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/pest-detection", response_model=Dict[str, Any])
async def pest_detection_by_crop(
    plot_name: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):

    today = datetime.utcnow().date()

    end_date = end_date or today.strftime("%Y-%m-%d")
    start_date = start_date or (
        today - timedelta(days=15)
    ).strftime("%Y-%m-%d")

    resolved_name, resolved_plot = _resolve_plot_or_refresh(plot_name)

    result = run_pest_detection_analysis_by_plot(
        plot_name=resolved_name,
        plot_data=resolved_plot,
        start_date=start_date,
        end_date=end_date,
    )

    if result is None:
        raise HTTPException(status_code=404, detail="No satellite data available")

    return result
 


@app.post("/grapes/canopy-vigour1", response_model=Dict[str, Any])
def grapes_canopy_vigour(
    plot_name: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):

    _, plot_data = _resolve_plot_or_refresh(plot_name)

    # -------------------- DEFAULT DATES --------------------
    today = datetime.utcnow().date()
    end_date = end_date or today.strftime("%Y-%m-%d")
    start_date = start_date or (today - timedelta(days=15)).strftime("%Y-%m-%d")

    geometry = ee.Geometry(plot_data["geometry"])
    geom_type = plot_data["geom_type"]
    original_coords = plot_data["original_coords"]

    # ------------------------------------------------------------------
    # Sentinel-2
    # ------------------------------------------------------------------
    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .select(["B4", "B8"])
    )

    if s2_col.size().getInfo() == 0:
        raise HTTPException(
            status_code=404,
            detail="No Sentinel-2 images for given date range and plot"
        )

    image_count = int(s2_col.size().getInfo())

    s2 = s2_col.median().clip(geometry)

    # ------------------------------------------------------------------
    # NDVI (UNCHANGED)
    # ------------------------------------------------------------------
    ndvi = (
        s2.select("B8")
        .subtract(s2.select("B4"))
        .divide(s2.select("B8").add(s2.select("B4")))
        .rename("NDVI")
    )

    ndvi_mean = (
        ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10,
            bestEffort=True
        )
        .get("NDVI")
    )

    ndvi_mean = ee.Number(ndvi_mean).getInfo() if ndvi_mean else None

    # ------------------------------------------------------------------
    # Vigour Classification (UNCHANGED)
    # ------------------------------------------------------------------
    vigour_class_img = (
        ee.Image(0)
        .where(ndvi.lt(0.3), 1)
        .where(ndvi.gte(0.3).And(ndvi.lt(0.5)), 2)
        .where(ndvi.gte(0.5).And(ndvi.lt(0.7)), 3)
        .where(ndvi.gte(0.7), 4)
        .clip(geometry)
        .toByte()
        .reproject("EPSG:3857", None, 10)
    )

    vigour_vis = {
        "min": 1,
        "max": 4,
        "palette": ["#FF0000", "#FFFF00", "#90EE90", "#006400"],
    }

    tile_url = vigour_class_img.getMapId(
        vigour_vis)["tile_fetcher"].url_format


    smoothed_class = vigour_class_img.focal_mean(radius=5, units="meters")

    smoothed_vis = smoothed_class.visualize(**vigour_vis).clip(geometry)

    tile_url = smoothed_vis.getMapId()["tile_fetcher"].url_format


    # =====================================================
    # ✅ PIXEL COORDINATES EXTRACTION
    # =====================================================

    pixel_lonlat = ee.Image.pixelLonLat()

    def extract_coords(mask):
        try:
            coords = (
                pixel_lonlat.updateMask(mask)
                .sample(
                    region=geometry,
                    scale=10,
                    geometries=True,
                    numPixels=5000
                )
                .geometry()
                .coordinates()
                .getInfo()
            )
            return coords
        except:
            return []

    poor_coords = extract_coords(vigour_class_img.eq(1))
    moderate_coords = extract_coords(vigour_class_img.eq(2))
    good_coords = extract_coords(vigour_class_img.eq(3))
    excellent_coords = extract_coords(vigour_class_img.eq(4))

    # =====================================================
    # PIXEL COUNTS
    # =====================================================

    one = ee.Image.constant(1)

    def safe_count(mask):
        try:
            value = one.updateMask(mask).reduceRegion(
                ee.Reducer.count(),
                geometry,
                10,
                bestEffort=True
            ).get("constant")
            return int(value.getInfo() or 0)
        except:
            return 0

    total_pixel_count = safe_count(one)

    poor_count = safe_count(vigour_class_img.eq(1))
    moderate_count = safe_count(vigour_class_img.eq(2))
    good_count = safe_count(vigour_class_img.eq(3))
    excellent_count = safe_count(vigour_class_img.eq(4))

    # =====================================================
    # FEATURE RESPONSE (SIMILAR FORMAT)
    # =====================================================

    feature = {
        "type": "Feature",
        "geometry": {
            "type": geom_type,
            "coordinates": original_coords,
        },
        "properties": {
            "plot_name": plot_name,
            "crop_type": "grapes",
            "start_date": start_date,
            "end_date": end_date,
            "image_count": image_count,
            "ndvi_mean": round(ndvi_mean, 3) if ndvi_mean else None,
            "tile_url": tile_url,
            "last_updated": datetime.utcnow().isoformat(),
        },
    }

    return {
        "type": "FeatureCollection",
        "features": [feature],
        "pixel_summary": {
            "total_pixel_count": total_pixel_count,

            "poor_vigour_pixel_count": poor_count,
            "poor_vigour_coordinates": poor_coords,

            "moderate_vigour_pixel_count": moderate_count,
            "moderate_vigour_coordinates": moderate_coords,

            "good_vigour_pixel_count": good_count,
            "good_vigour_coordinates": good_coords,

            "excellent_vigour_pixel_count": excellent_count,
            "excellent_vigour_coordinates": excellent_coords,
        },
    }


@app.post("/grapes/brix-grid-values", response_model=Dict[str, Any])
async def grapes_brix_grid_values(
    plot_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
):

    _, plot = _resolve_plot_or_refresh(plot_name)

    # ---------------- DEFAULT DATES ----------------
    today = datetime.utcnow().date()
    end_date = end_date or today.strftime("%Y-%m-%d")
    start_date = start_date or (
        today - timedelta(days=15)
    ).strftime("%Y-%m-%d")

    geometry = ee.Geometry(plot["geometry"])

    # ---------------- CROP TYPE ----------------
    props = plot.get("properties", {})
    crop_type = (
        props.get("crop_type_name")
        or props.get("crop_type")
        or props.get("crop")
        or "grapes"
    )

    # ---------------- IMAGE ----------------
    image = compute_brix_image(start_date, end_date, geometry)
    brix_img = image.select("Brix")

    samples = (
        brix_img.addBands(ee.Image.pixelLonLat())
        .sample(
            region=geometry,
            scale=10,
            geometries=True,
            tileScale=4,
        )
        .getInfo()
    )

    grid_values = []

    for f in samples.get("features", []):
        coords = f["geometry"]["coordinates"]
        val = f["properties"].get("Brix")

        grid_values.append({
            "latitude": coords[1],
            "longitude": coords[0],
            "brix": round(val, 2) if val else 0,
        })

    # ---------------- FEATURE ----------------
    feature = {
        "type": "Feature",
        "geometry": {
            "type": plot["geom_type"],
            "coordinates": plot["original_coords"],
        },
        "properties": {
            "plot_name": plot_name,
            "crop": crop_type,
            "date": end_date,
        },
    }

    return {
        "type": "FeatureCollection",
        "features": [feature],
        "grid_values": grid_values,
        "total_points": len(grid_values),
        "last_updated": datetime.utcnow().isoformat(),
    }

 


@app.get("/visualization-params")
async def get_visualization_params():
    """Get visualization parameters for indices"""
    return {"visualization_parameters": indexVisParams}
 
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "plot_count": len(plot_dict),
        "keepalive": keepalive_manager.status() if keepalive_manager else None
    }
 
@app.get("/plots/{plot_name}/tiles")
async def get_plot_tiles(
    plot_name: str,
    end_date: str = Query(default=date.today().strftime('%Y-%m-%d'), description="End date in YYYY-MM-DD format"),
    start_date: str = Depends(lambda end_date=Query(default=date.today().strftime('%Y-%m-%d')): default_start_date(end_date)),
):
    """Get tile URLs for a plot"""
    _, plot = _resolve_plot_or_refresh(plot_name)
   
    try:
        aoi = plot['geometry']
        print(f" Generating tiles for {plot_name}")
       
        # Get image collection
        coll = filter_s1(
            ee.ImageCollection('COPERNICUS/S1_GRD'),
            start_date, end_date, aoi
        ).map(addIndices)
       
        coll_size = coll.size().getInfo()
        if coll_size > 0:
            img = safe_median(coll.select(['VV', 'VH', 'VV_VH_ratio', 'SWI', 'RVI'])).clip(aoi)
        else:
            alt_img = get_alternative_image(aoi, end_date)
            if alt_img is not None:
                img = alt_img.select(['VV', 'VH', 'VV_VH_ratio', 'SWI', 'RVI']).clip(aoi)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="No Sentinel-1 images available for plot in the specified date range or extended range"
                )
       
        # Generate tile URLs
        tile_urls = {}
        for index_name in ['VV', 'VH', 'VV_VH_ratio', 'SWI', 'RVI']:
            smoothed = img.select(index_name).focal_mean(radius=30, units='meters').clip(aoi)
            tile_url = get_tile_url(smoothed, indexVisParams[index_name], index_name)
            tile_urls[f"{index_name}_tile_url"] = tile_url
       
        # Prepare response
        response_data = {
            "plot_name": plot_name,
            "start_date": start_date,
            "end_date": end_date,
            "tile_urls": tile_urls,
            "generated_at": datetime.now().isoformat()
        }
       
        return response_data
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile generation failed: {str(e)}")
 
def get_custom_cmap(index_name):
    """Get custom colormap for index"""
    if index_name in indexVisParams:
        return mcolors.LinearSegmentedColormap.from_list(
            f"{index_name}_cmap",
            indexVisParams[index_name]['palette']
        )
    return None
 
def get_vis_params(index_name):
    """Get visualization parameters for index"""
    return indexVisParams.get(index_name, {})




@app.get("/satellite-updates/{plot_name}")
async def check_satellite_updates(
    plot_name: str,
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format")
):
    """Check for satellite updates for a specific plot"""
    _, plot = _resolve_plot_or_refresh(plot_name)
   
    try:
        aoi = plot['geometry']
       
        # Get current collection info
        coll = filter_s1(
            ee.ImageCollection('COPERNICUS/S1_GRD'),
            start_date, end_date, aoi
        ).map(addIndices)
       
        current_satellite_update = get_latest_satellite_update(coll)
        image_count = coll.size().getInfo()
       
        return {
            "plot_name": plot_name,
            "date_range": f"{start_date} to {end_date}",
            "current_satellite_update": current_satellite_update,
            "image_count": image_count,
            "checked_at": datetime.now().isoformat()
        }
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update check failed: {str(e)}") 



plot_service = PlotSyncService()
@app.get("/distance")
def calculate_distances(
    lat: float = Query(..., description="Factory latitude"),
    lon: float = Query(..., description="Factory longitude")
):
    geojson_data = plot_service.get_plots_dict()  # call instance method
   
    factory_coords = (lat, lon)
    factory_point = Point(lon, lat)
 
    results = []
    total_distance = 0.0
    seen_names = set()
 
    for feature in geojson_data.get('features', []):
        farmer_name = feature['properties'].get('Name') or 'unknown'
        if farmer_name in seen_names:
            continue  # skip duplicates by name
        seen_names.add(farmer_name)
 
        plot_geom = shape(feature['geometry'])
 
        area_ha = calculate_area_hectares(plot_geom)
        area_acres = round(area_ha * 2.471, 2) if area_ha else None
 
        if isinstance(plot_geom, Polygon):
            boundary = plot_geom.exterior
            closest_point = boundary.interpolate(boundary.project(factory_point))
        else:
            closest_point = plot_geom.centroid
 
        closest_coords = (closest_point.y, closest_point.x)
        distance_km = geodesic(factory_coords, closest_coords).km
        total_distance += distance_km
 
        results.append({
            'name': farmer_name,
            'distance_km': round(distance_km, 2),
            'area_acres': area_acres
        })
 
    average_distance = round(total_distance / len(results), 2) if results else 0.0
 
    return {
        'all_plots': results,
        'average_distance_km': average_distance}
        

@app.post("/refresh-from-django")
async def refresh_from_django():
    """Manually refresh all plots from Django - useful after Django restart"""
    try:
        global plot_dict
        print("?? Manual refresh from Django requested...")
        # Force refresh from Django
        if keepalive_manager:
            plot_dict = await keepalive_manager.refresh_now(force=True)
        else:
            plot_dict = plot_sync_service.get_plots_dict(force_refresh=True)
        return {
            "status": "success", 
            "message": f"Successfully refreshed {len(plot_dict)} plots from Django",
            "plot_count": len(plot_dict),
            "plots_with_django_ids": len([p for p in plot_dict.values() if p.get('properties', {}).get('django_id')])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh from Django: {str(e)}")
        

if __name__ == "__main__":
    uvicorn.run("Admin:app", host="0.0.0.0", port=5000, reload=True)
