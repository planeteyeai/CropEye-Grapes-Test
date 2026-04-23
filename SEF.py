import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Any, Union
import ee
import ee.data
from cachetools import TTLCache
from datetime import date, timedelta, datetime
import uvicorn
import httpx
import math
import json
from contextlib import asynccontextmanager
from shared_services import PlotSyncService
from fastapi.middleware.cors import CORSMiddleware
import asyncio
# Initialize Earth Engine
raw = os.environ["EE_SERVICE_ACCOUNT_JSON"]
service_account = raw if isinstance(raw, str) else json.dumps(raw)
ee.Initialize(ee.ServiceAccountCredentials(None, key_data=service_account))

# Initialize plot sync service
plot_sync_service = PlotSyncService()
app=FastAPI(tittle="Admin API for SAR Index Mapping with Pest Detection")
# -------------------------------
# Load ROI/Plot Dictionary from Django API
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
plot_dict = plot_sync_service.get_plots_dict()
print(f"ðŸš€ events.py startup: Loaded {len(plot_dict)} plots from Django")


def _apply_plot_update(new_plots: Dict[str, Dict]) -> None:
    global plot_dict
    plot_dict = new_plots
    print(f"SEF.py background refresh: {len(plot_dict)} plots")

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

def find_plot_by_name(plot_name: str) -> Optional[Dict]:
    """Find plot by name with flexible matching"""
    # Direct lookup
    if plot_name in plot_dict:
        print(f"Found plot '{plot_name}' with direct lookup")
        return plot_dict[plot_name]
    
    # Try different variations for leading zeros
    variations = []
    
    # If plot_name contains underscore, try different combinations
    if '_' in plot_name:
        parts = plot_name.split('_')
        if len(parts) == 2:
            gat_part, plot_part = parts
            
            # Try original
            variations.append(plot_name)
            
            # Try with leading zeros (only for exact matches)
            try:
                if gat_part.startswith('0'):
                    variations.append(f"{int(gat_part)}_{plot_part}")
                if plot_part.startswith('0'):
                    variations.append(f"{gat_part}_{int(plot_part)}")
                if gat_part.startswith('0') and plot_part.startswith('0'):
                    variations.append(f"{int(gat_part)}_{int(plot_part)}")
            except ValueError:
                pass
    
    # Try variations
    for variation in variations:
        if variation in plot_dict:
            print(f"Found plot '{plot_name}' using variation '{variation}'")
            return plot_dict[variation]
    
    # Final attempt: instant refresh from Django and retry
    fresh = plot_sync_service.get_plots_dict(force_refresh=True)
    if fresh:
        _apply_plot_update(fresh)
        if plot_name in plot_dict:
            return plot_dict[plot_name]
        for variation in variations:
            if variation in plot_dict:
                return plot_dict[variation]

    print(f"Plot '{plot_name}' not found after refresh. Available plots: {list(plot_dict.keys())}")
    return None


def safe_compute(ee_obj, desc=""):
    """Safe Earth Engine compute replacement for .getInfo()"""
    try:
        return ee.data.computeValue(ee_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Earth Engine error ({desc}): {str(e)}")


# Pydantic models
class PlotInfo(BaseModel):
    name: str
    geometry_type: str
    area_hectares: Optional[float] = None
 
class HealthStats(BaseModel):
    mean: float
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
 
class PlotGeometry(BaseModel):
    type: str
    coordinates: List[List[List[float]]]
 
class HealthAnalysis(BaseModel):
    plot_name: str
    overall_health: float
    health_status: str
    statistics: HealthStats
    analysis_dates: Dict[str, str]
 
class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]
 
class GeoJSONProperties(BaseModel):
    name: str
    health_status: str
    overall_health: float
    statistics: HealthStats
 
class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties
 
class GeoJSONResponse(BaseModel):
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]
 
class DateRange(BaseModel):
    start_date: str
    end_date: str
 
    @validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        try:
            return datetime.strptime(v, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
 
    @validator('end_date')
    def validate_end_date(cls, v, values):
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end < start:
                raise ValueError('End date must be after start date')
        return v
 
 # Create cache: store up to 100 results, expire after 1 hour (3600 seconds)
analysis_cache = TTLCache(maxsize=700, ttl=7200)

def make_cache_key(plot_name: str, start_date: str, end_date: str) -> str:
    """Create a unique key for caching"""
    return f"{plot_name}_{start_date}_{end_date}"

class VegetationHealthAnalyzer:

    def preprocess_sentinel1(self, collection, roi):
        """Preprocess Sentinel-1 SAR data"""
        def apply_preprocessing(image):
            # Apply orbit file (already applied in GEE for most recent data)
            # Apply border noise removal
            image = image.updateMask(
                image.select('VV').gt(-25).And(image.select('VH').gt(-25))
            )
           
            # Apply speckle filter (Refined Lee)
            vv = image.select('VV')
            vh = image.select('VH')
           
            # Lee filter
            kernel = ee.Kernel.square(radius=1)
            vv_filtered = vv.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=kernel
            ).rename('VV')
           
            vh_filtered = vh.reduceNeighborhood(
                reducer=ee.Reducer.mean(),
                kernel=kernel
            ).rename('VH')
           
            return image.addBands(vv_filtered, overwrite=True).addBands(vh_filtered, overwrite=True)
       
        return collection.map(apply_preprocessing)
 
    def analyze_vegetation_health(self, plot_name: str, start_date: str, end_date: str):
        """Analyze vegetation health for selected plot using Sentinel-1 SAR data"""
        try:
            plot_feature = find_plot_by_name(plot_name)
            if not plot_feature:
                raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found")

            roi = ee.FeatureCollection([ee.Feature(plot_feature["geometry"])])
            geometry = plot_feature["geometry"]

           
            # Load Sentinel-1 SAR data
            s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterBounds(roi)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))  # Use descending pass for consistency
           
            # Check if we have any images
            count = safe_compute(s1.size(), desc="image count")
            if count == 0:
                # Try to find the nearest available date
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
               
                # Try different date ranges
                date_ranges = [
                    (start_dt - timedelta(days=30), end_dt),  # 30 days before
                    (start_dt, end_dt + timedelta(days=30)),  # 30 days after
                    (start_dt - timedelta(days=60), end_dt + timedelta(days=60)),  # 60 days before and after
                ]
               
                for new_start, new_end in date_ranges:
                    if new_start < datetime(2014, 10, 1):  # Sentinel-1 data starts from October 2014
                        new_start = datetime(2014, 10, 1)
                   
                    new_start_str = new_start.strftime('%Y-%m-%d')
                    new_end_str = new_end.strftime('%Y-%m-%d')
                   
                    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                          .filterBounds(roi)
                          .filterDate(new_start_str, new_end_str)
                          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                          .filter(ee.Filter.eq('instrumentMode', 'IW'))
                          .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')))
                   
                    count = safe_compute(s1.size(), desc="image count")
                    if count > 0:
                        start_date = new_start_str
                        end_date = new_end_str
                        break
               
                if count == 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No Sentinel-1 images found for the date range {start_date} to {end_date} or nearby dates. Please try a different date range."
                    )
           
            # Preprocess the collection
            s1_processed = self.preprocess_sentinel1(s1, roi)
           
            # Get the median image
            s1_median = s1_processed.median().clip(roi)
           
            # Check if the image has the required bands
            band_names = safe_compute(s1_median.bandNames(), desc="band names")
            required_bands = ['VV', 'VH']
            missing_bands = [band for band in required_bands if band not in band_names]
           
            if missing_bands:
                raise HTTPException(
                    status_code=500,
                    detail=f"Missing required bands: {', '.join(missing_bands)}"
                )
           
            # Calculate SAR-based vegetation indices
            vv = s1_median.select('VV')
            vh = s1_median.select('VH')
           
            # Convert to linear scale (from dB)
            vv_linear = ee.Image(10).pow(vv.divide(10))
            vh_linear = ee.Image(10).pow(vh.divide(10))
           
            # Calculate Radar Vegetation Index (RVI)
            # RVI = (8 * VH) / (VV + VH + 2 * sqrt(VV * VH))
            rvi = vh_linear.multiply(8).divide(
                vv_linear.add(vh_linear).add(
                    vv_linear.multiply(vh_linear).sqrt().multiply(2)
                )
            ).rename('RVI')
           
            # Calculate Cross-Polarization Ratio (VH/VV)
            cross_pol_ratio = vh.subtract(vv).rename('CrossPolRatio')
           
            # Calculate Normalized Difference Polarization Index (NDPI)
            ndpi = vv.subtract(vh).divide(vv.add(vh)).rename('NDPI')
           
            # Calculate Combined Vegetation Health Index
            # Normalize RVI to 0-1 scale and convert to percentage
            rvi_normalized = rvi.subtract(0.2).divide(0.8).clamp(0, 1)  # Typical RVI range: 0.2-1.0
           
            # Normalize Cross-pol ratio (higher values indicate more vegetation)
            cross_pol_normalized = cross_pol_ratio.add(25).divide(15).clamp(0, 1)  # Typical range: -25 to -10 dB
           
            # Combine indices for overall health (0-100%)
            overall_health = rvi_normalized.multiply(0.6).add(
                cross_pol_normalized.multiply(0.4)
            ).multiply(100).clamp(0, 100).rename('Overall_Health_Percent')
           
            # Get statistics
            stats = self.get_statistics(overall_health, roi)
           
            # Interpret health status
            health_status = self.interpret_health(stats['mean'])
           
            return {
                'plot_name': plot_name,
                'overall_health': stats['mean'],
                'health_status': health_status[0],
                'statistics': stats,
                # 'map_url': map_url,
                'analysis_dates': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                # 'geometry': geometry,
                # 'sar_metrics': sar_metrics
            }
           
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
 
    def get_statistics(self, overall_health, roi):
        """Get detailed statistics for the plot"""
        stats = overall_health.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                reducer2=ee.Reducer.mean(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ),
            geometry=roi.geometry(),
            scale=10,
            maxPixels=1e9
        )
       
        try:
            stats_dict = safe_compute(stats, desc="health stats")
            return {
                'mean': round(stats_dict.get('Overall_Health_Percent_mean', 0), 2),
                'min': round(stats_dict.get('Overall_Health_Percent_min', 0), 2),
                'max': round(stats_dict.get('Overall_Health_Percent_max', 0), 2),
                'std': round(stats_dict.get('Overall_Health_Percent_stdDev', 0), 2)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")
 
    def interpret_health(self, mean_percentage):
        """Interpret health status based on SAR-derived vegetation health"""
        if mean_percentage >= 75:
            return "Excellent", "darkgreen"
        elif mean_percentage >= 60:
            return "Good", "green"
        elif mean_percentage >= 45:
            return "Moderate", "orange"
        elif mean_percentage >= 30:
            return "Poor", "red"
        else:
            return "Very Poor", "darkred"
 
# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global plot_dict
        print("🔄 SEF.py: Initializing application and fetching plots from Django API...")

        # Fetch once at startup (force refresh)
        plot_dict = await asyncio.to_thread(
            plot_sync_service.get_plots_dict, True
        )

        print(f"✅ SEF.py startup: Loaded {len(plot_dict)} plots from Django")
        print("🚀 SEF.py: Application initialized successfully")

    except Exception as e:
        print(f"❌ Failed to initialize application: {e}")
        raise

    yield

    # Shutdown (no keepalive to stop anymore)
    print("🛑 Shutting down FastAPI application")

app = FastAPI(
    title="Sentinel-1 SAR Vegetation Health Analysis API",
    description="API for analyzing vegetation health using Sentinel-1 SAR data",
    version="1.0.0",
    lifespan=lifespan
)
 
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=512)
 
# Initialize analyzer
analyzer = VegetationHealthAnalyzer()
 
# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Sentinel-1 SAR Vegetation Health Analysis API",
        "version": "1.0.0",
        "description": "Analyzes vegetation health using SAR data from Sentinel-1"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "plot_count": len(plot_dict)
    }
 
 
@app.get("/analyze", response_model=HealthAnalysis)
async def analyze_plot(
    plot_name: str = Query(..., description="Name of the plot to analyze"),
    start_date: Optional[str] = Query(None, description="Start date in YYYY-MM-DD format (defaults to 30 days before current date)"),
    end_date: Optional[str] = Query(None, description="End date in YYYY-MM-DD format (defaults to current date)")
):
    """Analyze vegetation health for a plot using Sentinel-1 SAR data"""
    try:
        # Set default dates (SAR data needs longer time period for better temporal compositing)
        end = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
        start = (end - timedelta(days=30)) if start_date is None else datetime.strptime(start_date, '%Y-%m-%d')
       
        # Adjust dates if needed
        if end > datetime.now():
            end = datetime.now()
       
        if start > end:
            start = end - timedelta(days=30)
       
        if (end - start).days > 365:
            start = end - timedelta(days=365)
       
        # Ensure start date is not before 2014-10-01 (Sentinel-1 data availability)
        if start < datetime(2014, 10, 1):
            start = datetime(2014, 10, 1)

        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        # --- Check Cache ---
        cache_key = make_cache_key(plot_name, start_str, end_str)
        if cache_key in analysis_cache:
            return analysis_cache[cache_key]

        # --- Run analysis if not cached ---
        result = analyzer.analyze_vegetation_health(
            plot_name,
            start_str,
            end_str
        )

        # Save result in cache
        analysis_cache[cache_key] = result
        return result

       
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
 
# -------------------------------
# ET Computation Logic
# -------------------------------

 
async def fetch_et_from_openmeteo(lat: float, lon: float, start_date: str, end_date: str) -> float:
    """Fetch ET0 FAO Evapotranspiration from Open-Meteo API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "et0_fao_evapotranspiration",
        "start_date": start_date,
        "end_date": end_date
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
       
        et_values = data.get("daily", {}).get("et0_fao_evapotranspiration", [])
        if not et_values:
            return None
       
        # Calculate mean of all ET values in the date range
        valid_values = [v for v in et_values if v is not None]
        if not valid_values:
            return None
       
        return sum(valid_values) / len(valid_values)
   
    except Exception as e:
        print(f"Error fetching ET from Open-Meteo: {e}")
        return None
 

def get_ndvi(image):
    """Calculate NDVI from Sentinel-2 image"""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def compute_et(image):
    """Compute ET from NDVI"""
    et = image.select('NDVI').multiply(6.0).add(0.8).rename('ET')
    return et.set('system:time_start', image.get('system:time_start'))

async def fetch_hourly_et_from_openmeteo(lat: float, lon: float, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Fetch hourly ET0 FAO Evapotranspiration from Open-Meteo Historical Forecast API"""
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "et0_fao_evapotranspiration"
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
       
        hourly_data = data.get("hourly", {})
        times = hourly_data.get("time", [])
        et_values = hourly_data.get("et0_fao_evapotranspiration", [])
       
        if not times or not et_values:
            return []
       
        # Create list of hourly records
        hourly_records = []
        for time_str, et_value in zip(times, et_values):
            hourly_records.append({
                "time": time_str,
                "et0_fao_evapotranspiration": round(et_value, 2) if et_value is not None else None
            })
       
        return hourly_records
   
    except Exception as e:
        print(f"Error fetching hourly ET from Open-Meteo: {e}")
        return []
 
def calculate_et_statistics(geometry, start_date: str, end_date: str) -> Dict[str, any]:
    """Calculate ET statistics for a given geometry and date range"""
    try:
        # Create Sentinel-2 collection
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .map(get_ndvi)
            .select('NDVI')
            .map(lambda img: img.clip(geometry))
        )
        
        # Compute ET collection
        et_collection = s2_collection.map(compute_et)
        mean_et_image = et_collection.mean().clip(geometry)
        
        # Calculate ET statistics
        et_stats = mean_et_image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=geometry,
            scale=20,
            maxPixels=1e13
        ).getInfo()
        
        mean_et = mean_et_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=20,
            maxPixels=1e13
        ).getInfo()
        
        # Calculate area and total ET in liters
        roi_area = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=20,
            maxPixels=1e13
        ).get('area')
        
        et_liters = ee.Number(mean_et.get('ET', 0)).multiply(roi_area).getInfo()
        
        return {
            "ET_mean_mm_per_day": mean_et.get('ET'),
            "ET_total_liters_per_day": et_liters
        }
    
    except Exception as e:
        print(f"Error in ET calculation: {e}")
        return {
            "ET_mean_mm_per_day": None,
            "ET_total_liters_per_day": None
        }

# -------------------------------
# Endpoints
# -------------------------------

@app.get("/plots")
async def get_plots():
    """Get list of available plots"""
    return list(plot_dict.keys())


@app.post("/plots/{plot_name}/compute-et/")
async def compute_et_for_plot(
    plot_name: str,
    start_date: Optional[str] = Query(default=None),
    end_date: Optional[str] = Query(default=None)
):
    """Compute ET for a specific plot"""
    _, plot = _resolve_plot_or_refresh(plot_name)
   
    # Set both start_date and end_date to current date if not provided
    today = date.today().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = today
    if start_date is None:
        start_date = today
   
    try:
        geometry = plot['geometry']
        area_hectares = geometry.area().divide(10000).getInfo()
       
        # Get plot coordinates
        coords = plot['original_coords']
        geom_type = plot['geom_type']
       
        if geom_type == 'Polygon':
            lon, lat = coords[0][0]
        elif geom_type == 'MultiPolygon':
            lon, lat = coords[0][0][0]
        else:
            raise HTTPException(status_code=400, detail="Unsupported geometry type")
       
        # Fetch ET from Open-Meteo API
        et_mean = await fetch_et_from_openmeteo(lat, lon, start_date, end_date)
       
        if et_mean is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch ET data from Open-Meteo API"
            )
       
        # Fetch hourly ET records from Historical Forecast API
        hourly_records_et = await fetch_hourly_et_from_openmeteo(lat, lon, start_date, end_date)
       
        return JSONResponse({
            "plot_name": plot_name,
            "start_date": start_date,
            "end_date": end_date,
            "area_hectares": round(area_hectares, 2),
            "ET_mean_mm_per_day": round(et_mean, 2),
            "hourly_records_et": hourly_records_et
        })
   
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
   

# ============================
# horizontal Soil moisture
# ============================


# ============================
# ET calculation
# ============================
def get_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)
 
def compute_et(image):
    et = image.select('NDVI').multiply(6.0).add(0.8).rename('ET')
    return et.set('system:time_start', image.get('system:time_start'))
 
def calculate_et_statistics_soil(geometry, start_date: str, end_date: str):
    try:
        s2_collection = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .map(get_ndvi)
            .select('NDVI')
            .map(lambda img: img.clip(geometry))
        )
 
        et_collection = s2_collection.map(compute_et)
        mean_et_image = et_collection.mean().clip(geometry)
 
        mean_et = mean_et_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=20,
            maxPixels=1e13
        ).getInfo()

        roi_area = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=20,
            maxPixels=1e13
        ).get('area')
 
        et_value = mean_et.get('ET', 0)
        try:
            et_value = float(et_value)
            if math.isnan(et_value) or math.isinf(et_value):
                et_value = 0.0
        except (TypeError, ValueError):
            et_value = 0.0
 
        try:
            roi_area = float(roi_area)
            if math.isnan(roi_area) or math.isinf(roi_area):
                roi_area = 0.0
        except (TypeError, ValueError):
            roi_area = 0.0
  
        return et_value
        
 
    except Exception as e:
        print(f"Error in ET calculation: {e}")
        return 0.0


# ============================
# Rainfall fetch
# ============================
async def fetch_rainfall(lat: float, lon: float, start: date, end: date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "auto"
    }
    timeout = httpx.Timeout(connect=8.0, read=12.0, write=8.0, pool=8.0)
    retry_delays = [0.5, 1.0, 2.0]

    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                values = data.get("daily", {}).get("precipitation_sum", [])
                return values if isinstance(values, list) else []
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError, httpx.HTTPStatusError) as e:
            if attempt == len(retry_delays):
                print(f"Rainfall fetch failed after retries ({lat}, {lon}): {e}")
                return []
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"Unexpected rainfall fetch error ({lat}, {lon}): {e}")
            return []

# ============================
# Main endpoint
# ============================
@app.post("/soil-moisture/{plot_name}")
async def soil_moisture(plot_name: str):
    _, plot = _resolve_plot_or_refresh(plot_name)

    coords = plot['original_coords']
    geom_type = plot['geom_type']

    if geom_type == 'Polygon':
        lon, lat = coords[0][0]
    elif geom_type == 'MultiPolygon':
        lon, lat = coords[0][0][0]
    else:
        raise HTTPException(status_code=400, detail="Unsupported geometry type")

    today = date.today()
    start_date = today - timedelta(days=7)   # 7 days before today
    end_date = today

    # Fetch rainfall for (start_date - 1) .. (today - 1) to cover all previous days
    rainfall_values = await fetch_rainfall(lat, lon, start_date - timedelta(days=1), today - timedelta(days=1))

    # Fix yesterday's rainfall if missing
    if rainfall_values and rainfall_values[-1] is None:
        last_vals = [v for v in rainfall_values[-4:-1] if v is not None]
        if last_vals:
            rainfall_values[-1] = sum(last_vals) / len(last_vals)
        else:
            rainfall_values[-1] = 0.0

    # Rolling soil moisture calculation
    sm_prev = 80.0  # dummy initial value
    results = []
    for i in range(7):
        current_day = start_date + timedelta(days=i)
        prev_day_index = i  # rainfall index matches because we started from start_date-1
        rain = rainfall_values[prev_day_index] if prev_day_index < len(rainfall_values) else 0.0
        provisional = False
        if rain is None:
            rain = 0.0
            provisional = True

        # ET mean for window (current_day - 7) â†’ current_day
        et_end = current_day - timedelta(days=1)
        et_start = et_end-timedelta(days=7) 
        et_mean = calculate_et_statistics_soil(
            plot['geometry'],
            et_start.strftime("%Y-%m-%d"),
            et_end.strftime("%Y-%m-%d")
        )

        # Soil moisture update
            # Soil moisture update
        net_change = rain - et_mean
        percent_change = (net_change / 300.0) * 100.0
        sm_today = sm_prev + (percent_change / 2.0)

        # Cap soil moisture at 100
        if sm_today > 100:
            sm_today = 100.0
        elif sm_today < 0:
            sm_today = 0.0   # optional: prevent going negative

        results.append({
            "day": current_day.isoformat(),
            "soil_moisture": round(sm_today, 2),
            "rainfall_mm_yesterday": round(rain, 2), 
            "rainfall_provisional": provisional,
            "et_mean_mm_yesterday": round(et_mean, 2), 
        })

        # Carry capped soil moisture forward
        sm_prev = sm_today


    response_data = {
        "plot_name": plot_name,
        "latitude": float(lat),
        "longitude": float(lon),
        "soil_moisture_stack": results
    }   
    return JSONResponse(content=response_data)
    

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
        
        

if __name__ == "__main__":
    uvicorn.run(
        "SEF:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False
    )
 
