
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request
from fastapi.responses import Response
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, uvicorn,os
from cachetools import TTLCache 



app = FastAPI(title="Forecast Weather API")
# ------------------------------
# Prometheus Metrics (with unique names for Forecast API)
# ------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total_forecast",
    "Total HTTP Requests for Forecast API",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds_forecast",
    "Request latency for Forecast API",
    ["endpoint"]
) 
# Allow CORS (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cropeye.ai", "https://www.cropeye.ai", "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173","http://192.168.42.72:5174","https://cropeye-00.onrender.com","http://localhost:3098",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------------------------
# Prometheus Middleware
# ------------------------------
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    endpoint = request.url.path

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)

    return response
 
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# forecast cache: TTL 7200s (2h); current-weather cache: TTL 1800s (30min)
forecast_cache = TTLCache(maxsize=1000, ttl=7200)
cache = TTLCache(maxsize=4000, ttl=1800)

# ------------------------
# Forecast Endpoint (Open-Meteo)
# ------------------------
@app.get("/forecast")
async def forecast(lat: float = Query(...), lon: float = Query(...)):
    cache_key = f"forecast_{lat}_{lon}"
    if cache_key in forecast_cache:
        return {"source": "cache", "data": forecast_cache[cache_key]}

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "windspeed_10m_max,relative_humidity_2m_max",
        "timezone": "auto",
        "forecast_days": 8
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            response = await client.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:
            raise HTTPException(status_code=504, detail="Forecast provider timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Forecast provider error: HTTP {exc.response.status_code}",
            ) from exc
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail="Unable to reach forecast provider") from exc

    result = []
    for i in range(len(data["daily"]["time"])):
        result.append({
            "date": data["daily"]["time"][i],
            "temperature_max": f"{data['daily']['temperature_2m_max'][i]} °C",
            "temperature_min": f"{data['daily']['temperature_2m_min'][i]} °C",
            "precipitation": f"{data['daily']['precipitation_sum'][i]} mm",
            "wind_speed_max": f"{data['daily']['windspeed_10m_max'][i]} km/h",
            "humidity_max": f"{data['daily']['relative_humidity_2m_max'][i]} %"
        })

    forecast_cache[cache_key] = result
    return {"source": "api", "data": result}


# Rate Limiter (10 requests per minute per IP)
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEY = os.environ.get("WEATHER_API_KEY")
C_BASE_URL = "https://api.weatherapi.com/v1/current.json"


@app.get("/health")
async def health_check():
    """Health check - always 200 for Railway so container is not stopped"""
    return {"status": "healthy", "service": "current_forecast"}


@app.get("/current-weather")
# @limiter.limit("100/minute")  # limit per IP
async def get_curr_weather(
    request: Request,
    lat: float = Query(None, description="Latitude"),
    lon: float = Query(None, description="Longitude"),
    city: str = Query(None, description="City name (optional if lat/lon given)")
):
    if not API_KEY:
        raise HTTPException(503, "WEATHER_API_KEY not configured")

    # Build query string
    if lat is not None and lon is not None:
        q = f"{lat},{lon}"
    elif city:
        q = city
    else:
        raise HTTPException(400, "Provide either city or lat/lon")

    # Check cache first
    if q in cache:
        return cache[q]

    # Prepare request to WeatherAPI
    params = {"key": API_KEY, "q": q, "aqi": "yes"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(C_BASE_URL, params=params)
        if r.status_code != 200:
            raise HTTPException(502, f"WeatherAPI error: {r.text}")
        data = r.json()

    # Handle WeatherAPI errors (like quota exceeded)
    if "error" in data:
        err = data["error"]
        code = err.get("code")
        message = err.get("message", "Unknown WeatherAPI error")
        raise HTTPException(429 if code == 2007 else 502, detail=message)

    current = data["current"]

    # ------------------------------
    # Rain Prediction Logic
    # ------------------------------
    humidity = current.get("humidity", 0)
    cloud = current.get("cloud", 0)
    pressure_mb = current.get("pressure_mb", 1015)
    temperature_c = current.get("temp_c", 0)
    dewpoint_c = current.get("dewpoint_c", temperature_c)
    precip_mm = current.get("precip_mm", 0)
    wind_kph = current.get("wind_kph", 0)

    rain_score = 0

    # Humidity contribution
    if humidity >= 80:
        rain_score += 3
    elif humidity >= 60:
        rain_score += 2
    elif humidity >= 45:
        rain_score += 1

    # Cloud cover contribution
    if cloud >= 80:
        rain_score += 3
    elif cloud >= 60:
        rain_score += 2
    elif cloud >= 40:
        rain_score += 1

    # Pressure contribution (low pressure = rain chance)
    if pressure_mb <= 1005:
        rain_score += 3
    elif pressure_mb <= 1010:
        rain_score += 2
    elif pressure_mb <= 1015:
        rain_score += 1

    # Dew point difference
    temp_diff = temperature_c - dewpoint_c

    if temp_diff <= 2:
        rain_score += 3
    elif temp_diff <= 4:
        rain_score += 2
    elif temp_diff <= 6:
        rain_score += 1

    # Current rain indicator
    if precip_mm > 0:
        rain_score += 3

    # Wind effect
    if wind_kph >= 15 and humidity >= 60:
        rain_score += 1

    # ------------------------------
    # Rain Alert Mapping
    # ------------------------------
    if rain_score >= 10:
        alert = "HIGH CHANCE OF RAIN"
        probability = "75-90%"
    elif rain_score >= 7:
        alert = "MEDIUM CHANCE OF RAIN"
        probability = "50-70%"
    elif rain_score >= 4:
        alert = "LOW CHANCE OF RAIN"
        probability = "30-50%"
    else:
        alert = "VERY LOW CHANCE OF RAIN"
        probability = "0-20%"

    response = {
        "location": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "localtime": data["location"]["localtime"],
        "latitude": data["location"]["lat"],
        "longitude": data["location"]["lon"],

        # Weather details
        "temperature_c": current["temp_c"], 
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"],
        "precip_mm": current["precip_mm"],
        "cloud": current.get("cloud"),
        "pressure_mb": current.get("pressure_mb"),
        "dewpoint_c": current.get("dewpoint_c"),
        "condition_text": current.get("condition", {}).get("text"),

        # ✅ Rain prediction output
        "rain_score": rain_score,
        "rain_alert": alert,
        "rain_probability": probability
    }

    # Save to cache
    cache[q] = response
    return response


# ------------------------------
# Prometheus Metrics Endpoint
# ------------------------------
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint - accessible without authentication"""
    try:
        metrics_data = generate_latest()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        print(f"Error generating metrics: {e}")
        return Response(
            content=f"Error generating metrics: {str(e)}",
            status_code=500
        )
        
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
