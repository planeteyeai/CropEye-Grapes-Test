from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import httpx, uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, uvicorn
from cachetools import TTLCache
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime


_warmup_task = None

async def _warmup_loop(client: httpx.AsyncClient):
    while True:
        try:
            await client.get(BASE_URL, params={"latitude": 18.5, "longitude": 73.8, "daily": "temperature_2m_max", "forecast_days": 1, "timezone": "auto"})
        except Exception:
            pass
        await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warmup_task
    app.state.http_client = httpx.AsyncClient(timeout=15.0)
    _warmup_task = asyncio.create_task(_warmup_loop(app.state.http_client))
    yield
    _warmup_task.cancel()
    try:
        await _warmup_task
    except asyncio.CancelledError:
        pass
    await app.state.http_client.aclose()

app = FastAPI(title="Forecast Weather API", lifespan=lifespan)
 
# Allow CORS (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cropeye.ai", "https://www.cropeye.ai", "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173","http://192.168.42.72:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=256)
 
# Cache: max 100 entries, TTL = 7200 sec (2 hours)
cache = TTLCache(maxsize=1000, ttl=7200)
 
BASE_URL = "https://api.open-meteo.com/v1/forecast"
 
# ------------------------
# Forecast Endpoint (Open-Meteo)
# ------------------------
@app.get("/forecast")
async def forecast(request: Request, lat: float = Query(...), lon: float = Query(...)):
    cache_key = f"forecast_{lat}_{lon}"
    if cache_key in cache:
        return {"source": "cache", "data": cache[cache_key]}
 
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "windspeed_10m_max,relative_humidity_2m_max",
        "timezone": "auto",
        "forecast_days": 8
    }
 
    response = await request.app.state.http_client.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
 
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
 
    cache[cache_key] = result
    return {"source": "api", "data": result}



# Rate Limiter (10 requests per minute per IP)
# limiter = Limiter(key_func=get_remote_address)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# In-memory cache: max 500 entries, each cached for 1800 seconds (30 mins)
cache = TTLCache(maxsize=4000, ttl=1800)

API_KEY = "a7977cf38bb044e9a4d82500252008"   # generated using dev1 gmail account
C_BASE_URL = "https://api.weatherapi.com/v1/current.json"

@app.get("/current-weather")
# @limiter.limit("100/minute")  # limit per IP
async def get_curr_weather(
    request: Request,
    lat: float = Query(None, description="Latitude"),
    lon: float = Query(None, description="Longitude"),
    city: str = Query(None, description="City name (optional if lat/lon given)")
):
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
    r = await request.app.state.http_client.get(C_BASE_URL, params=params)
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
    }

    # Save to cache
    cache[q] = response
    return response
# 📥 Request Model
class WeatherData(BaseModel):
    temperature_c: float
    humidity: float
    wind_kph: float
    precip_mm: float = 0
    cloud: float = 0
    pressure_mb: float = 1015
    dewpoint_c: float = None


# 🌧 Core Logic Function
def calculate_rain_score(data: WeatherData):
    rain_score = 0

    temp = data.temperature_c
    humidity = data.humidity
    wind = data.wind_kph
    precip = data.precip_mm
    cloud = data.cloud
    pressure = data.pressure_mb
    dew = data.dewpoint_c if data.dewpoint_c is not None else temp - 8

    # Humidity
    if humidity >= 80:
        rain_score += 3
    elif humidity >= 60:
        rain_score += 2
    elif humidity >= 45:
        rain_score += 1

    # Cloud cover
    if cloud >= 80:
        rain_score += 3
    elif cloud >= 60:
        rain_score += 2
    elif cloud >= 40:
        rain_score += 1

    # Pressure
    if pressure <= 1005:
        rain_score += 3
    elif pressure <= 1010:
        rain_score += 2
    elif pressure <= 1015:
        rain_score += 1

    # Dew point difference
    diff = temp - dew
    if diff <= 2:
        rain_score += 3
    elif diff <= 4:
        rain_score += 2
    elif diff <= 6:
        rain_score += 1

    # Current rain
    if precip > 0:
        rain_score += 3

    # Wind factor
    if wind >= 15 and humidity >= 60:
        rain_score += 1

    return rain_score


# 🌧 Decision Logic
def get_rain_prediction(score: int):
    if score >= 10:
        return {"alert": "HIGH CHANCE OF RAIN", "probability": "75-90%"}
    elif score >= 7:
        return {"alert": "MEDIUM CHANCE OF RAIN", "probability": "50-70%"}
    elif score >= 4:
        return {"alert": "LOW CHANCE OF RAIN", "probability": "30-50%"}
    else:
        return {"alert": "VERY LOW CHANCE OF RAIN", "probability": "0-20%"}


# 🚀 API Endpoint
@app.post("/predict-rain")
def predict_rain(data: WeatherData):
    score = calculate_rain_score(data)
    result = get_rain_prediction(score)

    return {
        "rain_score": score,
        "prediction": result["alert"],
        "probability": result["probability"]
    }
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run("current_forecast:app", host="0.0.0.0", port=8007, reload=False)

