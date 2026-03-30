from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, uvicorn
from cachetools import TTLCache
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx, uvicorn
from cachetools import TTLCache


app = FastAPI(title="Forecast Weather API")
 
# Allow CORS (for frontend use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cropeye.ai", "https://www.cropeye.ai", "http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173","http://192.168.42.72:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Cache: max 100 entries, TTL = 7200 sec (2 hours)
cache = TTLCache(maxsize=1000, ttl=7200)
 
BASE_URL = "https://api.open-meteo.com/v1/forecast"
 
# ------------------------
# Forecast Endpoint (Open-Meteo)
# ------------------------
@app.get("/forecast")
async def forecast(lat: float = Query(...), lon: float = Query(...)):
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
 
    async with httpx.AsyncClient() as client:
        response = await client.get(BASE_URL, params=params)
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

if __name__ == "__main__":
    uvicorn.run("current_forecast:app", host="0.0.0.0", port=8007, reload=False)

