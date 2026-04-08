
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

    # ---------------- CURRENT WEATHER ----------------
    params = {"key": API_KEY, "q": q, "aqi": "yes"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(C_BASE_URL, params=params)
        if r.status_code != 200:
            raise HTTPException(502, f"WeatherAPI error: {r.text}")
        data = r.json()

    if "error" in data:
        err = data["error"]
        raise HTTPException(502, err.get("message"))

    current = data["current"]

    # ---------------- RAIN SCORE (CURRENT) ----------------
    def calculate_rain_score(humidity, cloud, pressure_mb, temp_c, dew_c, precip, wind):
        score = 0

        if humidity >= 80: score += 3
        elif humidity >= 60: score += 2
        elif humidity >= 45: score += 1

        if cloud >= 80: score += 3
        elif cloud >= 60: score += 2
        elif cloud >= 40: score += 1

        if pressure_mb <= 1005: score += 3
        elif pressure_mb <= 1010: score += 2
        elif pressure_mb <= 1015: score += 1

        diff = temp_c - dew_c
        if diff <= 2: score += 3
        elif diff <= 4: score += 2
        elif diff <= 6: score += 1

        if precip > 0: score += 3

        if wind >= 15 and humidity >= 60:
            score += 1

        return score

    rain_score = calculate_rain_score(
        current.get("humidity", 0),
        current.get("cloud", 0),
        current.get("pressure_mb", 1015),
        current.get("temp_c", 0),
        current.get("dewpoint_c", current.get("temp_c", 0)),
        current.get("precip_mm", 0),
        current.get("wind_kph", 0)
    )

    def map_alert(score):
        if score >= 10:
            return "HIGH CHANCE OF RAIN", "75-90%"
        elif score >= 7:
            return "MEDIUM CHANCE OF RAIN", "50-70%"
        elif score >= 4:
            return "LOW CHANCE OF RAIN", "30-50%"
        else:
            return "VERY LOW CHANCE OF RAIN", "0-20%"

    alert, probability = map_alert(rain_score)

    # ---------------- FORECAST (48 HOURS) ----------------
    forecast_url = "https://api.weatherapi.com/v1/forecast.json"
    params_forecast = {"key": API_KEY, "q": q, "hours": 48}

    async with httpx.AsyncClient(timeout=15) as client:
        f = await client.get(forecast_url, params=params_forecast)
        forecast_data = f.json()

    hourly_data = forecast_data["forecast"]["forecastday"]

    best_hour = None
    best_score = -1

    time_buckets = {
        "morning": [],
        "afternoon": [],
        "night": []
    }

    for day in hourly_data:
        for hour in day["hour"]:
            score = calculate_rain_score(
                hour.get("humidity", 0),
                hour.get("cloud", 0),
                hour.get("pressure_mb", 1015),
                hour.get("temp_c", 0),
                hour.get("dewpoint_c", hour.get("temp_c", 0)),
                hour.get("precip_mm", 0),
                hour.get("wind_kph", 0)
            )

            time_str = hour["time"]
            hour_val = int(time_str.split(" ")[1].split(":")[0])

            # Bucket classification
            if 6 <= hour_val < 12:
                bucket = "morning"
            elif 12 <= hour_val < 18:
                bucket = "afternoon"
            else:
                bucket = "night"

            time_buckets[bucket].append(score)

            if score > best_score:
                best_score = score
                best_hour = time_str

    # Find best time of day
    avg_scores = {k: (sum(v)/len(v) if v else 0) for k, v in time_buckets.items()}
    best_time_of_day = max(avg_scores, key=avg_scores.get)

    future_alert, future_probability = map_alert(best_score)

    # ---------------- RESPONSE ----------------
    response = {
        "location": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "localtime": data["location"]["localtime"],
        "latitude": data["location"]["lat"],
        "longitude": data["location"]["lon"],

        # Current weather
        "temperature_c": current["temp_c"],
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"],
        "precip_mm": current["precip_mm"],
        "cloud": current.get("cloud"),
        "pressure_mb": current.get("pressure_mb"),
        "dewpoint_c": current.get("dewpoint_c"),
        "condition_text": current.get("condition", {}).get("text"),

        # Current rain prediction
        "rain_score": rain_score,
        "rain_alert": alert,
        "rain_probability": probability,

        # 🔥 NEW: 24–48 hour prediction
        "next_48h_prediction": {
            "most_likely_time": best_hour,
            "best_time_of_day": best_time_of_day,
            "rain_alert": future_alert,
            "rain_probability": future_probability
        }
    }

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
