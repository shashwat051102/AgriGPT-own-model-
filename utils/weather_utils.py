import os
import requests

weather_api = os.getenv("WEATHER_API_KEY")

def get_weather(location):
    if not weather_api:
        return {"error": "Weather API key not configured"}
    try:
        base_url = f"http://api.weatherapi.com/v1/current.json?key={weather_api}&q={location}&aqi=no"
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Unable to fetch data: {response.status_code} - {response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}