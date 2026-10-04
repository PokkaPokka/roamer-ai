import os
import certifi
import requests
from dotenv import load_dotenv

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

API_KEY = os.getenv("SERPAPI_API_KEY")

BASE_URL = "https://serpapi.com/search.json"

DEFAULT_CURRENCY = os.getenv("FLIGHT_CURRENCY", "AUD")


def format_minutes(minutes: int | None) -> str:
    if not minutes:
        return "Unknown"
    return f"{minutes // 60}h {minutes % 60}m"


def format_option(index: int, option: dict, currency: str) -> str:
    legs = option.get("flights", [])
    layovers = option.get("layovers", [])
    price = option.get("price")

    stops = len(legs) - 1
    stops_text = "Direct" if stops == 0 else f"{stops} stop{'s' if stops > 1 else ''}"
    price_text = f"{currency} {price}" if price is not None else "Price unavailable"

    lines = [
        f"Option {index}: {price_text} ({option.get('type', 'Flight')}, price shown is for the whole trip)",
        f"Total duration: {format_minutes(option.get('total_duration'))} | {stops_text}",
    ]

    for leg in legs:
        dep = leg.get("departure_airport", {})
        arr = leg.get("arrival_airport", {})
        lines.append(
            f"- {leg.get('airline', 'Unknown airline')} {leg.get('flight_number', '')}: "
            f"{dep.get('id', '?')} {dep.get('time', '?')} -> {arr.get('id', '?')} {arr.get('time', '?')} "
            f"({format_minutes(leg.get('duration'))}, {leg.get('travel_class', 'Economy')})"
        )

    for layover in layovers:
        lines.append(
            f"- Layover: {format_minutes(layover.get('duration'))} at {layover.get('name', layover.get('id', '?'))}"
        )

    return "\n".join(lines)


def search_google_flights(
    origin_iata: str,
    destination_iata: str,
    outbound_date: str,
    return_date: str | None = None,
    adults: int = 1,
    currency: str = DEFAULT_CURRENCY,
    max_results: int = 5,
):
    """
    Searches Google Flights through SerpApi for real future fares.

    Dates use YYYY-MM-DD. Leave return_date empty for a one-way search.
    """

    if not API_KEY:
        return (
            "Flight API error: SERPAPI_API_KEY is missing.\n"
            "Please add this in your .env file:\n"
            "SERPAPI_API_KEY=your_api_key_here"
        )

    params = {
        "engine": "google_flights",
        "departure_id": origin_iata,
        "arrival_id": destination_iata,
        "outbound_date": outbound_date,
        "type": 1 if return_date else 2,
        "adults": adults,
        "currency": currency,
        "hl": "en",
        "api_key": API_KEY,
    }

    if return_date:
        params["return_date"] = return_date

    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Flight API request failed: {e}"
    except ValueError:
        return "Flight API returned invalid JSON."

    if "error" in data:
        return f"Flight API error: {data['error']}"

    options = (data.get("best_flights") or []) + (data.get("other_flights") or [])

    trip_text = f"{origin_iata} -> {destination_iata}, departing {outbound_date}"
    if return_date:
        trip_text += f", returning {return_date}"
    trip_text += f", {adults} adult{'s' if adults > 1 else ''}"

    if not options:
        return f"No flights found for {trip_text}."

    sections = [f"Google Flights results for {trip_text} (prices in {currency})"]

    insights = data.get("price_insights")
    if insights:
        typical = insights.get("typical_price_range")
        typical_text = f"{currency} {typical[0]}-{typical[1]}" if typical else "unknown"
        sections.append(
            f"Price insight: lowest {currency} {insights.get('lowest_price', '?')}, "
            f"typical range {typical_text}, current level: {insights.get('price_level', 'unknown')}"
        )

    sections.extend(
        format_option(i, option, currency)
        for i, option in enumerate(options[:max_results], 1)
    )

    return "\n\n".join(sections)


if __name__ == "__main__":
    print(search_google_flights("MEL", "NRT", "2026-11-10", "2026-11-13"))
