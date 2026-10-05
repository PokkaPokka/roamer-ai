"""
Rank Wikivoyage city articles by popularity and save the top N to data/kb_cities.json.

Popularity = total page views over the last 12 full months, taken from the
Wikimedia Pageviews "top 1000 articles" list for each month. Only pages in
Wikivoyage's "City articles" category are kept (no countries, regions or help pages).

Run from the backend folder:
    python -m scripts.rank_cities            # top 500
    python -m scripts.rank_cities --top 100
    python -m scripts.rank_cities --refresh-countries   # keep the list, fix countries only
"""

import argparse
import json
import time
from collections import Counter
from datetime import date
from pathlib import Path

import requests

USER_AGENT = "RoamerAI/0.1 (student travel-planner project; builds a small Wikivoyage knowledge base)"
PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikivoyage/all-access/{year}/{month:02d}/all-days"
WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
CITY_CATEGORY = "Category:City articles"

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "kb_cities.json"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def get_json(url: str, params: dict | None = None, retries: int = 5) -> dict:
    for attempt in range(retries):
        response = session.get(url, params=params, timeout=30)

        # Rate limited: wait as long as the API asks (or back off), then retry.
        if response.status_code == 429 and attempt < retries - 1:
            wait = int(response.headers.get("Retry-After", 0)) or 5 * 2 ** attempt
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        # Be polite to the Wikimedia APIs.
        time.sleep(1)
        return response.json()


def last_full_months(count: int, today: date) -> list[tuple[int, int]]:
    year, month = today.year, today.month
    months = []
    for _ in range(count):
        month -= 1
        if month == 0:
            year, month = year - 1, 12
        months.append((year, month))
    return months


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def sum_page_views(months: list[tuple[int, int]]) -> Counter:
    views = Counter()
    for year, month in months:
        data = get_json(PAGEVIEWS_URL.format(year=year, month=month))
        for article in data["items"][0]["articles"]:
            views[article["article"].replace("_", " ")] += article["views"]
        print(f"  {year}-{month:02d}: {len(data['items'][0]['articles'])} pages")
    return views


def find_city_articles(titles: list[str]) -> dict[str, dict]:
    """Returns {pageviews title: {"title", "wikidata"}} for titles that are city articles."""
    cities = {}

    for batch in batched(titles, 50):
        data = get_json(WIKIVOYAGE_API, {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "redirects": 1,
            "prop": "categories|pageprops",
            "clcategories": CITY_CATEGORY,
            # Category results are paged across the whole batch (default 10),
            # so without this most pages in a batch come back with no categories.
            "cllimit": "max",
            "ppprop": "wikibase_item",
        })
        query = data["query"]

        # Map each requested title to the final page title after normalising and redirects.
        final_title = {title: title for title in batch}
        for step in query.get("normalized", []) + query.get("redirects", []):
            for original, current in final_title.items():
                if current == step["from"]:
                    final_title[original] = step["to"]

        pages = {page["title"]: page for page in query["pages"].values()}

        for original, title in final_title.items():
            page = pages.get(title)
            if page and page.get("categories"):
                cities[original] = {
                    "title": title,
                    "wikidata": page.get("pageprops", {}).get("wikibase_item"),
                }

    return cities


def find_countries(wikidata_ids: list[str]) -> dict[str, str]:
    """Returns {city wikidata id: country name} using Wikidata's "country" property (P17)."""
    country_of = {}

    for batch in batched(wikidata_ids, 50):
        data = get_json(WIKIDATA_API, {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(batch),
            "props": "claims",
        })
        for entity_id, entity in data.get("entities", {}).items():
            claims = entity.get("claims", {}).get("P17", [])
            # Cities can list past countries too (London lists the Roman Empire).
            # Use the "preferred" claim if there is one, otherwise one with no end date (P582).
            preferred = [claim for claim in claims if claim.get("rank") == "preferred"]
            current = [claim for claim in claims if "P582" not in claim.get("qualifiers", {})]
            for claim in preferred or current or claims:
                value = claim["mainsnak"].get("datavalue", {}).get("value", {}).get("id")
                if value:
                    country_of[entity_id] = value
                    break

    names = {}
    country_ids = sorted(set(country_of.values()))
    for batch in batched(country_ids, 50):
        data = get_json(WIKIDATA_API, {
            "action": "wbgetentities",
            "format": "json",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": "en",
        })
        for entity_id, entity in data.get("entities", {}).items():
            names[entity_id] = entity.get("labels", {}).get("en", {}).get("value")

    return {city_id: names.get(country_id) for city_id, country_id in country_of.items()}


def refresh_countries():
    """Re-look-up countries for the saved list without re-ranking it."""
    result = json.loads(OUTPUT_PATH.read_text())
    cities = result["cities"]

    wikidata_of = {}
    for batch in batched([city["title"] for city in cities], 50):
        data = get_json(WIKIVOYAGE_API, {
            "action": "query",
            "format": "json",
            "titles": "|".join(batch),
            "prop": "pageprops",
            "ppprop": "wikibase_item",
        })
        for page in data["query"]["pages"].values():
            wikidata_of[page["title"]] = page.get("pageprops", {}).get("wikibase_item")

    countries = find_countries([wikidata for wikidata in wikidata_of.values() if wikidata])

    changed = 0
    for city in cities:
        city["wikidata"] = wikidata_of.get(city["title"])
        country = countries.get(city["wikidata"])
        if country and country != city["country"]:
            print(f"  {city['title']}: {city['country']} -> {country}")
            city["country"] = country
            changed += 1

    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Updated {changed} countries in {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=500)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--refresh-countries", action="store_true",
                        help="only re-look-up countries for the saved list")
    args = parser.parse_args()

    if args.refresh_countries:
        refresh_countries()
        return

    months = last_full_months(args.months, date.today())
    print(f"Summing page views for {months[-1][0]}-{months[-1][1]:02d} to {months[0][0]}-{months[0][1]:02d}")
    views = sum_page_views(months)
    print(f"{len(views)} distinct pages in the monthly top lists")

    print("Keeping city articles...")
    cities = find_city_articles(list(views))

    # Several titles can redirect to the same city, so add their views together.
    city_views = Counter()
    wikidata_of = {}
    for original, city in cities.items():
        city_views[city["title"]] += views[original]
        wikidata_of[city["title"]] = city["wikidata"]
    print(f"{len(city_views)} city articles found")

    top = city_views.most_common(args.top)
    if len(top) < args.top:
        print(f"Warning: only {len(top)} cities available, fewer than --top {args.top}")

    print("Looking up countries on Wikidata...")
    countries = find_countries([wikidata_of[title] for title, _ in top if wikidata_of[title]])

    result = {
        "generated": date.today().isoformat(),
        "method": f"Total en.wikivoyage page views over {args.months} months "
                  f"({months[-1][0]}-{months[-1][1]:02d} to {months[0][0]}-{months[0][1]:02d}), "
                  f"from the monthly top-1000 lists, filtered to '{CITY_CATEGORY}'",
        "cities": [
            {
                "rank": rank,
                "title": title,
                "country": countries.get(wikidata_of[title]),
                "wikidata": wikidata_of[title],
                "page_views": city_views[title],
                "source_url": "https://en.wikivoyage.org/wiki/" + title.replace(" ", "_"),
            }
            for rank, (title, _) in enumerate(top, start=1)
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(top)} cities to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
