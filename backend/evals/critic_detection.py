"""
Eval 2: how many plan errors does the code critic catch, and does it raise false alarms?

Method (mutation testing):
1. make: write 6 realistic plans with the app's final_agent prompt, real Wikivoyage
   excerpts and a fixed chosen flight per trip. A hosted model (Groq gpt-oss-120b) writes
   them; it only produces test material and isn't being scored.
2. run: the critic first cleans each plan (its fixes on the raw plan are listed for review).
   Then each kind of error is injected into the clean plan one at a time, and the critic
   must catch it. The clean plans are also checked for false alarms.

Run from the backend folder:
    python -m evals.critic_detection make [--set heldout]
    python -m evals.critic_detection run [--set heldout]

The "tuning" set found bugs that were fixed in the critic, and reviewing the "heldout"
set's flags found three more. The "final" set was written after all changes and never
used to change the critic, so its numbers are the ones to quote.
"""

import argparse
import asyncio
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("LANGSMITH_TRACING", "false")

import app.graph as g
from app.critic import AMOUNT, CITATION, budget_rows, claim_names, excerpt_places, normalize, parse_excerpts, plain, review_plan, section_lines
from app.db import close_db, open_db
from app.knowledge_base import city_in_knowledge_base

DATA = Path(__file__).parent / "data"

# Destination, origin, flight price (AUD, round trip) and landing time on day 1.
TRIPS = [
    ("Tokyo", "Melbourne", 1354, "18:30"),
    ("Paris", "London", 255, "11:05"),
    ("Rome", "Paris", 298, "09:40"),
    ("Bangkok", "Sydney", 912, "20:15"),
    ("Sydney", "Melbourne", 210, "08:25"),
    ("New York City", "Toronto", 640, "14:50"),
]
# Plans written after the critic was tuned on TRIPS, to check it on unseen text.
HELDOUT_TRIPS = [
    ("Barcelona", "London", 189, "17:45"),
    ("Seoul", "Sydney", 1120, "19:05"),
    ("Istanbul", "Berlin", 410, "13:20"),
]
# Written after all critic changes and never used to change it: the honest numbers.
FINAL_TRIPS = [
    ("Lisbon", "Madrid", 160, "21:10"),
    ("Hanoi", "Singapore", 380, "16:40"),
    ("Mexico City", "Los Angeles", 520, "12:35"),
]
SETS = {"tuning": TRIPS, "heldout": HELDOUT_TRIPS, "final": FINAL_TRIPS}
START = date(2026, 11, 20)
# Groq's free tier allows 8,000 tokens a minute; 8 excerpts keep a prompt under that.
EXCERPTS_PER_PLAN = 8


def trip_state(destination, origin, price, landing, guide_chunks) -> dict:
    outbound, inbound = START.isoformat(), (START + timedelta(days=3)).isoformat()
    option = {
        "number": 1, "price": price, "currency": "AUD", "airlines": ["Example Air"], "round_trip": True,
        "departure_airport": origin[:3].upper(), "departure_time": f"{outbound} 06:00",
        "arrival_airport": destination[:3].upper(), "arrival_time": f"{outbound} {landing}",
        "outbound_date": outbound, "return_date": inbound, "duration": "", "stops": 0,
        "text": f"Option 1: AUD {price} (Round trip, price shown is for the whole trip)\n"
                f"- Example Air EX 1: {origin[:3].upper()} {outbound} 06:00 -> {destination[:3].upper()} {outbound} {landing}",
    }
    excerpts = [f"[{n}] {chunk['content']}" for n, chunk in enumerate(guide_chunks, start=1)]
    return {
        "user_query": f"3 days in {destination} from {origin} (travel dates: leaving {outbound}, returning {inbound})",
        "flight_results": f"Google Flights results for {origin} -> {destination}, departing {outbound}, returning {inbound}, 1 adult (prices in AUD)\n\n{option['text']}",
        "chosen_flight": g.chosen_flight_text(option, 1),
        "chosen_option": {key: value for key, value in option.items() if key != "text"},
        "hotel_results": f"1. A well-reviewed mid-range hotel in central {destination}, about AUD 180 a night.\n"
                         f"2. A budget hostel near the main station in {destination}, about AUD 60 a night.",
        "guide_context": "\n\n".join(excerpts),
        "guide_sources": [
            {"number": n, "city": chunk["city"], "section": chunk["section"], "url": chunk["source_url"]}
            for n, chunk in enumerate(guide_chunks, start=1)
        ],
        "trip_request": {"departure_date": outbound, "return_date": inbound},
    }


async def make(name: str):
    FIXTURES = DATA / f"critic_plans_{name}.json"
    from langchain_groq import ChatGroq

    g.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6)
    g.report = lambda message: None  # final_agent reports progress; there's no stream here

    # Resumable: plans already written are kept.
    plans = json.loads(FIXTURES.read_text()) if FIXTURES.exists() else []
    done = {plan["destination"] for plan in plans}
    for destination, origin, price, landing in SETS[name]:
        if destination in done:
            continue
        city = destination if await city_in_knowledge_base(destination) else None
        chunks = (await g.fixed_guide_search(destination, city))[:EXCERPTS_PER_PLAN]
        state = trip_state(destination, origin, price, landing, chunks)
        result = await g.final_agent(state)
        state["final_plan"] = result["final_plan"]
        plans.append({"destination": destination, **state})
        FIXTURES.write_text(json.dumps(plans, indent=2, ensure_ascii=False))
        print(f"{destination}: {len(state['final_plan'])} chars, {len(chunks)} excerpts")
        await asyncio.sleep(60)  # one prompt per minute under the tokens-per-minute limit


# =========================
# Injected errors
# =========================

def budget_row(plan: str, words: tuple[str, ...]) -> tuple[int, str] | None:
    start, end = section_lines(plan, ["budget"])
    lines = plan.splitlines()
    for index in range(start, end):
        if any(word in lines[index].lower() for word in words) and re.search(r"\d", lines[index]):
            return index, lines[index]
    return None


def replace_first_amount(line: str, new: str) -> str:
    # The first amount after the label (skips "(3 nights)"-style notes in the label);
    # uses the critic's own number pattern so "1 354" is one number.
    label = re.split(r"[|:]", line, maxsplit=1)[0]
    return label + AMOUNT.sub(new, line[len(label):], count=1)


def set_line(plan: str, index: int, line: str) -> str:
    lines = plan.splitlines()
    lines[index] = line
    return "\n".join(lines)


def double_flight(plan, fixture):
    row = budget_row(plan, ("flight", "airfare"))
    price = fixture["chosen_option"]["price"]
    return row and set_line(plan, row[0], replace_first_amount(row[1], f"{2 * price:,}"))


def wrong_flight_price(plan, fixture):
    row = budget_row(plan, ("flight", "airfare"))
    price = fixture["chosen_option"]["price"]
    return row and set_line(plan, row[0], replace_first_amount(row[1], f"{price + 300:,}"))


def readable_budget(plan) -> bool:
    """One total and no rows with alternatives: the only budgets the critic checks."""
    rows = budget_rows(plan)
    return sum("total" in row["label"].lower() for row in rows) == 1 and not any(row.get("ambiguous") for row in rows)


def wrong_total(plan, fixture):
    row = budget_row(plan, ("total",))
    if not row or not readable_budget(plan):
        return None  # several totals or alternatives: the critic doesn't check these by design
    amount = AMOUNT.search(re.split(r"[|:]", row[1], maxsplit=1)[1])
    value = float(amount.group(1).replace(",", "").replace(" ", ""))
    return set_line(plan, row[0], replace_first_amount(row[1], f"{round(value * 1.15):,}"))


def cited_claims(plan, fixture):
    """(line index, citation number, names in its claim) for citations the critic keeps."""
    excerpts = parse_excerpts(fixture["guide_context"])
    found = []
    for index, line in enumerate(plan.splitlines()):
        last = 0
        for match in CITATION.finditer(line):
            names = claim_names(line[last:match.start()], excerpt_places(fixture["guide_context"]))
            if names and int(match.group(1)) in excerpts:
                found.append((index, match, names))
            last = match.end()
    return found


def citation_wrong_excerpt(plan, fixture):
    excerpts = parse_excerpts(fixture["guide_context"])
    for index, match, names in cited_claims(plan, fixture):
        # Point the citation at an excerpt that names none of the claim's places.
        other = next((n for n, body in excerpts.items()
                      if n != int(match.group(1)) and not any(plain(name) in plain(body) for name in names)), None)
        if other:
            line = plan.splitlines()[index]
            return set_line(plan, index, line[:match.start()] + f"[{other}]" + line[match.end():])
    return None


def citation_missing_excerpt(plan, fixture):
    found = cited_claims(plan, fixture)
    if not found:
        return None
    index, match, _ = found[0]
    line = plan.splitlines()[index]
    return set_line(plan, index, line[:match.start()] + "[99]" + line[match.end():])


def missing_section(plan, fixture):
    start, end = section_lines(plan, ["recommendation"])
    if not start:
        return None
    lines = plan.splitlines()
    return "\n".join(lines[:start - 1] + lines[end:])


def missing_dates(plan, fixture):
    # Move the whole plan to other dates, so the chosen flight's dates aren't mentioned.
    moved = plan
    for iso in (fixture["chosen_option"]["outbound_date"], fixture["chosen_option"]["return_date"]):
        day = date.fromisoformat(iso)
        other = day + timedelta(days=9)
        moved = moved.replace(iso, other.isoformat())
        for month in (day.strftime("%B"), day.strftime("%b")):
            moved = re.sub(rf"\b0?{day.day}(st|nd|rd|th)?\s+{month}\b", f"{other.day} {month}", moved)
            moved = re.sub(rf"\b{month}\.?\s+0?{day.day}(st|nd|rd|th)?\b", f"{month} {other.day}", moved)
    return moved if moved != plan else None


def early_day_one(plan, fixture):
    if int(fixture["chosen_option"]["arrival_time"].split(" ")[1][:2]) < 12:
        return None  # a morning landing allows morning plans
    # Insert under the itinerary's Day 1 heading, not a mention of "Day 1" elsewhere.
    lines = plan.splitlines()
    start, end = section_lines(plan, ["itinerary", "day-by-day", "day by day"])
    for index in range(start, end):
        if re.search(r"\bDay\s*1\b", lines[index]):
            lines.insert(index + 1, "- **Morning:** Visit the city's most famous museum.")
            return "\n".join(lines)
    return None


def copied_prompt(plan, fixture):
    return plan + "\n\nTravel Guide Excerpts (from Wikivoyage, numbered):\n" + fixture["guide_context"][:600]


MUTATIONS = {
    "flight counted twice": (double_flight, lambda r: any("twice" in f for f in r.fixes)),
    "wrong flight cost": (wrong_flight_price, lambda r: any("flight cost" in f for f in r.fixes)),
    "wrong budget total": (wrong_total, lambda r: any("budget total" in f for f in r.fixes)),
    "citation to the wrong excerpt": (citation_wrong_excerpt, lambda r: any("citation" in f for f in r.fixes)),
    "citation to a missing excerpt": (citation_missing_excerpt, lambda r: any("citation" in f for f in r.fixes)),
    "missing section": (missing_section, lambda r: any("Missing sections" in p for p in r.problems)),
    "travel dates missing": (missing_dates, lambda r: any("travel date" in p for p in r.problems)),
    "plans before landing": (early_day_one, lambda r: any("flight lands" in p for p in r.problems)),
    "copied prompt text": (copied_prompt, lambda r: any("copies text" in p for p in r.problems)),
}


def review(plan, fixture):
    return review_plan(
        g.strip_sources(plan), fixture["guide_context"], fixture["chosen_option"]["price"],
        g.travel_dates(fixture), fixture["chosen_option"]["arrival_time"],
    )


def run(name: str):
    fixtures = json.loads((DATA / f"critic_plans_{name}.json").read_text())
    print(f"== {name} set: {len(fixtures)} plans")
    results = {name: {"caught": 0, "cases": 0} for name in MUTATIONS}
    raw_flags, false_alarms = [], 0

    for fixture in fixtures:
        first = review(fixture["final_plan"], fixture)
        raw_flags.append({"destination": fixture["destination"], "fixes": first.fixes, "problems": first.problems})
        clean = normalize(first.plan)
        again = review(clean, fixture)
        false_alarms += len(again.fixes) + len(again.problems)

        for error, (mutate, caught) in MUTATIONS.items():
            broken = mutate(clean, fixture)
            if not broken:
                continue
            results[error]["cases"] += 1
            results[error]["caught"] += bool(caught(review(broken, fixture)))

    print("Flags on the raw generated plans (check by hand whether each is real):")
    for item in raw_flags:
        print(f"  {item['destination']}: fixes {item['fixes']} | problems {item['problems']}")

    total_caught = sum(r["caught"] for r in results.values())
    total_cases = sum(r["cases"] for r in results.values())
    print(f"\nInjected errors caught: {total_caught}/{total_cases}")
    for error, result in results.items():
        print(f"  {error:32} {result['caught']}/{result['cases']}")
    print(f"False alarms on the {len(fixtures)} cleaned plans: {false_alarms}")
    skipped = sum(not readable_budget(normalize(review(f["final_plan"], f).plan)) for f in fixtures)
    print(f"Budgets the critic skips by design (several totals or alternatives): {skipped}/{len(fixtures)}")

    (DATA / f"critic_results_{name}.json").write_text(json.dumps({"by_error": results, "caught": total_caught, "cases": total_cases,
                                   "false_alarms_on_clean": false_alarms, "raw_flags": raw_flags}, indent=2))


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["make", "run"])
    parser.add_argument("--set", choices=list(SETS), default="tuning")
    args = parser.parse_args()
    if args.stage == "run":
        run(args.set)
        return
    await open_db()
    try:
        await make(args.set)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
