"""
Checks a finished travel plan with plain code, no LLM.

The local 8B model can't reliably proofread its own plan, so the critic uses rules:
- mechanical problems are fixed here directly: the budget total, the flight row in
  the budget, citations that don't match their excerpt
- content problems are returned as a list for one LLM rewrite: a missing section,
  missing travel dates, copied prompt text

review_plan() is the entry point; the rest are small helpers kept separate so they
can be tested on saved plans.
"""

from dataclasses import dataclass, field
import re
import unicodedata


@dataclass
class Review:
    plan: str                                           # the plan with code fixes applied
    fixes: list[str] = field(default_factory=list)      # what the code fixed
    problems: list[str] = field(default_factory=list)   # what needs an LLM rewrite


# =========================
# Sections and copied prompt text
# =========================

# Each required section, with the words that count as that section's heading.
REQUIRED_SECTIONS = {
    "Trip Summary": ["summary"],
    "Flight Information": ["flight"],
    "Hotel Suggestions": ["hotel", "accommodation"],
    "Day-by-Day Itinerary": ["itinerary", "day-by-day", "day by day"],
    "Estimated Budget": ["budget"],
    "Final Recommendations": ["recommendation"],
}

# Prompt text that should never appear in a plan. When it does, the model has
# started copying its input (once for 7,900 tokens: hotels, then the excerpts).
# Search facts like "Price insight:" can be quoted legitimately, so only the
# excerpt block and instructions count.
PROMPT_MARKERS = [
    "Travel Guide Excerpts (from Wikivoyage",
    "How to use the excerpts",
    "Problems found by an automatic check",
    "Format the answer using these sections",
]

MAX_PLAN_CHARS = 20000


def headings(plan: str) -> list[str]:
    return [line.lstrip("#").strip().lower() for line in plan.splitlines() if line.startswith("#")]


def missing_sections(plan: str) -> list[str]:
    found = headings(plan)
    return [
        name for name, words in REQUIRED_SECTIONS.items()
        if not any(word in heading for heading in found for word in words)
    ]


def copied_prompt(plan: str) -> list[str]:
    return [marker for marker in PROMPT_MARKERS if marker.lower() in plan.lower()]


# =========================
# Budget
# =========================

# A money amount: "1,354", "360–540", "2,114 - 3,444", "600 to 800".
AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)(?:\s*(?:–|—|-|to)\s*(\d[\d,]*(?:\.\d+)?))?")


def to_number(text: str) -> float:
    return float(text.replace(",", ""))


def format_amount(low: float, high: float) -> str:
    if round(low) == round(high):
        return f"{round(low):,}"
    return f"{round(low):,}–{round(high):,}"


def section_lines(plan: str, words: list[str]) -> tuple[int, int]:
    """Line range (start, end) of the first section whose heading has one of the words."""
    lines = plan.splitlines()
    start, level = None, 0
    for index, line in enumerate(lines):
        if line.startswith("#"):
            depth = len(line) - len(line.lstrip("#"))
            # A sub-heading ("### Day 1" under "## 4. Itinerary") stays in the section.
            if start is not None and depth <= level:
                return start, index
            heading = line.lstrip("#").strip().lower()
            if start is None and any(word in heading for word in words):
                start, level = index + 1, depth
    return (start, len(lines)) if start is not None else (0, 0)


def first_amount(text: str, offset: int) -> tuple[int, int, float, float] | None:
    """The first amount in text that isn't inside brackets: (start, end, low, high)."""
    depth = 0
    brackets = []
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        brackets.append(depth)

    for match in AMOUNT.finditer(text):
        if brackets[match.start()] == 0:
            low = to_number(match.group(1))
            high = to_number(match.group(2)) if match.group(2) else low
            return offset + match.start(), offset + match.end(), low, high
    return None


def budget_rows(plan: str) -> list[dict]:
    """
    The rows of the budget, from a Markdown table or a bullet list. Each row has its
    line number, label, amount (low, high) and where the amount is in the line, so
    it can be rewritten.
    """
    lines = plan.splitlines()
    start, end = section_lines(plan, ["budget"])
    rows = []

    for index in range(start, end):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("|"):
            cells, position = [], line.index("|") + 1
            for cell in line[position:].split("|")[:-1]:
                cells.append((cell, position))
                position += len(cell) + 1
            if not cells or set(cells[0][0].strip()) <= set("-: "):
                continue  # separator row
            label = cells[0][0].replace("*", "").strip()
            for cell, cell_start in cells[1:]:
                found = first_amount(cell, cell_start)
                if found:
                    rows.append({"line": index, "label": label, "start": found[0], "end": found[1],
                                 "low": found[2], "high": found[3]})
                    break

        elif stripped[:1] in "-*" and ":" in stripped:
            colon = line.index(":")
            label = line[:colon].replace("*", "").strip(" -")
            found = first_amount(line[colon + 1:], colon + 1)
            if found:
                rows.append({"line": index, "label": label, "start": found[0], "end": found[1],
                             "low": found[2], "high": found[3]})

    return rows


def set_amount(plan: str, row: dict, text: str) -> str:
    lines = plan.splitlines()
    line = lines[row["line"]]
    lines[row["line"]] = line[:row["start"]] + text + line[row["end"]:]
    return "\n".join(lines)


def check_budget(plan: str, flight_price: float | None) -> tuple[str, list[str]]:
    """Fixes the flight row and the total in code. Returns the plan and what was fixed."""
    fixes = []
    rows = budget_rows(plan)
    if not rows:
        return plan, fixes

    is_total = lambda row: "total" in row["label"].lower()
    is_flight = lambda row: "flight" in row["label"].lower() or "airfare" in row["label"].lower()

    # The chosen fare is the whole round trip, counted once.
    if flight_price:
        flight_rows = [row for row in rows if is_flight(row) and not is_total(row)]
        for row in flight_rows[:1]:
            if abs(row["low"] - flight_price) > 1 or abs(row["high"] - flight_price) > 1:
                twice = abs(row["low"] - 2 * flight_price) <= 1
                plan = set_amount(plan, row, format_amount(flight_price, flight_price))
                fixes.append(
                    f"The budget counted the flight twice; set it to {flight_price:,.0f} once"
                    if twice else f"Set the budget's flight cost to the chosen fare, {flight_price:,.0f}"
                )
        rows = budget_rows(plan)

    totals = [row for row in rows if is_total(row)]
    items = [row for row in rows if not is_total(row)]
    if totals and items:
        total = totals[-1]
        low = sum(row["low"] for row in items)
        high = sum(row["high"] for row in items)
        # The rows are whole numbers, so the total must match them (1 unit for rounding).
        if abs(total["low"] - low) > 1 or abs(total["high"] - high) > 1:
            plan = set_amount(plan, total, format_amount(low, high))
            fixes.append(
                f"Corrected the budget total from {format_amount(total['low'], total['high'])} "
                f"to {format_amount(low, high)} (the sum of the rows)"
            )

    return plan, fixes


# =========================
# Dates
# =========================

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def mentions_date(plan: str, iso_date: str) -> bool:
    """True if the plan names the date in any common form: 2026-11-22, 22 Nov, November 22nd."""
    if iso_date in plan:
        return True
    year, month, day = (int(part) for part in iso_date.split("-"))
    name = MONTHS[month - 1]
    month_names = f"(?:{name}|{name[:3]})"
    day_text = rf"0?{day}(?:st|nd|rd|th)?"
    pattern = rf"\b{day_text}\s+{month_names}\b|\b{month_names}\.?\s+{day_text}\b|\b{month:02d}/{day:02d}\b"
    return re.search(pattern, plan, re.I) is not None


# =========================
# Arrival day
# =========================

# A line that starts a part of the day: "- **Morning:**", "### Afternoon", "Evening -".
# Evening and night only mark where the afternoon ends; they're never too early.
DAY_PART = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*|#{2,6}\s*)?(morning|afternoon|evening|night)\b", re.I | re.M)
DAY_HEADING = re.compile(r"^.*\bDay\s*(\d+)\b.*$", re.I | re.M)
# Getting from the airport to the hotel is fine before landing time's day part ends.
LOGISTICS = re.compile(r"\b(arriv\w*|land\w*|airport|transfer|check[- ]?in|check into|hotel|settle)\b", re.I)


def day_one(plan: str) -> str:
    """The text of Day 1 in the itinerary, up to Day 2 (or the next section)."""
    start, end = section_lines(plan, ["itinerary", "day-by-day", "day by day"])
    lines = plan.splitlines()[start:end]
    block, inside = [], False
    for line in lines:
        heading = DAY_HEADING.match(line)
        if heading and len(line) < 120:
            if inside and heading.group(1) != "1":
                break
            inside = inside or heading.group(1) == "1"
        if inside:
            block.append(line)
    return "\n".join(block)


def check_arrival_day(plan: str, arrival_time: str | None) -> list[str]:
    """
    Day 1 can't have morning plans when the flight lands after noon, or afternoon
    plans when it lands after 5 pm.
    """
    if not arrival_time or " " not in arrival_time:
        return []
    try:
        hour = int(arrival_time.split(" ")[1].split(":")[0])
    except ValueError:
        return []

    # Each day part's text runs to the next day part. A part that's only about
    # arriving (airport, transfer, check-in) doesn't count as plans.
    block = day_one(plan)
    found = list(DAY_PART.finditer(block))
    parts = set()
    for index, match in enumerate(found):
        text = block[match.end():found[index + 1].start() if index + 1 < len(found) else len(block)]
        if not LOGISTICS.search(text):
            parts.add(match.group(1).lower())

    too_early = [part for part, limit in (("morning", 12), ("afternoon", 17)) if part in parts and hour >= limit]
    if not too_early:
        return []
    return [
        f"Day 1 has {' and '.join(too_early)} plans, but the flight lands at {arrival_time}. "
        "Start Day 1 after landing."
    ]


# =========================
# Citations
# =========================

# Capitalised words that aren't place names, so they don't count as evidence.
NOT_NAMES = {
    "the", "this", "these", "that", "there", "then", "and", "for", "from", "with", "near", "after",
    "day", "days", "morning", "afternoon", "evening", "night", "lunch", "dinner", "breakfast",
    "visit", "explore", "head", "enjoy", "take", "try", "walk", "return", "check", "spend", "stop",
    "grab", "have", "discover", "experience", "see", "start", "end", "relax", "consider", "use",
    "book", "optional", "note", "tip", "tips", "budget", "cost", "free", "entry", "price", "hotel",
    "hotels", "flight", "flights", "trip", "travel", "local", "option", "options", "aud", "usd",
    "food", "eat", "drink", "shopping", "dining", "transport", "train", "bus", "metro", "station",
    "line", "pass", "card", "airport", "arrival", "departure", "summary", "itinerary", "recommendations",
    "get", "buy", "sleep", "understand", "do", "around", "also", "many", "most", "some",
    # Labels the model puts before a citation, e.g. "**Excerpt Reference:** [4]".
    "excerpt", "excerpts", "reference", "references", "source", "sources", "guide", "wikivoyage",
    "citation", "cited", "based",
}

WORD = re.compile(r"\b[A-Z][\w'’-]{2,}")
CITATION = re.compile(r"\[(\d{1,2})\]")


def plain(text: str) -> str:
    """Lowercase without accents, so "Sensō-ji" matches "Senso-ji"."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def parse_excerpts(guide_context: str) -> dict[int, str]:
    """{number: body} from "[n] City, Country — Section\\nbody" blocks; the heading line is dropped."""
    excerpts = {}
    for block in re.split(r"\n\n(?=\[\d+\] )", guide_context or ""):
        match = re.match(r"\[(\d+)\] ([^\n]*)\n?(.*)", block, re.S)
        if match:
            excerpts[int(match.group(1))] = match.group(3)
    return excerpts


def check_citations(plan: str, excerpts: dict[int, str]) -> tuple[str, list[str]]:
    """
    Removes a citation [n] when n doesn't exist, or when the words just before it
    name a place and none of those names appear in excerpt n.
    """
    removed = 0
    out_lines = []

    for line in plan.splitlines():
        result, last = "", 0
        for match in CITATION.finditer(line):
            number = int(match.group(1))
            # Judge only the words since the previous citation or sentence end.
            claim = re.split(r"[.;!?]\s|\]", line[last:match.start()])[-1]
            names = [word for word in WORD.findall(claim) if plain(word) not in NOT_NAMES]
            body = plain(excerpts.get(number, ""))
            matches = any(plain(name).strip("'’-") in body for name in names)

            keep = number in excerpts and (not names or matches)
            result += line[last:match.start()] + (match.group(0) if keep else "")
            removed += not keep
            last = match.end()
        out_lines.append(result + line[last:])

    plan = "\n".join(out_lines)
    plan = re.sub(r"[ \t]+([.,;:])", r"\1", plan)  # "Temple ." left by a removed citation
    fixes = [f"Removed {removed} citation{'s' if removed > 1 else ''} that didn't match the guide excerpt"] if removed else []
    return plan, fixes


# =========================
# Entry point
# =========================

def review_plan(
    plan: str,
    guide_context: str = "",
    flight_price: float | None = None,
    travel_dates: list[str] | None = None,
    arrival_time: str | None = None,
) -> Review:
    review = Review(plan=plan)

    copied = copied_prompt(plan)
    if copied or len(plan) > MAX_PLAN_CHARS:
        review.problems.append(
            "The plan copies text from the prompt (search results or instructions). "
            "Write only the six sections, in your own words."
        )

    missing = missing_sections(plan)
    if missing:
        review.problems.append("Missing sections: " + ", ".join(missing) + ".")

    for iso_date in travel_dates or []:
        if not mentions_date(plan, iso_date):
            review.problems.append(f"The plan doesn't mention the travel date {iso_date}; build it on that date.")

    review.problems += check_arrival_day(plan, arrival_time)

    review.plan, fixes = check_budget(review.plan, flight_price)
    review.fixes += fixes

    if guide_context:
        review.plan, fixes = check_citations(review.plan, parse_excerpts(guide_context))
        review.fixes += fixes

    return review
