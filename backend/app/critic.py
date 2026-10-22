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


# Some models write non-breaking hyphens ("Day‑by‑Day") and narrow spaces ("Day 1",
# "1 354"). They look the same but break every pattern, so they become plain ones.
UNICODE_FIXES = str.maketrans({
    "\u2010": "-", "\u2011": "-",
    "\u00a0": " ", "\u2007": " ", "\u2009": " ", "\u200a": " ", "\u202f": " ",
})


def normalize(plan: str) -> str:
    return plan.translate(UNICODE_FIXES)


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


SECTION_WORDS = [word for words in REQUIRED_SECTIONS.values() for word in words]
# A heading the model wrote without "#": "**2. Flight Information**", "Flight Information:",
# or "**5. Estimated Budget** *(per person)*".
BARE_HEADING = re.compile(
    r"^(\*\*)?\s*(\d+[.)]\s*)?([A-Za-z][^*|:\n]{2,60}?)\s*:?\s*(\*\*)?\s*:?"
    r"(?:\s*\*?\([^)]*\)\*?)?\s*$"
)


def heading(line: str) -> tuple[int, str] | None:
    """(depth, lowercase text) if the line is a heading, else None."""
    if line.startswith("#"):
        return len(line) - len(line.lstrip("#")), line.lstrip("#").strip().lower()
    match = BARE_HEADING.match(line.strip())
    if not match:
        return None
    text = match.group(3).strip().lower()
    if any(word in text for word in SECTION_WORDS):
        return 2, text
    # Other bold lines ("**Day 1**") are sub-headings inside a section.
    return (3, text) if match.group(1) else None


def headings(plan: str) -> list[str]:
    return [found[1] for found in map(heading, plan.splitlines()) if found]


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

# A money amount: "1,354", "1 354", "360–540", "2,114 - 3,444", "600 to 800".
NUMBER = r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
AMOUNT = re.compile(rf"({NUMBER})(?:\s*(?:–|—|-|to)\s*({NUMBER}))?")

# A cell the critic can't read safely: alternatives ("hotel: 540 <br> hostel: 180") or
# several sums. The budget check then says nothing rather than "fixing" the wrong number.
AMBIGUOUS = "ambiguous"


def to_number(text: str) -> float:
    return float(text.replace(",", "").replace(" ", ""))


def format_amount(low: float, high: float) -> str:
    if round(low) == round(high):
        return f"{round(low):,}"
    return f"{round(low):,}–{round(high):,}"


def section_lines(plan: str, words: list[str]) -> tuple[int, int]:
    """Line range (start, end) of the first section whose heading has one of the words."""
    lines = plan.splitlines()
    start, level = None, 0
    for index, line in enumerate(lines):
        found = heading(line)
        if found:
            depth, text = found
            # A sub-heading ("### Day 1" under "## 4. Itinerary") stays in the section.
            if start is not None and depth <= level:
                return start, index
            if start is None and any(word in text for word in words):
                start, level = index + 1, depth
    return (start, len(lines)) if start is not None else (0, 0)


def first_amount(text: str, offset: int) -> tuple[int, int, float, float] | str | None:
    """
    The amount in a cell, outside brackets: (start, end, low, high). For a sum like
    "3 × 180 = 540" it's the number after "=". AMBIGUOUS for alternatives or several sums.
    """
    depth = 0
    brackets = []
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        brackets.append(depth)

    equals = [index for index, char in enumerate(text) if char == "=" and brackets[index] == 0]
    matches = [match for match in AMOUNT.finditer(text) if brackets[match.start()] == 0]
    if len(equals) > 1 or ("<br" in text.lower() and len(matches) > 1):
        return AMBIGUOUS
    if equals:
        matches = [match for match in matches if match.start() > equals[0]]

    for match in matches:
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
                if found == AMBIGUOUS:
                    rows.append({"line": index, "label": label, "ambiguous": True})
                    break
                if found:
                    rows.append({"line": index, "label": label, "start": found[0], "end": found[1],
                                 "low": found[2], "high": found[3]})
                    break

        elif stripped[:1] in "-*" and ":" in stripped:
            colon = line.index(":")
            label = line[:colon].replace("*", "").strip(" -")
            found = first_amount(line[colon + 1:], colon + 1)
            if found == AMBIGUOUS:
                rows.append({"line": index, "label": label, "ambiguous": True})
            elif found:
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
        flight_rows = [row for row in rows if is_flight(row) and not is_total(row) and not row.get("ambiguous")]
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
    # Several totals (one per hotel option) or a row with alternatives: the rows don't
    # add up to one total, so don't check it.
    readable = len(totals) == 1 and not any(row.get("ambiguous") for row in rows)
    if readable and items:
        total = totals[0]
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

    # A day part's text is its own line plus the lines indented under it, so timed
    # bullets after it ("**18:30** - Land at Narita") don't count as its plans.
    # A part that's only about arriving (airport, transfer, check-in) isn't plans.
    lines = day_one(plan).splitlines()
    parts = set()
    for index, line in enumerate(lines):
        match = DAY_PART.match(line)
        if not match:
            continue
        indent = len(line) - len(line.lstrip())
        text = [line[match.end():]]
        for below in lines[index + 1:]:
            if below.strip() and len(below) - len(below.lstrip()) <= indent:
                break
            text.append(below)
        if not LOGISTICS.search("\n".join(text)):
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
    # Dates are capitalised but aren't places.
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
}

WORD = re.compile(r"\b[A-Z][\w'’-]{2,}")
CITATION = re.compile(r"\[(\d{1,2})\]")


def plain(text: str) -> str:
    """Lowercase without accents, so "Sensō-ji" matches "Senso-ji"."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def excerpt_places(guide_context: str) -> set[str]:
    """
    Words from the excerpts' headings ("Istanbul, Turkey — Eat"). The body leaves out
    the city's name, so "iconic Istanbul street food [7]" says nothing about excerpt 7.
    """
    headings = re.findall(r"^\[\d+\] ([^\n]*)", guide_context or "", re.M)
    return {plain(word) for heading in headings for word in re.findall(r"[\w'’-]+", heading)}


def parse_excerpts(guide_context: str) -> dict[int, str]:
    """{number: body} from "[n] City, Country — Section\\nbody" blocks; the heading line is dropped."""
    excerpts = {}
    for block in re.split(r"\n\n(?=\[\d+\] )", guide_context or ""):
        match = re.match(r"\[(\d+)\] ([^\n]*)\n?(.*)", block, re.S)
        if match:
            excerpts[int(match.group(1))] = match.group(3)
    return excerpts


# Where a clause ends: sentence, comma, colon, dash, an earlier citation or a table cell.
CLAUSE_END = re.compile(r"[.;:!?,]\s|\]|\s[–—-]\s|\|")


def claim_names(text_before: str, ignore: set[str] = frozenset()) -> list[str]:
    """
    The place names in the clause right before a citation. Judging only this clause
    keeps the check stable when it runs again after removing a citation.
    """
    claim = CLAUSE_END.split(text_before)[-1]
    names = [
        re.sub(r"['’]s$", "", word)  # "Gaudí’s" -> "Gaudí"
        for word in WORD.findall(claim)
        # Acronyms like NYC or BTS are usually written out in the guide.
        if plain(word) not in NOT_NAMES and not (word.isupper() and len(word) <= 5)
        and plain(re.sub(r"['’]s$", "", word)) not in ignore
    ]

    # A capitalised first word is usually just the start of a sentence ("Carry a
    # water bottle"), unless it looks like a name: "Senso-ji", "Central Park".
    start = re.match(r"[\s*_>#•\-\d.)]*([A-Z][\w'’-]*)\s*(\S*)", claim)
    if start and names and names[0] == re.sub(r"['’]s$", "", start.group(1)):
        first, following = start.group(1), start.group(2).lstrip("*_")
        if "-" not in first and first.isascii() and not following[:1].isupper():
            names = names[1:]
    return names


def mentions(body: str, name: str) -> bool:
    """Whether an excerpt names this place, ignoring accents, case and hyphens."""
    spaced = lambda text: plain(text).replace("-", " ")
    return spaced(name).strip("'’ ") in spaced(body)


def check_citations(plan: str, excerpts: dict[int, str], ignore: set[str] = frozenset()) -> tuple[str, list[str]]:
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
            names = claim_names(line[last:match.start()], ignore)
            body = excerpts.get(number, "")
            matches = any(mentions(body, name) for name in names)

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
    plan = normalize(plan)
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
        review.plan, fixes = check_citations(review.plan, parse_excerpts(guide_context), excerpt_places(guide_context))
        review.fixes += fixes

    return review
