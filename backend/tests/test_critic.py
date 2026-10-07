"""
Critic checks on plans that reproduce real failures from test runs.

Run from the backend folder:
    python -m unittest tests.test_critic
"""

import unittest

from app.critic import (
    check_arrival_day, check_budget, check_citations, mentions_date, parse_excerpts, review_plan,
)

SECTIONS = """# Tokyo trip

## 1. Trip Summary
Three days in Tokyo, leaving 2026-11-22 and returning 2026-11-24.

## 2. Flight Information
China Eastern, AUD 1,354 round trip.

## 3. Hotel Suggestions
A mid-range hotel in Shinjuku.

## 4. Day-by-Day Itinerary
Day 1: Asakusa.

## 6. Final Recommendations
Get a Suica card.
"""

# Tokyo run, Stage A: the round-trip fare (AUD 1,354) was counted twice.
DOUBLE_FLIGHT = SECTIONS + """
## 5. Estimated Budget

| **Category**         | **Estimated Cost (AUD)** |
|----------------------|--------------------------|
| Flights (Round Trip) | 2,708                    |
| Hotel (3 Nights)     | 1,500 (Mid-Range)        |
| Food & Drinks        | 500                      |
| Transportation       | 300                      |
| Attractions & Tours  | 300                      |
| **Total**            | **5,308**                |
"""

# Tokyo run, Stage C: the top of the total range should be 2,494, not 3,444.
WRONG_TOTAL = SECTIONS + """
## 5. Estimated Budget

| Category         | Estimated Cost (AUD) |
|------------------|----------------------|
| Flights          | 1354                 |
| Accommodation    | 360–540 (3 nights)   |
| Food             | 200–300              |
| Transportation   | 100–150              |
| Miscellaneous    | 100–150              |
| **Total**        | **2114–3444**        |
"""

BULLET_BUDGET = SECTIONS + """
## 5. Estimated Budget
- **Flights:** AUD 1,354
- **Hotel:** AUD 600 (3 nights)
- **Total:** AUD 1,500
"""

EXCERPTS = """[1] Tokyo, Japan — See
Sensō-ji is the oldest temple in Tokyo, in the Asakusa district.

[2] Tokyo, Japan — Eat
Tsukiji Outer Market has many small sushi stalls."""


def budget_line(plan: str, label: str) -> str:
    return next(line for line in plan.splitlines() if label in line)


class BudgetTests(unittest.TestCase):
    def test_flight_counted_twice_is_set_once_and_total_follows(self):
        plan, fixes = check_budget(DOUBLE_FLIGHT, flight_price=1354)
        self.assertIn("1,354", budget_line(plan, "Flights"))
        self.assertIn("3,954", budget_line(plan, "Total"))
        self.assertTrue(any("twice" in fix for fix in fixes))

    def test_wrong_range_total_is_recomputed(self):
        plan, fixes = check_budget(WRONG_TOTAL, flight_price=1354)
        self.assertIn("2,114–2,494", budget_line(plan, "Total"))
        self.assertEqual(len(fixes), 1)

    def test_correct_budget_is_left_alone(self):
        plan, fixes = check_budget(check_budget(WRONG_TOTAL, 1354)[0], 1354)
        self.assertEqual(fixes, [])

    def test_total_off_by_rounding_is_corrected(self):
        # Browser run, Stage E: the rows added up to 9,551 but the total said 9,600.
        plan = WRONG_TOTAL.replace("**2114–3444**", "**2114–2500**")
        fixed, _ = check_budget(plan, flight_price=1354)
        self.assertIn("2,114–2,494", budget_line(fixed, "Total"))

    def test_bullet_list_budget(self):
        plan, fixes = check_budget(BULLET_BUDGET, flight_price=1354)
        self.assertIn("1,954", budget_line(plan, "Total"))

    def test_brackets_are_not_amounts(self):
        # "(3 nights)" must not be read as 3.
        self.assertIn("360–540 (3 nights)", check_budget(WRONG_TOTAL, 1354)[0])


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.excerpts = parse_excerpts(EXCERPTS)

    def test_parse_drops_heading_line(self):
        self.assertEqual(set(self.excerpts), {1, 2})
        self.assertNotIn("Japan", self.excerpts[1])

    def test_matching_citation_kept_even_with_accents(self):
        plan, fixes = check_citations("Visit Senso-ji in Asakusa [1].", self.excerpts)
        self.assertIn("[1]", plan)
        self.assertEqual(fixes, [])

    def test_hotel_from_web_search_cited_to_guide_is_removed(self):
        # Tokyo run, Phase 2: a hotel found by Tavily was cited as excerpt [1].
        plan, fixes = check_citations("Stay at The Gate Hotel Kaminarimon [1].", self.excerpts)
        self.assertEqual(plan, "Stay at The Gate Hotel Kaminarimon.")
        self.assertEqual(len(fixes), 1)

    def test_citation_to_missing_excerpt_is_removed(self):
        plan, _ = check_citations("Eat at Tsukiji Outer Market [7].", self.excerpts)
        self.assertNotIn("[7]", plan)

    def test_claim_without_names_is_kept(self):
        plan, _ = check_citations("Try the fresh sushi stalls [2].", self.excerpts)
        self.assertIn("[2]", plan)

    def test_label_before_citation_is_not_a_place_name(self):
        # Browser run, Stage E: the model put citations on their own lines.
        plan, fixes = check_citations("- **Excerpt Reference:** [1]", self.excerpts)
        self.assertIn("[1]", plan)
        self.assertEqual(fixes, [])

    def test_each_citation_judged_on_its_own_words(self):
        line = "See Senso-ji [1] and eat at Tsukiji [2], then the Gate Hotel [2]."
        plan, _ = check_citations(line, self.excerpts)
        self.assertEqual(plan, "See Senso-ji [1] and eat at Tsukiji [2], then the Gate Hotel.")


class ContentTests(unittest.TestCase):
    def test_dates_in_any_common_form(self):
        self.assertTrue(mentions_date("Leave on 22 Nov", "2026-11-22"))
        self.assertTrue(mentions_date("Leave on November 22nd", "2026-11-22"))
        self.assertTrue(mentions_date("Leave on 2026-11-22", "2026-11-22"))
        self.assertFalse(mentions_date("Leave on 2 Nov", "2026-11-22"))

    def test_good_plan_has_no_problems(self):
        review = review_plan(WRONG_TOTAL, EXCERPTS, 1354, ["2026-11-22", "2026-11-24"])
        self.assertEqual(review.problems, [])
        self.assertEqual(len(review.fixes), 1)

    def test_missing_section_wrong_date_and_copied_prompt(self):
        plan = SECTIONS + "\nTravel Guide Excerpts (from Wikivoyage, numbered):\n[1] ..."
        review = review_plan(plan, "", None, ["2026-12-01"])
        text = " ".join(review.problems)
        self.assertIn("Estimated Budget", text)
        self.assertIn("2026-12-01", text)
        self.assertIn("copies text from the prompt", text)


EVENING_ARRIVAL = """## 4. Day-by-Day Itinerary

### Day 1: Arrival in Asakusa
- **Morning:** Visit Senso-ji.
- **Afternoon:** Walk Nakamise Street.
- **Evening:** Dinner near the hotel.

### Day 2: Shinjuku
- **Morning:** Gyoen garden.

## 5. Estimated Budget
"""


class ArrivalDayTests(unittest.TestCase):
    def test_evening_arrival_with_morning_plans(self):
        # Tokyo run, Stage A: Day 1 had morning activities but the flight landed at 20:35.
        problems = check_arrival_day(EVENING_ARRIVAL, "2026-11-20 20:35")
        self.assertEqual(len(problems), 1)
        self.assertIn("morning and afternoon", problems[0])

    def test_lunchtime_arrival_allows_afternoon(self):
        problems = check_arrival_day(EVENING_ARRIVAL.replace("- **Morning:** Visit Senso-ji.\n", ""), "2026-11-20 13:00")
        self.assertEqual(problems, [])

    def test_morning_arrival_is_fine(self):
        self.assertEqual(check_arrival_day(EVENING_ARRIVAL, "2026-11-20 07:10"), [])

    def test_getting_to_the_hotel_is_not_a_plan(self):
        # Browser run, 3-days fix: landing 18:10, "Afternoon: head to Asakusa and check
        # into your hotel" was flagged and cost a 4-minute rewrite.
        plan = EVENING_ARRIVAL.replace(
            "- **Morning:** Visit Senso-ji.\n- **Afternoon:** Walk Nakamise Street.\n",
            "- **Afternoon:** Head to Asakusa and check into your hotel.\n",
        )
        self.assertEqual(check_arrival_day(plan, "2026-11-20 18:10"), [])

    def test_day_two_morning_is_not_day_one(self):
        plan = EVENING_ARRIVAL.replace("- **Morning:** Visit Senso-ji.\n- **Afternoon:** Walk Nakamise Street.\n", "")
        self.assertEqual(check_arrival_day(plan, "2026-11-20 20:35"), [])


if __name__ == "__main__":
    unittest.main()
