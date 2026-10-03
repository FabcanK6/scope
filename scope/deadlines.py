"""Reporting deadlines, checked by code: the AI model copies the dates from the note, SCOPE does the counting.

Language models are unreliable at calendar arithmetic ("Thursday to Monday is how many business days?"), so for
topics with a deadline in the study profile (e.g. SAEs within 24 hours, or within 2 business days of awareness under
one protocol and 3 calendar days under another) SCOPE:

1. takes the date the clock starts (``clock_start``, e.g. site awareness) and the date of the report
   (``reported_on``) that the model copied from the note, and checks both are really in the note;
2. counts the elapsed time in the unit the protocol uses (calendar days, business days Monday-Friday, or hours);
3. decides on time / late itself, and overrides the model's status and severity when it disagrees.

Public holidays are not known to SCOPE; a business-day count is Monday to Friday.
"""

from __future__ import annotations

import datetime as _dt
import re

UNITS = ["hours", "calendar_days", "business_days"]
UNIT_WORDS = {"hours": "hours", "calendar_days": "calendar days", "business_days": "business days"}

_WEEKDAY = re.compile(r"\b(mon|tues|wednes|thurs|fri|satur|sun)day\b,?", re.I)
_ORDINAL = re.compile(r"\b(\d{1,2})(st|nd|rd|th)\b", re.I)
_FORMATS = ["%d %B %Y", "%B %d %Y", "%d %b %Y", "%b %d %Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d-%B-%Y",
            "%d.%m.%Y", "%m/%d/%y", "%d%b%Y", "%d %B", "%B %d", "%d %b", "%b %d", "%d-%b", "%m/%d"]


def parse_date(text: str | None, default_year: int | None = None) -> _dt.date | None:
    """'Thursday 1 October 2026', 'Oct 1st, 2026', '01-Oct-2026', '10/01/2026', '1 Oct' (+ default_year)."""
    if not text:
        return None
    t = _WEEKDAY.sub("", str(text))
    t = _ORDINAL.sub(r"\1", t)
    t = re.sub(r"\b(on|the|of)\b", " ", t, flags=re.I)
    t = re.sub(r"[,]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    for fmt in _FORMATS:
        try:
            d = _dt.datetime.strptime(t, fmt)
        except ValueError:
            continue
        if "%Y" not in fmt and "%y" not in fmt:
            if default_year is None:
                return None
            d = d.replace(year=default_year)
        return d.date()
    return None


def business_days_between(start: _dt.date, end: _dt.date) -> int:
    """Business days after ``start`` up to and including ``end`` (Mon-Fri). Thursday -> Monday = 2."""
    days, d = 0, start
    while d < end:
        d += _dt.timedelta(days=1)
        if d.weekday() < 5:
            days += 1
    return days


def check(start: _dt.date, reported: _dt.date, amount: float, unit: str) -> dict:
    """{'verdict': 'on time' | 'late' | 'unclear', 'elapsed': n, 'unit': unit, 'allowed': amount}.

    With dates only (no times), an hours deadline is decided only when it is clear: same day or within the full days
    allowed is on time; more than the allowed days plus one is late; anything in between is unclear."""
    if reported < start:
        return {"verdict": "unclear", "elapsed": None, "unit": unit, "allowed": amount,
                "why": "the report date is before the start date"}
    if unit == "business_days":
        elapsed = business_days_between(start, reported)
        verdict = "on time" if elapsed <= amount else "late"
    elif unit == "calendar_days":
        elapsed = (reported - start).days
        verdict = "on time" if elapsed <= amount else "late"
    else:  # hours, but the note gives dates only
        days = (reported - start).days
        elapsed = days * 24
        full_days = int(amount // 24)
        if days <= max(0, full_days - 1) or days == 0:
            verdict = "on time"
        elif days > full_days:
            verdict = "late"
        else:
            verdict = "unclear"  # e.g. 24 hours, reported the next day: depends on the time of day
    return {"verdict": verdict, "elapsed": elapsed, "unit": unit, "allowed": amount}


def describe(result: dict, topic_display: str) -> str:
    unit = UNIT_WORDS.get(result["unit"], result["unit"])
    allowed = f"{result['allowed']:g} {unit}"
    if result["verdict"] == "unclear":
        return (f"{topic_display}: SCOPE could not tell from the dates alone whether the report met the "
                f"{allowed} deadline (times are needed). Check it yourself.")
    elapsed = (f"{result['elapsed'] // 24} day(s)" if result["unit"] == "hours" else f"{result['elapsed']} {unit}")
    return (f"{topic_display}: reported after {elapsed}; this study allows {allowed}, so it was "
            f"{result['verdict']} (dates checked by SCOPE).")
