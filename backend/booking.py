"""
booking.py
Robust Cal.com v2 booking flow for chat + voice.

Uses:
- Cal.com for real slots and bookings
- Groq as a meaning-extraction layer
- Groq contact extraction for spoken emails
- Backend state validation before any real booking/cancellation

Voice improvements:
- User-facing slot labels do not say "IST"
- Slot labels are spoken naturally: Monday, June 8th at 5 PM
- After slots are shown, unclear speech does not trigger random new slot fetching
- User must choose one of shown options or ask for more slots
"""

import os
import re
import json
import httpx
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass, field
from dotenv import load_dotenv
from llm_client import call_groq_json

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────

CAL_API_KEY = os.environ["CAL_API_KEY"]
CAL_EVENT_TYPE_ID = os.environ["CAL_EVENT_TYPE_ID"]
CAL_USERNAME = os.environ["CAL_USERNAME"]

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CONTACT_EXTRACTION_MODEL = os.getenv("CONTACT_EXTRACTION_MODEL", "llama-3.1-8b-instant")

CAL_API_BASE = "https://api.cal.com/v2"
CAL_SLOTS_API_VERSION = "2024-09-04"
CAL_BOOKINGS_API_VERSION = "2026-02-25"

SLOT_DAYS_AHEAD = 14
MAX_SLOTS_SHOWN = 3

TIMEZONE_NAME = "Asia/Kolkata"
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)

# Keep timezone internally for Cal.com, but do not speak "IST" to users.
SHOW_TIMEZONE_IN_USER_MESSAGES = False


# ── STAGES ───────────────────────────────────────────────────────────────────

STAGE_STARTED = "started"
STAGE_ASKING_PREFERENCE = "asking_preference"
STAGE_SHOWING_SLOTS = "showing_slots"
STAGE_COLLECTING_INFO = "collecting_info"
STAGE_AWAITING_CONFIRMATION = "awaiting_confirmation"
STAGE_CONFIRMED = "confirmed"

STAGE_CANCEL_REASON = "cancel_reason"
STAGE_CANCEL_CONFIRMING = "cancel_confirming"


@dataclass
class BookingSession:
    stage: str = STAGE_STARTED

    all_slots: list[dict] = field(default_factory=list)
    filtered_slots: list[dict] = field(default_factory=list)
    pending_slots: list[dict] = field(default_factory=list)
    shown_offset: int = 0
    chosen_slot: dict = field(default_factory=dict)

    guest_name: str = ""
    guest_email: str = ""
    awaiting_email_confirmation: bool = False

    confirmed_booking_uid: str = ""
    confirmed_label: str = ""
    confirmed_guest_email: str = ""

    cancellation_reason: str = ""


sessions: dict[str, BookingSession] = {}


def get_session(session_id: str) -> BookingSession:
    if session_id not in sessions:
        sessions[session_id] = BookingSession()
    return sessions[session_id]


def reset_session(session_id: str) -> BookingSession:
    sessions[session_id] = BookingSession()
    return sessions[session_id]


def clear_session(session_id: str):
    sessions.pop(session_id, None)


BOOKING_KEYWORDS = [
    "book", "schedule", "interview", "call", "meeting",
    "availability", "available", "slot", "calendar",
    "set up a time", "connect", "appointment", "talk",
    "reschedule", "cancel"
]


def is_booking_intent(message: str) -> bool:
    msg = message.lower()
    return any(keyword in msg for keyword in BOOKING_KEYWORDS)


def is_in_booking_flow(session_id: str) -> bool:
    session = sessions.get(session_id)
    return session is not None and session.stage != STAGE_CONFIRMED


# ── TIME HELPERS ──────────────────────────────────────────────────────────────

DAY_MAP = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def now_ist() -> datetime:
    return datetime.now(timezone.utc) + TIMEZONE_OFFSET


def to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc) + TIMEZONE_OFFSET


def ordinal_day(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def format_slot_label(dt: datetime) -> str:
    """
    Human-friendly slot label for chat/voice.
    Example: Monday, June 8th at 5 PM
    """
    ist_dt = to_ist(dt)

    weekday = ist_dt.strftime("%A")
    month = ist_dt.strftime("%B")
    day = ordinal_day(ist_dt.day)

    hour = ist_dt.strftime("%I").lstrip("0")
    minute = ist_dt.strftime("%M")
    ampm = ist_dt.strftime("%p")

    if minute == "00":
        time_text = f"{hour} {ampm}"
    else:
        time_text = f"{hour}:{minute} {ampm}"

    label = f"{weekday}, {month} {day} at {time_text}"

    if SHOW_TIMEZONE_IN_USER_MESSAGES:
        label += " IST"

    return label


def parse_preference_from_text(preference_text: str) -> dict:
    msg = (preference_text or "").lower()
    current_ist = now_ist()

    preferred_dates: list[date] = []
    preferred_days: list[int] = []
    strict_date = False

    if "day after tomorrow" in msg:
        preferred_dates.append((current_ist + timedelta(days=2)).date())
        strict_date = True
    elif "tomorrow" in msg:
        preferred_dates.append((current_ist + timedelta(days=1)).date())
        strict_date = True
    elif "today" in msg:
        preferred_dates.append(current_ist.date())
        strict_date = True

    for word, weekday_index in DAY_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", msg):
            preferred_days.append(weekday_index)

    period = None
    if any(w in msg for w in ["morning", "am", "forenoon", "before lunch"]):
        period = "morning"
    elif any(w in msg for w in ["afternoon", "lunch", "midday", "mid-day"]):
        period = "afternoon"
    elif any(w in msg for w in ["evening", "late", "after 5", "after 6", "pm"]):
        period = "evening"

    return {
        "dates": preferred_dates,
        "days": list(set(preferred_days)),
        "period": period,
        "strict_date": strict_date,
        "raw": preference_text,
    }


def slot_matches_preference(slot_dt: datetime, prefs: dict) -> bool:
    ist_dt = to_ist(slot_dt)

    if prefs.get("dates") and ist_dt.date() not in prefs["dates"]:
        return False

    if prefs.get("days") and ist_dt.weekday() not in prefs["days"]:
        return False

    period = prefs.get("period")
    hour = ist_dt.hour

    if period == "morning" and not (7 <= hour < 12):
        return False
    if period == "afternoon" and not (12 <= hour < 17):
        return False
    if period == "evening" and not (17 <= hour < 21):
        return False

    return True


# ── CAL.COM API ───────────────────────────────────────────────────────────────

def slots_headers() -> dict:
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": CAL_SLOTS_API_VERSION,
    }


def bookings_headers() -> dict:
    return {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": CAL_BOOKINGS_API_VERSION,
        "Content-Type": "application/json",
    }


def parse_slot_start(slot: dict | str) -> str | None:
    if isinstance(slot, str):
        return slot
    if isinstance(slot, dict):
        return slot.get("start") or slot.get("time")
    return None


async def fetch_all_slots() -> list[dict]:
    now = datetime.now(timezone.utc)
    start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now + timedelta(days=SLOT_DAYS_AHEAD)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
        "start": start,
        "end": end,
        "timeZone": TIMEZONE_NAME,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{CAL_API_BASE}/slots",
            headers=slots_headers(),
            params=params,
            timeout=15
        )

        if response.status_code >= 400:
            raise Exception(f"Cal slots error {response.status_code}: {response.text}")

        data = response.json()

    raw_slots = data.get("data", {})
    iterable = []

    if isinstance(raw_slots, dict):
        for _, day_slots in raw_slots.items():
            if isinstance(day_slots, list):
                iterable.extend(day_slots)
    elif isinstance(raw_slots, list):
        iterable = raw_slots

    slots = []

    for slot in iterable:
        start_iso = parse_slot_start(slot)

        if not start_iso:
            continue

        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        label = format_slot_label(start_dt)

        slots.append({
            "start": start_iso,
            "label": label,
            "dt": start_dt,
        })

    slots.sort(key=lambda s: s["dt"])
    return slots


async def create_booking(slot_start: str, guest_name: str, guest_email: str) -> dict:
    payload = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
        "start": slot_start,
        "attendee": {
            "name": guest_name,
            "email": guest_email,
            "timeZone": TIMEZONE_NAME,
            "language": "en"
        },
        "metadata": {
            "source": "scaler-ai-persona"
        }
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAL_API_BASE}/bookings",
            headers=bookings_headers(),
            json=payload,
            timeout=15
        )

        if response.status_code >= 400:
            raise Exception(f"Cal booking error {response.status_code}: {response.text}")

        return response.json()


async def cancel_booking(booking_uid: str, reason: str) -> dict:
    payload = {
        "cancellationReason": reason or "Cancelled through Gaurav AI persona"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAL_API_BASE}/bookings/{booking_uid}/cancel",
            headers=bookings_headers(),
            json=payload,
            timeout=15
        )

        if response.status_code >= 400:
            raise Exception(f"Cal cancel error {response.status_code}: {response.text}")

        return response.json()


# ── JSON HELPERS ──────────────────────────────────────────────────────────────

def safe_json_loads(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return {}

    return {}


def slot_snapshot(slots: list[dict]) -> list[dict]:
    return [
        {
            "number": i,
            "label": s["label"],
            "start": s["start"]
        }
        for i, s in enumerate(slots, 1)
    ]


# ── GROQ BOOKING PARSER ───────────────────────────────────────────────────────

def parse_booking_message(user_message: str, session: BookingSession) -> dict:
    system_prompt = """
You are a strict booking-message parser for an AI scheduling backend.

Return ONLY valid JSON. No explanation. No markdown.

The current_stage is the most important signal.
Interpret the user message according to the current_stage.

Possible intents:
- provide_preference: user gives preferred day/time
- select_slot: user chooses one of the shown slots
- provide_contact: user gives attendee name and/or email
- confirm_booking: user confirms final booking OR confirms cancellation when stage is cancel_confirming
- deny_or_change: user rejects, wants to change, or refuses cancellation
- show_more_slots: user asks for other slots
- cancel_flow: user wants to stop the current unconfirmed booking flow
- cancel_confirmed_booking: user wants to cancel an already confirmed calendar booking
- provide_cancellation_reason: user gives reason for cancelling a confirmed booking
- restart_flow: user wants to start booking again
- unknown: unclear

Return this exact schema:
{
  "intent": "one_of_the_intents",
  "preference_text": null_or_string,
  "slot_number": null_or_integer,
  "slot_reference": null_or_string,
  "name": null_or_string,
  "email": null_or_string,
  "confirmation": null_or_boolean,
  "cancellation_reason": null_or_string
}

Stage rules:
- cancel_reason: treat message as cancellation reason unless user clearly refuses
- cancel_confirming: yes/agree -> confirm_booking + confirmation true; no/refuse -> deny_or_change + confirmation false
- awaiting_confirmation: yes -> confirm_booking + confirmation true; no/change -> deny_or_change + confirmation false
- showing_slots: choose a shown option -> select_slot. Ask for more -> show_more_slots. If unclear -> unknown.
- collecting_info: extract name/email -> provide_contact. If confirming heard email -> confirm_booking.
- asking_preference: treat as availability preference -> provide_preference
- confirmed: cancel meeting -> cancel_confirmed_booking; new booking -> restart_flow

General:
- Do not invent email, name, slot number, confirmation, or cancellation reason.
- If the user gives a spoken email, convert it to normal email only if you are confident.
- If unclear return unknown.
""".strip()

    user_payload = {
        "current_stage": session.stage,
        "user_message": user_message,
        "shown_slots": slot_snapshot(session.pending_slots),
        "chosen_slot": session.chosen_slot.get("label") if session.chosen_slot else None,
        "known_name": session.guest_name or None,
        "known_email": session.guest_email or None,
        "awaiting_email_confirmation": session.awaiting_email_confirmation,
        "confirmed_booking": {
            "uid": session.confirmed_booking_uid or None,
            "label": session.confirmed_label or None,
            "email": session.confirmed_guest_email or None,
        },
        "known_cancellation_reason": session.cancellation_reason or None,
    }

    try:
        parsed = call_groq_json(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0,
            max_tokens=300,
        )

    except Exception as e:
        print(f"[booking] parse_booking_message failed: {e}")
        parsed = {}

    intent = parsed.get("intent") or "unknown"

    allowed = {
        "provide_preference",
        "select_slot",
        "provide_contact",
        "confirm_booking",
        "deny_or_change",
        "show_more_slots",
        "cancel_flow",
        "cancel_confirmed_booking",
        "provide_cancellation_reason",
        "restart_flow",
        "unknown",
    }

    if intent not in allowed:
        intent = "unknown"

    return {
        "intent": intent,
        "preference_text": parsed.get("preference_text"),
        "slot_number": parsed.get("slot_number"),
        "slot_reference": parsed.get("slot_reference"),
        "name": parsed.get("name"),
        "email": parsed.get("email"),
        "confirmation": parsed.get("confirmation"),
        "cancellation_reason": parsed.get("cancellation_reason"),
    }


# ── CONTACT EXTRACTION USING GROQ ─────────────────────────────────────────────

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def is_valid_email(email: str) -> bool:
    if not email:
        return False

    email = email.strip().lower()

    if not EMAIL_PATTERN.fullmatch(email):
        return False

    if ".." in email:
        return False

    if email.startswith(".") or email.endswith("."):
        return False

    return True


def clean_name(name: str | None) -> str:
    if not name:
        return ""

    name = re.sub(r"[^A-Za-z\s.'-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name[:80]


def clean_reason(reason: str | None) -> str:
    if not reason:
        return ""

    reason = re.sub(r"\s+", " ", reason.strip())
    return reason[:250]


def extract_contact_with_groq(user_message: str, known_name: str = "", known_email: str = "") -> dict:
    system_prompt = """
You extract contact details for a calendar booking.

Return ONLY valid JSON:
{
  "name": null_or_string,
  "email": null_or_string,
  "email_confidence": "high" | "medium" | "low",
  "needs_email_repeat": boolean
}

Rules:
- Extract the attendee's full name if present.
- Extract and normalize the email address if the user provides it in typed or spoken form.
- Spoken examples may include: "at", "at the rate", "dot", "gmail dot com", numbers spoken as words.
- Convert spoken email to a normal email ONLY if you are confident.
- If the email is incomplete, ambiguous, or missing the username/domain, return email null and needs_email_repeat true.
- Do not invent missing parts.
- Do not guess a Gmail address unless the user clearly said it.
- If an already known name/email is supplied, keep it unless the user clearly changes it.
""".strip()

    payload = {
        "user_message": user_message,
        "known_name": known_name or None,
        "known_email": known_email or None,
    }

    try:
        parsed = call_groq_json(
            model=CONTACT_EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0,
            max_tokens=120,
        )

    except Exception as e:
        print(f"[booking] contact extraction failed: {e}")
        parsed = {}

    name = clean_name(parsed.get("name") or known_name)
    email = (parsed.get("email") or known_email or "").strip().lower()
    email_confidence = parsed.get("email_confidence") or "low"
    needs_email_repeat = bool(parsed.get("needs_email_repeat", False))

    if not is_valid_email(email):
        email = ""
        needs_email_repeat = True

    return {
        "name": name,
        "email": email,
        "email_confidence": email_confidence,
        "needs_email_repeat": needs_email_repeat,
    }


# ── SLOT MATCHING ─────────────────────────────────────────────────────────────

def match_slot_by_number(slot_number: int | None, pending_slots: list[dict]) -> dict | None:
    if slot_number is None:
        return None

    try:
        index = int(slot_number) - 1
    except Exception:
        return None

    if 0 <= index < len(pending_slots):
        return pending_slots[index]

    return None


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9: ]", " ", text.lower())


def match_slot_by_reference(reference: str | None, pending_slots: list[dict]) -> dict | None:
    if not reference or not pending_slots:
        return None

    ref = normalize_for_match(reference)
    scored = []

    for slot in pending_slots:
        label = normalize_for_match(slot["label"])
        score = sum(1 for token in ref.split() if token and token in label)

        ist_dt = to_ist(slot["dt"])
        hour12 = ist_dt.hour % 12 or 12
        minute = ist_dt.minute

        time_patterns = [
            f"{hour12}",
            f"{hour12} pm" if ist_dt.hour >= 12 else f"{hour12} am",
            f"{hour12}:{minute:02d}",
            f"{hour12}:{minute:02d} pm" if ist_dt.hour >= 12 else f"{hour12}:{minute:02d} am",
        ]

        if any(p in ref for p in time_patterns):
            score += 3

        scored.append((score, slot))

    scored = [(score, slot) for score, slot in scored if score > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    if len(scored) == 1:
        return scored[0][1]

    if len(scored) >= 2 and scored[0][0] > scored[1][0]:
        return scored[0][1]

    return None


# ── SLOT DISPLAY ──────────────────────────────────────────────────────────────

def render_slots(slots: list[dict]) -> str:
    lines = ["I found these available slots:"]

    for i, slot in enumerate(slots, 1):
        lines.append(f"Option {i}: {slot['label']}.")

    lines.append(
        "Which option works for you? You can say option one, option two, option three, "
        "or ask for other slots."
    )

    return "\n".join(lines)


def show_current_slot_window(session: BookingSession) -> str:
    if not session.filtered_slots:
        return (
            f"I don't see open slots in the next {SLOT_DAYS_AHEAD} days. "
            f"You can check more availability here: https://cal.com/{CAL_USERNAME}"
        )

    start = session.shown_offset
    session.pending_slots = session.filtered_slots[start:start + MAX_SLOTS_SHOWN]

    if not session.pending_slots:
        session.shown_offset = 0
        session.pending_slots = session.filtered_slots[:MAX_SLOTS_SHOWN]

    return render_slots(session.pending_slots)


def show_next_slot_window(session: BookingSession) -> str:
    if not session.filtered_slots:
        return (
            f"No more open slots in the next {SLOT_DAYS_AHEAD} days. "
            f"You can check availability at https://cal.com/{CAL_USERNAME}"
        )

    session.shown_offset += MAX_SLOTS_SHOWN

    if session.shown_offset >= len(session.filtered_slots):
        session.shown_offset = 0
        prefix = "I've looped back to the earliest available options.\n"
    else:
        prefix = "Here are some other available slots:\n"

    return prefix + show_current_slot_window(session)


# ── STAGE ACTIONS ─────────────────────────────────────────────────────────────

def stage_1_ask_preference() -> str:
    return (
        "Sure, I can help schedule an interview with Gaurav. "
        "What day and time generally works best for you? "
        "For example: Wednesday afternoon, tomorrow evening, or any morning this week."
    )


async def show_slots_for_preference(session_id: str, preference_text: str) -> str:
    session = get_session(session_id)
    prefs = parse_preference_from_text(preference_text)

    try:
        all_slots = await fetch_all_slots()
    except Exception:
        return (
            "I had trouble fetching Gaurav's calendar right now. "
            f"You can also book directly at https://cal.com/{CAL_USERNAME}."
        )

    session.all_slots = all_slots

    filtered = [
        slot for slot in all_slots
        if slot_matches_preference(slot["dt"], prefs)
    ]

    if not filtered and prefs.get("strict_date"):
        requested_date = prefs["dates"][0].strftime("%A, %B ") + ordinal_day(prefs["dates"][0].day)
        return (
            f"I don't see open slots on {requested_date}"
            f"{' ' + prefs['period'] if prefs.get('period') else ''}. "
            "Would another time work?"
        )

    if not filtered:
        filtered = all_slots

    session.filtered_slots = filtered
    session.shown_offset = 0
    session.stage = STAGE_SHOWING_SLOTS

    return show_current_slot_window(session)


async def handle_slot_selection(session_id: str, parsed: dict) -> str:
    session = get_session(session_id)

    selected = match_slot_by_number(parsed.get("slot_number"), session.pending_slots)

    if not selected:
        selected = match_slot_by_reference(parsed.get("slot_reference"), session.pending_slots)

    if not selected:
        slots_list = "\n".join(
            f"Option {i + 1}: {s['label']}"
            for i, s in enumerate(session.pending_slots)
        )

        return (
            "I couldn't confidently match that to one of the available options. "
            "Please say option one, option two, option three, or ask for other slots.\n"
            f"{slots_list}"
        )

    session.chosen_slot = selected
    session.stage = STAGE_COLLECTING_INFO
    session.awaiting_email_confirmation = False

    return (
        f"Great, {session.chosen_slot['label']} works. "
        "Please share the attendee's full name and email address for the calendar invite. "
        "For voice, please say the email slowly."
    )


async def handle_contact_collection(session_id: str, parsed: dict, user_message: str) -> str:
    session = get_session(session_id)

    if session.awaiting_email_confirmation:
        if parsed.get("confirmation") is True or parsed.get("intent") == "confirm_booking":
            session.awaiting_email_confirmation = False
            session.stage = STAGE_AWAITING_CONFIRMATION

            return (
                f"I'll book {session.chosen_slot['label']} for "
                f"{session.guest_name} at {session.guest_email}. "
                "Should I confirm this booking?"
            )

        if parsed.get("confirmation") is False or parsed.get("intent") == "deny_or_change":
            session.guest_email = ""
            session.awaiting_email_confirmation = False
            return "Okay, please repeat the full email slowly."

    contact = extract_contact_with_groq(
        user_message=user_message,
        known_name=session.guest_name,
        known_email=session.guest_email,
    )

    if contact.get("name"):
        session.guest_name = contact["name"]

    if contact.get("email") and is_valid_email(contact["email"]):
        session.guest_email = contact["email"]

    if not session.guest_name and not session.guest_email:
        return (
            "I need the attendee's full name and email address. "
            "Please say the name first, then the email slowly."
        )

    if not session.guest_name:
        return "I got the email. Please share the attendee's full name."

    if not session.guest_email:
        return (
            "I got the name, but I could not capture a valid email address clearly. "
            "Please repeat the full email slowly. "
            f"If voice is difficult, you can also book directly at https://cal.com/{CAL_USERNAME}."
        )

    if contact.get("email_confidence") == "low":
        session.awaiting_email_confirmation = True
        return (
            f"I heard the email as {session.guest_email}. "
            "Please confirm if that email is correct by saying yes, or repeat the email."
        )

    session.awaiting_email_confirmation = False
    session.stage = STAGE_AWAITING_CONFIRMATION

    return (
        f"I'll book {session.chosen_slot['label']} for "
        f"{session.guest_name} at {session.guest_email}. "
        "Should I confirm this booking?"
    )


async def confirm_booking_action(session_id: str) -> tuple[str, bool]:
    session = get_session(session_id)

    if not session.guest_name or not session.guest_email or not is_valid_email(session.guest_email):
        session.stage = STAGE_COLLECTING_INFO
        return (
            "Before I book it, I need a valid attendee name and email address. "
            "Please repeat the name and email slowly."
        ), True

    try:
        confirmation = await create_booking(
            slot_start=session.chosen_slot["start"],
            guest_name=session.guest_name,
            guest_email=session.guest_email,
        )

        data = confirmation.get("data", confirmation)
        booking_id = data.get("uid") or data.get("id") or "N/A"

        session.confirmed_booking_uid = booking_id
        session.confirmed_label = session.chosen_slot["label"]
        session.confirmed_guest_email = session.guest_email
        session.stage = STAGE_CONFIRMED

        return (
            "All set! The interview is confirmed.\n\n"
            f"Slot: {session.confirmed_label}\n"
            f"Name: {session.guest_name}\n"
            f"Confirmation ID: {booking_id}\n\n"
            f"A calendar invite has been sent to {session.guest_email}."
        ), False

    except Exception:
        session.stage = STAGE_SHOWING_SLOTS
        return (
            "Something went wrong while booking. "
            f"You can try another slot or book directly at https://cal.com/{CAL_USERNAME}."
        ), True


async def cancel_confirmed_booking_action(session_id: str) -> tuple[str, bool]:
    session = get_session(session_id)

    if not session.confirmed_booking_uid:
        clear_session(session_id)
        return "I don't have a confirmed booking in this session to cancel.", False

    reason = session.cancellation_reason or "Cancelled through Gaurav AI persona"

    try:
        await cancel_booking(
            booking_uid=session.confirmed_booking_uid,
            reason=reason
        )

        cancelled_label = session.confirmed_label
        clear_session(session_id)

        return (
            f"The booking for {cancelled_label} has been cancelled.\n"
            f"Reason: {reason}"
        ), False

    except Exception:
        return (
            "I could not cancel automatically. "
            f"Please manage it directly at https://cal.com/{CAL_USERNAME}."
        ), False


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

async def handle_booking(session_id: str, user_message: str) -> tuple[str, bool]:
    session = get_session(session_id)
    parsed = parse_booking_message(user_message, session)
    intent = parsed["intent"]

    if session.stage == STAGE_CANCEL_REASON:
        if parsed.get("confirmation") is False or intent == "deny_or_change":
            session.stage = STAGE_CONFIRMED
            session.cancellation_reason = ""
            return "No problem. I have kept the booking as it is.", False

        reason = clean_reason(parsed.get("cancellation_reason") or user_message)

        if not reason:
            return "Please share a short reason for cancelling this interview.", True

        session.cancellation_reason = reason
        session.stage = STAGE_CANCEL_CONFIRMING

        return (
            f"Thanks. I'll cancel the confirmed booking for {session.confirmed_label} "
            f"with reason: \"{session.cancellation_reason}\". Should I proceed?"
        ), True

    if session.stage == STAGE_CANCEL_CONFIRMING:
        if parsed.get("confirmation") is True or intent == "confirm_booking":
            return await cancel_confirmed_booking_action(session_id)

        if parsed.get("confirmation") is False or intent == "deny_or_change":
            session.stage = STAGE_CONFIRMED
            return "No problem. I have kept the booking as it is.", False

        return (
            f"Please confirm whether I should cancel the booking for {session.confirmed_label}."
        ), True

    if session.stage == STAGE_AWAITING_CONFIRMATION:
        if parsed.get("confirmation") is True or intent == "confirm_booking":
            return await confirm_booking_action(session_id)

        if parsed.get("confirmation") is False or intent == "deny_or_change":
            session.stage = STAGE_SHOWING_SLOTS
            return (
                "No problem, I have not booked it. "
                "Choose another slot, ask for more options, or say cancel."
            ), True

        return (
            "Please confirm before I create the calendar invite. "
            "Say yes to confirm, no to change, or cancel."
        ), True

    if intent == "restart_flow":
        session = reset_session(session_id)
        session.stage = STAGE_ASKING_PREFERENCE
        return "Sure, let's start again. What day and time works best for the interview?", True

    if intent == "cancel_flow":
        clear_session(session_id)
        return (
            "No problem, I've cancelled this booking flow. "
            "Ask again whenever you want to schedule an interview."
        ), False

    if intent == "cancel_confirmed_booking":
        if session.confirmed_booking_uid:
            session.stage = STAGE_CANCEL_REASON
            return (
                f"I found a confirmed booking for {session.confirmed_label}. "
                "Could you share the reason for cancellation?"
            ), True

        return (
            "I don't have a confirmed booking in this session to cancel. "
            f"You can manage bookings at https://cal.com/{CAL_USERNAME}."
        ), False

    if session.stage == STAGE_STARTED:
        session.stage = STAGE_ASKING_PREFERENCE
        return stage_1_ask_preference(), True

    if session.stage == STAGE_ASKING_PREFERENCE:
        preference_text = parsed.get("preference_text") or user_message
        reply = await show_slots_for_preference(session_id, preference_text)
        return reply, True

    if session.stage == STAGE_SHOWING_SLOTS:
        if intent == "show_more_slots":
            return show_next_slot_window(session), True

        if intent == "select_slot":
            reply = await handle_slot_selection(session_id, parsed)
            return reply, True

        if intent == "provide_preference":
            slots_list = "\n".join(
                f"Option {i + 1}: {s['label']}"
                for i, s in enumerate(session.pending_slots)
            )

            return (
                "I couldn't confidently match that to one of the shown options. "
                "Please choose one of these, or say show other slots:\n"
                f"{slots_list}"
            ), True

        return (
            "Please choose option one, option two, option three, or say show other slots."
        ), True

    if session.stage == STAGE_COLLECTING_INFO:
        reply = await handle_contact_collection(session_id, parsed, user_message)
        return reply, True

    if session.stage == STAGE_CONFIRMED:
        return (
            "The interview is already confirmed. "
            "If you want to cancel it, say you want to cancel the confirmed booking."
        ), False

    session.stage = STAGE_ASKING_PREFERENCE
    return stage_1_ask_preference(), True