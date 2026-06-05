"""
booking.py
Robust Cal.com v2 booking flow for chat + voice.

Uses:
- Cal.com for real slots and bookings
- Groq as a meaning-extraction layer
- Backend state validation before any real booking/cancellation

Cancellation:
- Current booking flow cancellation: supported.
- Confirmed booking cancellation: supported in the same active backend session.
- Before cancelling a confirmed booking, the assistant asks for a cancellation reason,
  then asks for final confirmation, then calls Cal.com cancellation API.
"""

import os
import re
import json
import httpx
from datetime import datetime, timezone, timedelta, date
from dataclasses import dataclass, field
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────────────────────

CAL_API_KEY = os.environ["CAL_API_KEY"]
CAL_EVENT_TYPE_ID = os.environ["CAL_EVENT_TYPE_ID"]
CAL_USERNAME = os.environ["CAL_USERNAME"]

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CAL_API_BASE = "https://api.cal.com/v2"

CAL_SLOTS_API_VERSION = "2024-09-04"
CAL_BOOKINGS_API_VERSION = "2026-02-25"

SLOT_DAYS_AHEAD = 14
MAX_SLOTS_SHOWN = 3

TIMEZONE_LABEL = "IST"
TIMEZONE_NAME = "Asia/Kolkata"
TIMEZONE_OFFSET = timedelta(hours=5, minutes=30)

groq_client = Groq(api_key=GROQ_API_KEY)

# ── STAGES ───────────────────────────────────────────────────────────────────

STAGE_STARTED = "started"
STAGE_ASKING_PREFERENCE = "asking_preference"
STAGE_SHOWING_SLOTS = "showing_slots"
STAGE_COLLECTING_INFO = "collecting_info"
STAGE_AWAITING_CONFIRMATION = "awaiting_confirmation"
STAGE_CONFIRMED = "confirmed"

# Cancellation stages
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


# ── BOOKING INTENT ────────────────────────────────────────────────────────────

BOOKING_KEYWORDS = [
    "book", "schedule", "interview", "call", "meeting",
    "availability", "available", "slot", "calendar",
    "set up a time", "connect", "appointment", "talk",
    "reschedule", "cancel"
]


def is_booking_intent(message: str) -> bool:
    msg = message.lower()
    return any(keyword in msg for keyword in BOOKING_KEYWORDS)


def is_active_booking_session(session_id: str) -> bool:
    session = sessions.get(session_id)
    return session is not None and session.stage not in (STAGE_CONFIRMED,)


def is_in_booking_flow(session_id: str) -> bool:
    return is_active_booking_session(session_id)


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

    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt + TIMEZONE_OFFSET


def parse_preference_from_text(preference_text: str) -> dict:
    """
    Deterministic date/time filter from parser output.
    The Groq layer extracts preference_text; this function safely filters slots.
    """
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
    if any(word in msg for word in ["morning", "am", "forenoon", "before lunch"]):
        period = "morning"
    elif any(word in msg for word in ["afternoon", "lunch", "midday", "mid-day"]):
        period = "afternoon"
    elif any(word in msg for word in ["evening", "late", "after 5", "after 6", "pm"]):
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
        ist_dt = to_ist(start_dt)
        label = ist_dt.strftime("%A, %d %b · %I:%M %p") + f" {TIMEZONE_LABEL}"

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


# ── GROQ MEANING PARSER ───────────────────────────────────────────────────────

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
    snapshot = []

    for index, slot in enumerate(slots, 1):
        snapshot.append({
            "number": index,
            "label": slot["label"],
            "start": slot["start"],
        })

    return snapshot


def parse_booking_message(user_message: str, session: BookingSession) -> dict:
    """
    Uses Groq only to understand meaning.
    It does not perform booking/cancellation.
    Backend validates every action against session state.
    """

    system_prompt = """
You are a strict booking-message parser for an AI scheduling backend.

Return ONLY valid JSON.
Do not explain.
Do not add markdown.

The current_stage is the most important signal.
Interpret the user message according to the current_stage, not as a fresh independent command.

Possible intents:
- provide_preference: user gives preferred day/time for a meeting.
- select_slot: user chooses one of the shown slots.
- provide_contact: user gives attendee name and/or email.
- confirm_booking: user confirms final booking OR confirms cancellation when current_stage is cancel_confirming.
- deny_or_change: user rejects confirmation, wants to change selected details, or refuses cancellation.
- show_more_slots: user asks for other slots or says shown slots do not work.
- cancel_flow: user wants to cancel/stop the current unconfirmed booking flow.
- cancel_confirmed_booking: user wants to cancel an already confirmed calendar booking.
- provide_cancellation_reason: user gives reason for cancelling a confirmed booking.
- restart_flow: user wants to start booking again.
- unknown: unclear.

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

Stage-specific rules:

1. If current_stage is "cancel_reason":
   - The assistant has already asked the user for the reason for cancelling a confirmed booking.
   - Treat the user message as the cancellation reason unless it clearly refuses cancellation.
   - Return intent "provide_cancellation_reason" and fill cancellation_reason.
   - Do NOT return "cancel_confirmed_booking" in this stage.

2. If current_stage is "cancel_confirming":
   - The assistant has already asked whether to cancel the confirmed calendar booking.
   - If the user agrees to cancel in any wording, return intent "confirm_booking" and confirmation true.
   - If the user refuses or changes their mind, return intent "deny_or_change" and confirmation false.
   - Do NOT return "cancel_confirmed_booking" in this stage.

3. If current_stage is "awaiting_confirmation":
   - If the user agrees to create the booking, return intent "confirm_booking" and confirmation true.
   - If the user refuses, delays, or wants to change details, return intent "deny_or_change" and confirmation false.

4. If current_stage is "showing_slots":
   - If user selects a shown slot, return intent "select_slot".
   - If user asks for other timings, return "show_more_slots".
   - If user gives a new day/time preference, return "provide_preference".

5. If current_stage is "collecting_info":
   - Extract attendee name and email if present.
   - Return intent "provide_contact".

6. If current_stage is "asking_preference":
   - Treat the message as availability preference unless it clearly cancels or restarts.
   - Return intent "provide_preference".

7. If current_stage is "confirmed":
   - If user wants to cancel the confirmed meeting, return "cancel_confirmed_booking".
   - If user wants another booking, return "restart_flow".

General extraction rules:
- If user says something like "the second one", set intent select_slot and slot_number 2.
- If user says a time like "Monday 5 PM", set intent select_slot and slot_reference to that text.
- If user gives name/email, set intent provide_contact.
- Do not invent email, name, slot number, confirmation, or cancellation reason.
- If unclear, return "unknown".
""".strip()

    user_payload = {
        "current_stage": session.stage,
        "user_message": user_message,
        "shown_slots": slot_snapshot(session.pending_slots),
        "chosen_slot": session.chosen_slot.get("label") if session.chosen_slot else None,
        "known_name": session.guest_name or None,
        "known_email": session.guest_email or None,
        "confirmed_booking": {
            "uid": session.confirmed_booking_uid or None,
            "label": session.confirmed_label or None,
            "email": session.confirmed_guest_email or None,
        },
        "known_cancellation_reason": session.cancellation_reason or None,
    }

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()
        parsed = safe_json_loads(raw)

    except Exception:
        parsed = {}

    intent = parsed.get("intent") or "unknown"

    allowed_intents = {
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

    if intent not in allowed_intents:
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


# ── DETERMINISTIC FALLBACKS ───────────────────────────────────────────────────

EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")


def extract_email_fallback(message: str) -> str:
    match = EMAIL_PATTERN.search(message)
    return match.group() if match else ""


def clean_name(name: str | None) -> str:
    if not name:
        return ""

    name = re.sub(r"[^A-Za-z\s.'-]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name[:80]


def clean_reason(reason: str | None) -> str:
    if not reason:
        return ""

    reason = reason.strip()
    reason = re.sub(r"\s+", " ", reason)

    if len(reason) > 250:
        reason = reason[:250].strip()

    return reason


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
    """
    Backend validation for LLM slot_reference.
    Only matches among already shown pending_slots.
    """
    if not reference or not pending_slots:
        return None

    ref = normalize_for_match(reference)

    scored = []

    for slot in pending_slots:
        label = normalize_for_match(slot["label"])
        score = 0

        for token in ref.split():
            if token and token in label:
                score += 1

        ist_dt = to_ist(slot["dt"])
        hour12 = ist_dt.hour % 12 or 12
        minute = ist_dt.minute
        time_patterns = [
            f"{hour12}",
            f"{hour12} pm" if ist_dt.hour >= 12 else f"{hour12} am",
            f"{hour12}:{minute:02d}",
            f"{hour12}:{minute:02d} pm" if ist_dt.hour >= 12 else f"{hour12}:{minute:02d} am",
        ]

        if any(pattern in ref for pattern in time_patterns):
            score += 3

        scored.append((score, slot))

    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda x: x[0], reverse=True)

    if len(scored) == 1:
        return scored[0][1]

    if len(scored) >= 2 and scored[0][0] > scored[1][0]:
        return scored[0][1]

    return None


# ── SLOT DISPLAY ──────────────────────────────────────────────────────────────

def render_slots(slots: list[dict]) -> str:
    lines = ["Here are the best matching available slots:"]

    for index, slot in enumerate(slots, 1):
        lines.append(f"{index}. {slot['label']}")

    lines.append(
        "Which one works for you? You can reply naturally, for example: "
        "'the second one', 'Monday 5 PM', or 'show other slots'."
    )

    return "\n".join(lines)


def show_current_slot_window(session: BookingSession) -> str:
    if not session.filtered_slots:
        return (
            f"I don't see open slots in the next {SLOT_DAYS_AHEAD} days. "
            f"You can check more availability here: https://cal.com/{CAL_USERNAME}"
        )

    start = session.shown_offset
    end = start + MAX_SLOTS_SHOWN
    session.pending_slots = session.filtered_slots[start:end]

    if not session.pending_slots:
        session.shown_offset = 0
        session.pending_slots = session.filtered_slots[:MAX_SLOTS_SHOWN]

    return render_slots(session.pending_slots)


def show_next_slot_window(session: BookingSession) -> str:
    if not session.filtered_slots:
        return (
            f"I don't see more open slots in the next {SLOT_DAYS_AHEAD} days. "
            f"You can check availability directly at https://cal.com/{CAL_USERNAME}"
        )

    session.shown_offset += MAX_SLOTS_SHOWN

    if session.shown_offset >= len(session.filtered_slots):
        session.shown_offset = 0
        prefix = "I have looped back to the earliest available options.\n"
    else:
        prefix = "Here are some other available slots:\n"

    return prefix + show_current_slot_window(session)


# ── STAGE ACTIONS ─────────────────────────────────────────────────────────────

def stage_1_ask_preference() -> str:
    return (
        "Sure, I can help schedule an interview with Gaurav. "
        "What day and time generally works best for you? "
        "For example, you can say Wednesday afternoon, tomorrow evening, or any morning this week."
    )


async def show_slots_for_preference(session_id: str, preference_text: str) -> str:
    session = get_session(session_id)
    prefs = parse_preference_from_text(preference_text)

    try:
        all_slots = await fetch_all_slots()
    except Exception as e:
        return (
            "I had trouble fetching Gaurav's calendar right now. "
            f"Debug error: {str(e)}. "
            f"You can also book directly at https://cal.com/{CAL_USERNAME}"
        )

    session.all_slots = all_slots

    filtered = [
        slot for slot in all_slots
        if slot_matches_preference(slot["dt"], prefs)
    ]

    if not filtered and prefs.get("strict_date"):
        requested_date = prefs["dates"][0].strftime("%A, %d %b")
        return (
            f"I don't see any open slots matching {requested_date}"
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

    selected_slot = match_slot_by_number(parsed.get("slot_number"), session.pending_slots)

    if not selected_slot:
        selected_slot = match_slot_by_reference(parsed.get("slot_reference"), session.pending_slots)

    if not selected_slot:
        slots_list = "\n".join(
            f"{index + 1}. {slot['label']}"
            for index, slot in enumerate(session.pending_slots)
        )

        return (
            "I couldn't confidently identify which slot you meant. "
            "Please choose one of these options:\n"
            f"{slots_list}"
        )

    session.chosen_slot = selected_slot
    session.stage = STAGE_COLLECTING_INFO

    return (
        f"Great, {session.chosen_slot['label']} works. "
        "Please share the attendee's full name and email address for the calendar invite."
    )


async def handle_contact_collection(session_id: str, parsed: dict, user_message: str) -> str:
    session = get_session(session_id)

    parsed_name = clean_name(parsed.get("name"))
    parsed_email = parsed.get("email") or extract_email_fallback(user_message)

    if parsed_name:
        session.guest_name = parsed_name

    if parsed_email:
        session.guest_email = parsed_email

    if not session.guest_name and not session.guest_email:
        return (
            "I still need the attendee's name and email address. "
            "For example: Rahul Sharma, rahul@email.com"
        )

    if not session.guest_email:
        return "I got the name. Please share the attendee's email address."

    if not session.guest_name:
        return "I got the email. Please share the attendee's full name."

    session.stage = STAGE_AWAITING_CONFIRMATION

    return (
        f"I’ll book **{session.chosen_slot['label']}** for "
        f"{session.guest_name} at {session.guest_email}. "
        "Should I confirm this booking?"
    )


async def confirm_booking(session_id: str) -> tuple[str, bool]:
    session = get_session(session_id)

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
            f"Slot: **{session.confirmed_label}**\n"
            f"Name: {session.guest_name}\n"
            f"Confirmation ID: {booking_id}\n\n"
            f"A calendar invite has been sent to {session.guest_email}."
        ), False

    except Exception as error:
        return (
            f"Something went wrong while booking: {str(error)}. "
            f"Please try directly at https://cal.com/{CAL_USERNAME}"
        ), False


async def cancel_confirmed_booking(session_id: str) -> tuple[str, bool]:
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
            f"The booking for **{cancelled_label}** has been cancelled.\n\n"
            f"Cancellation reason: {reason}"
        ), False

    except Exception as error:
        return (
            f"I could not cancel the booking automatically: {str(error)}. "
            f"Please manage it directly at https://cal.com/{CAL_USERNAME}"
        ), False


# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

async def handle_booking(session_id: str, user_message: str):
    session = get_session(session_id)
    parsed = parse_booking_message(user_message, session)
    intent = parsed["intent"]

    # ── Stage has highest priority: cancellation reason collection ────────────
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
            f"Thanks. I’ll cancel the confirmed booking for **{session.confirmed_label}** "
            f"with this reason: “{session.cancellation_reason}”. "
            "Should I proceed?"
        ), True

    # ── Stage has highest priority: cancellation confirmation ─────────────────
    if session.stage == STAGE_CANCEL_CONFIRMING:
        if parsed.get("confirmation") is True or intent == "confirm_booking":
            return await cancel_confirmed_booking(session_id)

        if parsed.get("confirmation") is False or intent == "deny_or_change":
            session.stage = STAGE_CONFIRMED
            return "No problem. I have kept the booking as it is.", False

        return (
            f"Please confirm whether I should cancel the confirmed booking for "
            f"**{session.confirmed_label}**."
        ), True

    # ── Stage has high priority: final booking confirmation ───────────────────
    if session.stage == STAGE_AWAITING_CONFIRMATION:
        if parsed.get("confirmation") is True or intent == "confirm_booking":
            return await confirm_booking(session_id)

        if parsed.get("confirmation") is False or intent == "deny_or_change":
            session.stage = STAGE_SHOWING_SLOTS
            return (
                "No problem. I have not booked it. "
                "You can choose another slot, ask for more slots, or say cancel."
            ), True

        return (
            "Please confirm before I create the calendar invite. "
            "You can say yes to confirm, no to change, or cancel."
        ), True

    # ── Global restart after stage-specific checks ────────────────────────────
    if intent == "restart_flow":
        session = reset_session(session_id)
        session.stage = STAGE_ASKING_PREFERENCE
        return (
            "Sure, let's start the booking flow again. "
            "What day and time works best for the interview?"
        ), True

    # ── Global cancel current unconfirmed flow ────────────────────────────────
    if intent == "cancel_flow":
        clear_session(session_id)
        return (
            "No problem, I’ve cancelled this booking flow. "
            "You can ask again whenever you want to schedule an interview."
        ), False

    # ── Confirmed booking cancellation request ────────────────────────────────
    if intent == "cancel_confirmed_booking":
        if session.confirmed_booking_uid:
            session.stage = STAGE_CANCEL_REASON
            return (
                f"I found a confirmed booking for **{session.confirmed_label}**. "
                "Before I cancel it, could you share the reason for cancellation?"
            ), True

        return (
            "I don't have a confirmed booking in this active session to cancel. "
            f"You can manage bookings directly at https://cal.com/{CAL_USERNAME}"
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
            preference_text = parsed.get("preference_text") or user_message
            reply = await show_slots_for_preference(session_id, preference_text)
            return reply, True

        return (
            "Please choose one of the shown slots, ask for other slots, or say a new preferred day/time."
        ), True

    if session.stage == STAGE_COLLECTING_INFO:
        reply = await handle_contact_collection(session_id, parsed, user_message)
        return reply, True

    if session.stage == STAGE_CONFIRMED:
        return (
            "The interview is already confirmed. "
            "If you want to cancel this confirmed booking, say that you want to cancel the confirmed booking."
        ), False

    session.stage = STAGE_ASKING_PREFERENCE
    return stage_1_ask_preference(), True