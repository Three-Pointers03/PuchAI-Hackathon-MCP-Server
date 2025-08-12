import asyncio
import json
from typing import Annotated
import os
import sys
import warnings
import threading
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy
import db
import quiz_utils
import re

import logging
import uuid
from random import random

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format=os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s %(name)s [pid=%(process)d] %(filename)s:%(lineno)d - %(message)s",
    ),
)
logger = logging.getLogger("mcp_server")

# Route warnings to logging and enable default display
warnings.simplefilter("default")
logging.captureWarnings(True)


def _install_global_exception_handlers() -> None:
    """Install hooks so unhandled exceptions are logged with tracebacks."""

    def _sys_excepthook(exc_type, exc, tb):
        logging.getLogger("uncaught").error(
            "Uncaught exception", exc_info=(exc_type, exc, tb)
        )

    def _thread_excepthook(args: threading.ExceptHookArgs):
        logging.getLogger("uncaught").error(
            f"Unhandled exception in thread {args.thread.name}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    msg = context.get("message")
    exc = context.get("exception")
    if exc is not None:
        logging.getLogger("asyncio").error(
            f"Unhandled exception in asyncio loop: {msg or type(exc).__name__}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logging.getLogger("asyncio").error(
            f"Unhandled error in asyncio loop: {msg or context}"
        )


# Shared HTTP client with connection pooling
HTTP_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "100"))
HTTP_MAX_KEEPALIVE = int(os.getenv("HTTP_MAX_KEEPALIVE", "20"))
HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=HTTP_MAX_KEEPALIVE, max_connections=HTTP_MAX_CONNECTIONS
)
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


def _httpx_log_response(response: httpx.Response) -> None:
    try:
        if response.is_error:
            req = response.request
            logging.getLogger("httpx").error(
                f"HTTP error {response.status_code} {response.reason_phrase} for {req.method} {req.url}"
            )
    except Exception:
        # Avoid propagating logging failures
        pass


HTTP_CLIENT = httpx.AsyncClient(
    timeout=45.0,
    http2=False,
)

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

                                 
# Perplexity is optional for this server; analyze_ingredients will error if missing
# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None


# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        client = HTTP_CLIENT
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except httpx.HTTPError as e:
            logger.exception(f"Fetch failed for {url}")
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}")
            )

        if response.status_code >= 400:
            logger.error(
                f"Failed to fetch {url} - status code {response.status_code}"
            )
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch {url} - status code {response.status_code}",
                )
            )

        page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        client = HTTP_CLIENT
        resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
        if resp.status_code != 200:
            return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            href = a["href"]
            if "http" in href:
                links.append(href)
            if len(links) >= num_results:
                break

        return links or ["<error>No results found.</error>"]





async def _post_with_retries(
    url: str, headers: dict, payload: dict, attempts: int = 3
) -> httpx.Response:
    base = 0.5
    for attempt in range(1, attempts + 1):
        try:
            resp = await HTTP_CLIENT.post(url, headers=headers, json=payload)
        except httpx.HTTPError:
            logger.warning(
                f"POST {url} network error on attempt {attempt}/{attempts}; retrying"
            )
            if attempt == attempts:
                raise
            delay = min(8.0, base * (2 ** (attempt - 1)) + random() * 0.2)
            await asyncio.sleep(delay)
            continue

        if resp.status_code in (429, 500, 502, 503, 504):
            logger.warning(
                f"POST {url} got status {resp.status_code} on attempt {attempt}/{attempts}; retrying"
            )
            if attempt == attempts:
                return resp
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = min(10.0, float(retry_after))
            else:
                delay = min(8.0, base * (2 ** (attempt - 1)) + random() * 0.2)
            await asyncio.sleep(delay)
            continue
        return resp
    return resp


# --- MCP Server Setup ---
mcp = FastMCP(
    "Type-to-Talk MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

@mcp.tool
async def about() -> dict:
    return {"name": mcp.name, "description": "This is a Personality Coach mcp server that provides personality insights and guidance based on the user's MBTI type. It is a work in progress and will be updated regularly. It uses the MBTI personality type to provide guidance and insights to the user."}

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


###############################################
# Type-to-Talk: quiz, profiles, matching, rooms
###############################################

# Persisted state is handled via Supabase through db.py helpers


def _normalize_mbti_type(t: str) -> str:
    t = (t or "").upper().strip()
    if len(t) != 4:
        return t
    # Replace any non-EI/SN/TF/JP with 'X'
    slots = [
        t[0] if t[0] in ("E", "I") else "X",
        t[1] if t[1] in ("S", "N") else "X",
        t[2] if t[2] in ("T", "F") else "X",
        t[3] if t[3] in ("J", "P") else "X",
    ]
    return "".join(slots)


def _derive_mbti_from_axis(axis_scores: dict[str, float]) -> str:
    def pick(axis: str, pos: str, neg: str) -> str:
        val = axis_scores.get(axis, 0.0)
        if val > 0:
            return pos
        if val < 0:
            return neg
        return "X"

    return (
        pick("EI", "E", "I")
        + pick("SN", "S", "N")
        + pick("TF", "T", "F")
        + pick("JP", "J", "P")
    )


def _confidence_by_axis(
    responses: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    # returns (confidence_by_axis, sums_by_axis)
    sums: dict[str, float] = {"EI": 0.0, "SN": 0.0, "TF": 0.0, "JP": 0.0}
    totals: dict[str, float] = {"EI": 0.0, "SN": 0.0, "TF": 0.0, "JP": 0.0}
    for r in responses or []:
        axis = str(r.get("axis", "")).upper()
        try:
            value = float(r.get("value"))
        except Exception:
            continue
        if axis in sums:
            sums[axis] += value
            totals[axis] += abs(value)
    conf: dict[str, float] = {}
    for k in sums:
        max_possible = totals[k] if totals[k] > 0 else 1.0
        conf[k] = min(1.0, abs(sums[k]) / max_possible)
    return conf, sums


GenerateQuizDescription = RichToolDescription(
    description=(
        "Generate a fixed MBTI-style questionnaire (16 items: 4 per axis). Present ALL questions at once, "
        "collect '{1a 2b ...}', then call submit_quiz_compact(user_id, answers_compact)."
    ),
    use_when="User needs a ready-to-ask quiz to compute type with submit_quiz_compact.",
    side_effects=None,
)


def _question_bank(axis: str, variant: str) -> list[dict]:
    v = variant.lower().strip() if variant else "general"

    def item(prompt: str, pole: str) -> dict:
        return {"prompt": prompt, "positive_pole": pole}

    if axis == "EI":
        general = [
            item("I feel energized by group conversations.", "E"),
            item("I prefer to process ideas internally before sharing.", "I"),
            item("I enjoy meeting new people spontaneously.", "E"),
            item("Long social gatherings tend to drain me.", "I"),
            item("I speak to think; talking helps me clarify ideas.", "E"),
            item("I think to speak; I prefer to form thoughts before sharing.", "I"),
        ]
        work = [
            item("Team huddles energize me more than solo work blocks.", "E"),
            item("I do my best work with minimal interruptions.", "I"),
        ]
        social = [
            item("I often start conversations with strangers.", "E"),
            item("I prefer a few close friends over many acquaintances.", "I"),
        ]
        return general + (work if v == "work" else social if v == "social" else [])

    if axis == "SN":
        general = [
            item("I focus on concrete facts more than possibilities.", "S"),
            item("I enjoy spotting patterns and future implications.", "N"),
            item("I prefer step-by-step instructions over broad goals.", "S"),
            item("I like exploring unconventional ideas before details.", "N"),
            item("I trust what I can observe and verify directly.", "S"),
            item("I’m drawn to what could be rather than what is.", "N"),
        ]
        work = [
            item("I want requirements spelled out precisely.", "S"),
            item("I enjoy proposing novel product directions.", "N"),
        ]
        social = [
            item("I prefer practical advice to theory in discussions.", "S"),
            item("I like conversations that explore big-picture themes.", "N"),
        ]
        return general + (work if v == "work" else social if v == "social" else [])

    if axis == "TF":
        general = [
            item("I make decisions by weighing logic and consistency.", "T"),
            item("I consider people and values before logic.", "F"),
            item("Direct, candid feedback is most useful to me.", "T"),
            item("Maintaining harmony matters more than being right.", "F"),
            item("I evaluate pros and cons to reach an objective choice.", "T"),
            item("I ask how decisions will feel to those affected.", "F"),
        ]
        work = [
            item("I prefer performance data over sentiment in reviews.", "T"),
            item("I prioritize team morale when choosing tradeoffs.", "F"),
        ]
        social = [
            item("I value constructive critique even if it stings.", "T"),
            item("I tailor my words to protect people’s feelings.", "F"),
        ]
        return general + (work if v == "work" else social if v == "social" else [])

    if axis == "JP":
        general = [
            item("I prefer clear plans and closure.", "J"),
            item("I like to keep options open and adapt as I go.", "P"),
            item("I feel comfortable with set deadlines.", "J"),
            item("I work best with flexible timelines.", "P"),
            item("I’m motivated by checklists and finishing tasks.", "J"),
            item("I’m energized by exploring rather than finishing.", "P"),
        ]
        work = [
            item("I want decisions documented and owners assigned.", "J"),
            item("I prefer iterative plans that can change weekly.", "P"),
        ]
        social = [
            item("I plan trips well in advance.", "J"),
            item("I enjoy spontaneous plans with friends.", "P"),
        ]
        return general + (work if v == "work" else social if v == "social" else [])

    return []


@mcp.tool(description=GenerateQuizDescription.model_dump_json())
async def generate_quiz(
    num_per_axis: Annotated[
        int, Field(description="Ignored. Always 4 per axis.")
    ] = 4,
    variant: Annotated[
        str, Field(description="Ignored. Always uses 'general'.")
    ] = "general",
) -> str:
    # Fixed configuration: 4 questions per axis, using the 'general' bank only
    num = 4
    axes = ["EI", "SN", "TF", "JP"]
    questions: list[dict] = []
    for axis in axes:
        bank = _question_bank(axis, "general")
        if not bank:
            continue
        slice_items = bank[:num]
        for idx, itm in enumerate(slice_items, start=1):
            questions.append(
                {
                    "id": f"{axis}-{idx}",
                    "axis": axis,
                    "prompt": itm["prompt"],
                    "positive_pole": itm["positive_pole"],
                }
            )

    payload = {
        "version": "1.0",
        "variant": "fixed",
        "scale": {
            "labels": [
                "Strongly disagree",
                "Disagree",
                "Slightly disagree",
                "Neutral",
                "Slightly agree",
                "Agree",
                "Strongly agree",
            ],
            "values": [-3, -2, -1, 0, 1, 2, 3],
            "note": "Map answers to -3..3; positive values mean stronger alignment with the question's positive_pole.",
        },
        "instructions": (
        "Present ALL questions at once. Ask the user to reply with a compact string like "
        "'{1a 2b 3c ...}'. Supported: a..d -> -3,-1,1,3 or a..g -> -3..3. "
        "After collecting, call submit_quiz_compact(user_id, answers_compact='{...}')."),
        "questions": questions,
    }
    return json.dumps(payload)

SubmitQuizCompactDescription = RichToolDescription(
    description=(
        "Submit compact quiz answers in one call. Provide answers_compact as a space-separated string of question-answer pairs.\n\n"
        "Format: '1a 2b 3c 4d 5e 6f 7g 8a 9c 10e 11b 12f 13d 14g 15a 16c'\n"
        "- Numbers (1,2,3...) = question index (1-based)\n"
        "- Letters map to values:\n"
        "  * 7-choice: a=-3, b=-2, c=-1, d=0, e=1, f=2, g=3\n"
        "  * 4-choice: a=-3, b=-1, c=1, d=3\n\n"
        "Examples:\n"
        "- '1a 2c 3e 4b 5f 6d 7g 8a' (8 questions)\n"
        "- '1b 3d 5f 7a 9c 11e 13g 15b' (partial responses ok)\n"
        "- '1g 2g 3g 4g' (all strongly agree)\n\n"
        "Returns: JSON with 'type' (MBTI) and 'confidence_by_axis' (EI/SN/TF/JP scores 0-1)."
    ),
    use_when="You have a compact single-string response for the entire quiz.",
    side_effects="Stores quiz result to DB for matching.",
)

@mcp.tool(description=SubmitQuizCompactDescription.model_dump_json())
async def submit_quiz_compact(
    user_id: Annotated[str, Field(description="Unique ID of the user submitting the quiz.")],
    answers_compact: Annotated[str, Field(description="Compact response string like '{1a 2b 3c ...}'.")],
) -> str:
    if not user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="user_id is required"))
    # Reconstruct the fixed questions (deterministic) instead of requiring quiz_json
    axes = ["EI", "SN", "TF", "JP"]
    questions: list[dict] = []
    for axis in axes:
        bank = _question_bank(axis, "general")
        for idx, itm in enumerate(bank[:4], start=1):
            questions.append(
                {
                    "id": f"{axis}-{idx}",
                    "axis": axis,
                    "prompt": itm["prompt"],
                    "positive_pole": itm["positive_pole"],
                }
            )

    tokens = quiz_utils.parse_compact_tokens(answers_compact or "")
    if not tokens:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="answers_compact must contain entries like '1a 2d'"))

    # Support both 4- and 7-choice inputs
    map7 = {"a": -3, "b": -2, "c": -1, "d": 0, "e": 1, "f": 2, "g": 3}
    map4 = {"a": -3, "b": -1, "c": 1, "d": 3}
    POS_FOR_AXIS = {"EI": "E", "SN": "S", "TF": "T", "JP": "J"}

    def letter_value(ch: str) -> float | None:
        ch = ch.lower()
        return map7.get(ch, map4.get(ch))

    responses: list[dict] = []
    for idx, raw_val in tokens:
        i = int(idx) - 1
        if i < 0 or i >= len(questions):
            continue
        q = questions[i]
        # Allow letters, numbers, or text phrases
        v = quiz_utils.sanitize_value(raw_val)
        if v is None:
            continue
        axis = q.get("axis")
        pole = q.get("positive_pole")
        # Ensure E/S/T/J are positive overall
        if axis in POS_FOR_AXIS and pole and pole != POS_FOR_AXIS[axis]:
            v = -v
        responses.append({"axis": axis, "value": float(v)})

    if not responses:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="No valid responses parsed"))

    # Reuse existing flow (validates, stores, returns {type, confidence_by_axis})
    # Include persistence of raw and sanitized answers
    result_json = await _submit_quiz_internal(user_id=user_id, responses=responses)
    return result_json


SubmitQuizDescription = RichToolDescription(
    description=(
        "Submit a short 16-type quiz. Accepts robust inputs: list of {axis, value} or strings like 'EI: a' or 'tf = strongly agree'. "
        "Values can be letters (a..g), numbers (-3..3), or Likert text. Returns {type, confidence_by_axis}. Stores results for user_id."
    ),
    use_when="User has completed a brief MBTI-style quiz and you need to compute their type and confidence.",
    side_effects="Stores quiz result to DB for matching.",
)


async def _submit_quiz_internal(user_id: str, responses: list[dict | str]) -> str:
    """Internal helper function for quiz submission logic."""
    if not user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="user_id is required"))
    if not isinstance(responses, list) or len(responses) == 0:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="responses[] is required and must be non-empty",
            )
        )

    # Sanitize and normalize payload
    try:
        sanitized_list = quiz_utils.sanitize_responses(responses)
    except Exception:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="Invalid responses format; unable to sanitize"
            )
        )
    if not sanitized_list:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="No valid responses after sanitization",
            )
        )

    confidence, sums = _confidence_by_axis(sanitized_list)
    mbti = _derive_mbti_from_axis(sums)
    try:
        await db.upsert_users_quiz(
            HTTP_CLIENT,
            user_id=user_id,
            mbti=mbti,
            confidence_by_axis=confidence,
            axis_sums=sums,
            raw_answers=responses,
            sanitized_answers=sanitized_list,
        )
        logger.info(f"[submit_quiz] upsert users_quiz ok user_id={user_id}")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(f"[submit_quiz] users_quiz error user_id={user_id}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to store quiz: {e!r}")
        )

    return json.dumps({"type": mbti, "confidence_by_axis": confidence})


SaveProfileDescription = RichToolDescription(
    description=(
        "Save user profile preferences for matching. Expects availability windows, topics, and intent."
        " Will also save user type; if omitted, will use last quiz result."
    ),
    use_when="User sets or updates preferences before matching.",
    side_effects="Stores/updates profile in DB.",
)


@mcp.tool(description=SaveProfileDescription.model_dump_json())
async def save_profile(
    user_id: Annotated[str, Field(description="User ID to save profile for.")],
    type: Annotated[
        str | None,
        Field(description="MBTI type like INTJ; optional if quiz was submitted."),
    ] = None,
    preferences: Annotated[
        dict | None,
        Field(
            description="{availability:[{day,start,end,tz?}], topics:[string], intent:string, is_looking:boolean}"
        ),
    ] = None,
) -> str:
    if not user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="user_id is required"))

    # determine type: explicit param, else last quiz type from DB, else XXXX
    quiz_type: str | None = None
    try:
        rows = await db.get_users_quiz_by_ids(HTTP_CLIENT, [user_id])
        if rows:
            quiz_type = (rows[0] or {}).get("type")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[save_profile] read users_quiz error user_id={user_id}"
        )
        # continue with fallback
    final_type = _normalize_mbti_type(type or quiz_type or "XXXX")

    prefs = preferences or {}
    # normalize
    avail = prefs.get("availability") or []
    topics = prefs.get("topics") or []
    intent = (prefs.get("intent") or "casual_chat").lower()
    is_looking_val = prefs.get("is_looking")
    if isinstance(is_looking_val, str):
        is_looking = is_looking_val.lower() in ("1", "true", "yes", "on")
    elif is_looking_val is None:
        is_looking = False
    else:
        is_looking = bool(is_looking_val)

    try:
        await db.upsert_users_profile(
            HTTP_CLIENT,
            user_id=user_id,
            type_=final_type,
            availability=avail,
            topics=topics,
            intent=intent,
            is_looking=is_looking,
        )
        logger.info(f"[save_profile] upsert users_profile ok user_id={user_id}")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[save_profile] users_profile error user_id={user_id}"
        )
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to save profile: {e!r}")
        )

    return json.dumps({"ok": True, "type": final_type})


def _availability_overlap(a: list[dict], b: list[dict]) -> tuple[bool, str | None]:
    # naive overlap: same day string and time window intersection
    def parse_time(hm: str) -> tuple[int, int]:
        try:
            hh, mm = hm.split(":")
            return int(hh), int(mm)
        except Exception:
            return 0, 0

    for wa in a or []:
        day_a = str(wa.get("day", "")).strip().lower()
        sa = str(wa.get("start", "00:00"))
        ea = str(wa.get("end", "00:00"))
        for wb in b or []:
            day_b = str(wb.get("day", "")).strip().lower()
            if day_a and day_a == day_b:
                ah, am = parse_time(sa)
                bh, bm = parse_time(wb.get("start", "00:00"))
                aeh, aem = parse_time(ea)
                beh, bem = parse_time(wb.get("end", "00:00"))
                start_a = ah * 60 + am
                end_a = aeh * 60 + aem
                start_b = bh * 60 + bm
                end_b = beh * 60 + bem
                start = max(start_a, start_b)
                end = min(end_a, end_b)
                if end > start:
                    # format overlap window hh:mm-hh:mm
                    sh, sm = divmod(start, 60)
                    eh, em = divmod(end, 60)
                    return True, f"{day_a.title()} {sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
    return False, None


def _type_fit_score(t1: str, t2: str, c1: dict | None, c2: dict | None) -> float:
    t1 = _normalize_mbti_type(t1)
    t2 = _normalize_mbti_type(t2)
    axes = [
        (0, "EI", ("E", "I")),
        (1, "SN", ("S", "N")),
        (2, "TF", ("T", "F")),
        (3, "JP", ("J", "P")),
    ]
    score = 0.0
    for idx, axis, (pos, neg) in axes:
        a = t1[idx] if len(t1) == 4 else "X"
        b = t2[idx] if len(t2) == 4 else "X"
        same = a == b and a in (pos, neg)
        base = 1.0 if same else 0.5 if (a in (pos, neg) and b in (pos, neg)) else 0.25
        # weight by confidence if available
        w1 = (c1 or {}).get(axis, 1.0)
        w2 = (c2 or {}).get(axis, 1.0)
        weight = (w1 + w2) / 2.0
        score += base * weight
    # max per axis ~1 -> total up to ~4
    return score


def _rationale_for_pair(u1: dict, u2: dict, overlap_str: str | None) -> str:
    shared_topics = sorted(set(u1.get("topics", [])) & set(u2.get("topics", [])))
    topic_part = (
        f"shared interests in {', '.join(shared_topics[:3])}"
        if shared_topics
        else "complementary interests"
    )
    type_part = f"type fit {u1.get('type','XXXX')} × {u2.get('type','XXXX')}"
    time_part = f"time overlap {overlap_str}" if overlap_str else "no immediate overlap"
    intent_part = (
        "similar intent"
        if u1.get("intent") == u2.get("intent")
        else "compatible intents"
    )
    return f"Matched due to {topic_part}, {type_part}, {intent_part}, and {time_part}."


FindMatchesDescription = RichToolDescription(
    description=(
        "Find best match candidates for a user. Returns a list of {match_id, user_id, type, score, rationale}."
    ),
    use_when="You need to propose conversation partners after quiz/profile save.",
    side_effects=None,
)



def _generate_starter_prompts(
    type1: str, type2: str, shared_topics: list[str]
) -> list[str]:
    topics_str = (
        ", ".join(shared_topics[:3])
        if shared_topics
        else "interests you both care about"
    )
    return [
        f"Share a recent win and a challenge; ask how an {type2} would approach the challenge.",
        f"Pick one topic from {topics_str} and swap a 2-minute story each.",
        f"Plan a tiny collaboration (15 min) that leverages {type1}'s strengths with {type2}'s style.",
    ]


CreateRoomDescription = RichToolDescription(
    description=(
        "Create a private room for a proposed match_id of the form 'userA|userB'. Returns {room_id, tokens, expires_at, starter_prompts}."
    ),
    use_when="User selects a candidate and wants to start a private conversation.",
    side_effects="Creates a DB room and ephemeral tokens.",
)

from datetime import datetime, timedelta, timezone
import secrets



###############################################
# Type Coach: counterpart, tips, translate
###############################################

SetCounterpartDescription = RichToolDescription(
    description="Set or update counterpart's MBTI type for coaching utilities.",
    use_when="User wants tailored communication tips for a specific type.",
    side_effects="Stores counterpart type in DB for the user.",
)


@mcp.tool(description=SetCounterpartDescription.model_dump_json())
async def set_counterpart(
    user_id: Annotated[str, Field(description="User ID setting a counterpart type.")],
    counterpart_type: Annotated[str, Field(description="MBTI type like ESFP.")],
) -> str:
    if not user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="user_id is required"))
    t = _normalize_mbti_type(counterpart_type)
    if len(t) != 4:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="counterpart_type must be a 4-letter MBTI type",
            )
        )
    try:
        await db.upsert_counterpart(HTTP_CLIENT, user_id=user_id, counterpart_type=t)
        logger.info(f"[set_counterpart] upsert user_counterpart_type ok user_id={user_id}")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[set_counterpart] user_counterpart_type error user_id={user_id}"
        )
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR, message=f"Failed to save counterpart: {e!r}"
            )
        )
    return json.dumps({"ok": True, "counterpart_type": t})


def _tips_for_type(t: str, context: str) -> list[str]:
    t = _normalize_mbti_type(t)
    tips: list[str] = []
    if len(t) != 4:
        return [
            "Be clear and respectful.",
            "State the goal, then the key facts.",
            "Ask one focused question to move forward.",
        ]
    # Axis-based micro-tips
    if t[0] == "E":
        tips.append("Open with energy; invite quick back-and-forth.")
    elif t[0] == "I":
        tips.append("Give time to think; prefer written or structured prompts.")

    if t[1] == "S":
        tips.append("Use concrete examples tied to the current situation.")
    elif t[1] == "N":
        tips.append("Frame the big picture and patterns before details.")

    if t[2] == "T":
        tips.append("Lead with rationale and tradeoffs, not feelings.")
    elif t[2] == "F":
        tips.append("Acknowledge people impact and values first.")

    if t[3] == "J":
        tips.append("Offer a clear plan and next steps.")
    elif t[3] == "P":
        tips.append("Give options and keep it adaptable.")

    # Include the provided context to anchor suggestions
    if context:
        tips.append(f"Context cue: {context.strip()[:120]}")
    return tips[:3]


CoachTipDescription = RichToolDescription(
    description="Return 3 tailored suggestions to communicate with a target type.",
    use_when="User asks how to give feedback, collaborate, or resolve conflict with a type.",
    side_effects=None,
)


@mcp.tool(description=CoachTipDescription.model_dump_json())
async def coach_tip(
    user_id: Annotated[
        str,
        Field(
            description="Caller user ID. Used to look up stored counterpart type if target_type omitted."
        ),
    ],
    context: Annotated[
        str,
        Field(
            description="Brief description like 'give feedback about missed deadline'."
        ),
    ],
    target_type: Annotated[
        str | None,
        Field(description="MBTI type to target; if omitted, uses set_counterpart."),
    ] = None,
) -> str:
    t = target_type
    if not t:
        try:
            t = await db.get_counterpart(HTTP_CLIENT, user_id)
        except (httpx.HTTPError, RuntimeError) as e:
            logger.exception(
                f"[coach_tip] read user_counterpart_type error user_id={user_id}"
            )
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR, message=f"Failed to load counterpart: {e!r}"
                )
            )
    if not t:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="target_type not provided and no stored counterpart; call set_counterpart first",
            )
        )
    tips = _tips_for_type(t, context)
    return json.dumps({"target_type": _normalize_mbti_type(t), "tips": tips})


TranslateDescription = RichToolDescription(
    description="Rewrite a message to better fit the target type's style.",
    use_when="User wants a tone/style translator for a counterpart type.",
    side_effects=None,
)


def _rewrite_for_type(message: str, t: str) -> str:
    t = _normalize_mbti_type(t)
    text = message.strip()
    # Simple heuristic rewrites
    lines: list[str] = []
    if t[2] == "T":
        lines.append("Goal: ")
        lines.append("Rationale: ")
        lines.append("Options: ")
    if t[2] == "F":
        lines.append("Appreciation: ")
        lines.append("Concern: ")
        lines.append("Request: ")
    if t[3] == "J":
        lines.append("Next step by (date/time): ")
    if t[3] == "P":
        lines.append("Open choices: ")

    bullet_prefix = "- " if t[1] == "T" else "- "
    segments = [s.strip() for s in text.split(".") if s.strip()]
    body = "\n".join(f"{bullet_prefix}{s}." for s in segments[:4])
    header = f"For {t}:\n"
    scaffold = "\n".join(lines[:4])
    return f"{header}{body}\n\n{scaffold}".strip()


@mcp.tool(description=TranslateDescription.model_dump_json())
async def translate(
    message: Annotated[str, Field(description="Original message to rewrite.")],
    target_type: Annotated[str, Field(description="MBTI type to tailor to.")],
) -> str:
    t = _normalize_mbti_type(target_type)
    if len(t) != 4:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="target_type must be a 4-letter MBTI type"
            )
        )
    rewritten = _rewrite_for_type(message, t)
    return json.dumps({"target_type": t, "rewritten": rewritten})


def _get_personality_guidance_data(personality_type: str) -> dict:
    """Comprehensive guidance data for all 16 MBTI types."""
    
    guidance_data = {
        "INTJ": {
            "career": {
                "strengths": ["Natural strategic thinking", "Independent work style", "Long-term planning", "System optimization"],
                "success_tips": ["Seek roles with autonomy", "Focus on complex problem-solving", "Build expertise in specialized areas", "Find mentors who appreciate your vision"],
                "pitfalls": ["May struggle with office politics", "Can appear aloof to colleagues", "Might neglect networking", "Impatience with inefficiency"],
                "warning_signs": ["Feeling micromanaged", "Forced into too many meetings", "Asked to do repetitive tasks", "Colleagues seem to avoid you"]
            },
            "relationships": {
                "strengths": ["Loyal and committed", "Values deep connections", "Direct communication", "Supportive of partner's goals"],
                "success_tips": ["Express appreciation explicitly", "Schedule quality time", "Be patient with emotions", "Share your thought process"],
                "pitfalls": ["May seem emotionally distant", "Can be overly critical", "Struggles with small talk", "Difficulty expressing feelings"],
                "warning_signs": ["Partner feels unheard", "Avoiding social gatherings", "Focusing only on problems", "Dismissing emotional needs"]
            }
        },
        "INTP": {
            "career": {
                "strengths": ["Analytical thinking", "Creative problem-solving", "Research abilities", "Theoretical understanding"],
                "success_tips": ["Pursue intellectually stimulating work", "Seek flexible schedules", "Find roles allowing deep focus", "Connect with like-minded colleagues"],
                "pitfalls": ["Procrastination on routine tasks", "Difficulty with deadlines", "May ignore practical constraints", "Struggles with self-promotion"],
                "warning_signs": ["Bored by mundane tasks", "Feeling rushed constantly", "No time for exploration", "Forced into people management"]
            },
            "relationships": {
                "strengths": ["Intellectual companionship", "Respect for independence", "Loyalty to close friends", "Honest communication"],
                "success_tips": ["Share intellectual interests", "Give space for thinking", "Appreciate their insights", "Be patient with emotional expression"],
                "pitfalls": ["May neglect emotional needs", "Forgets important dates", "Avoids conflict resolution", "Difficulty with routine relationship maintenance"],
                "warning_signs": ["Partner feels emotionally neglected", "Avoiding social commitments", "Intellectualizing feelings", "Withdrawing when stressed"]
            }
        },
        "ENTJ": {
            "career": {
                "strengths": ["Natural leadership", "Strategic vision", "Goal achievement", "Efficiency focus"],
                "success_tips": ["Take on leadership roles", "Set ambitious goals", "Build strong networks", "Develop emotional intelligence"],
                "pitfalls": ["May be seen as domineering", "Impatience with slower colleagues", "Workaholic tendencies", "Difficulty delegating"],
                "warning_signs": ["Team seems intimidated", "High turnover in your area", "Constantly working overtime", "Stressed about details"]
            },
            "relationships": {
                "strengths": ["Protective and supportive", "Helps partners achieve goals", "Clear communication", "Long-term commitment"],
                "success_tips": ["Listen before advising", "Show vulnerability", "Appreciate your partner's style", "Schedule relationship time"],
                "pitfalls": ["May try to 'fix' partner", "Can be controlling", "Work takes priority", "Difficulty showing emotions"],
                "warning_signs": ["Partner feels controlled", "Relationship feels like a project", "No time for intimacy", "Always talking about work"]
            }
        },
        "ENTP": {
            "career": {
                "strengths": ["Innovation and creativity", "Adaptability", "Networking abilities", "Big-picture thinking"],
                "success_tips": ["Seek variety in projects", "Build on networking strengths", "Partner with detail-oriented people", "Focus on innovation roles"],
                "pitfalls": ["Struggles with routine tasks", "May abandon projects mid-way", "Difficulty with administrative work", "Overpromises on timelines"],
                "warning_signs": ["Feeling trapped by routine", "Multiple unfinished projects", "Avoiding paperwork", "Losing interest quickly"]
            },
            "relationships": {
                "strengths": ["Enthusiastic and engaging", "Brings excitement to relationships", "Intellectually stimulating", "Adaptable to change"],
                "success_tips": ["Keep relationships fresh and interesting", "Follow through on commitments", "Listen actively", "Show appreciation for stability"],
                "pitfalls": ["May get bored easily", "Struggles with routine relationship tasks", "Can be argumentative", "Difficulty with emotional depth"],
                "warning_signs": ["Partner feels neglected", "Avoiding serious conversations", "Looking for excitement elsewhere", "Making promises you don't keep"]
            }
        },
        "INFJ": {
            "career": {
                "strengths": ["Empathy and insight", "Long-term vision", "Helping others develop", "Creative problem-solving"],
                "success_tips": ["Find meaning-driven work", "Seek quiet work environments", "Use your insight to help others", "Set boundaries to prevent burnout"],
                "pitfalls": ["Perfectionism leads to stress", "Difficulty saying no", "May burn out from helping others", "Sensitive to criticism"],
                "warning_signs": ["Feeling overwhelmed constantly", "Taking on too much", "Losing sleep over work", "Avoiding feedback sessions"]
            },
            "relationships": {
                "strengths": ["Deep understanding of others", "Loyal and caring", "Good listener", "Committed to growth"],
                "success_tips": ["Communicate your needs clearly", "Take time for self-care", "Share your insights", "Set healthy boundaries"],
                "pitfalls": ["May sacrifice own needs", "Holds unrealistic expectations", "Avoids conflict", "Takes things too personally"],
                "warning_signs": ["Feeling resentful", "Partner doesn't know your needs", "Avoiding difficult conversations", "Feeling emotionally drained"]
            }
        },
        "INFP": {
            "career": {
                "strengths": ["Creativity and imagination", "Values alignment", "Empathy for others", "Adaptability"],
                "success_tips": ["Find work aligned with values", "Seek creative outlets", "Work with supportive teams", "Focus on helping others"],
                "pitfalls": ["Struggles with criticism", "May avoid conflict", "Difficulty with strict deadlines", "Procrastination on unpleasant tasks"],
                "warning_signs": ["Work feels meaningless", "Constant criticism", "No creative freedom", "Toxic work environment"]
            },
            "relationships": {
                "strengths": ["Deep emotional connection", "Supportive and caring", "Values authenticity", "Loyal to loved ones"],
                "success_tips": ["Express feelings openly", "Appreciate your partner's uniqueness", "Create meaningful traditions", "Practice direct communication"],
                "pitfalls": ["May idealize relationships", "Avoids confrontation", "Takes criticism personally", "Needs excessive reassurance"],
                "warning_signs": ["Feeling misunderstood", "Partner seems distant", "Avoiding important discussions", "Feeling emotionally overwhelmed"]
            }
        },
        "ENFJ": {
            "career": {
                "strengths": ["Inspiring others", "Team building", "Communication skills", "People development"],
                "success_tips": ["Take on mentoring roles", "Focus on team success", "Use your communication skills", "Set boundaries to avoid burnout"],
                "pitfalls": ["May neglect own needs", "Takes on too much responsibility", "Sensitive to team conflicts", "Difficulty saying no"],
                "warning_signs": ["Constantly helping others", "No time for yourself", "Team conflicts affect you deeply", "Feeling responsible for everyone"]
            },
            "relationships": {
                "strengths": ["Supportive and encouraging", "Great communication", "Helps partner grow", "Creates harmony"],
                "success_tips": ["Focus on your own needs too", "Allow partner independence", "Communicate directly", "Take breaks from giving"],
                "pitfalls": ["May be overly accommodating", "Neglects self-care", "Takes relationship problems personally", "Tries to fix everything"],
                "warning_signs": ["Feeling unappreciated", "Partner seems overwhelmed by attention", "You're always giving", "Relationship feels one-sided"]
            }
        },
        "ENFP": {
            "career": {
                "strengths": ["Enthusiasm and energy", "Creativity", "People skills", "Seeing potential in others"],
                "success_tips": ["Seek variety and flexibility", "Use your people skills", "Focus on big-picture projects", "Work with inspiring colleagues"],
                "pitfalls": ["Struggles with routine tasks", "May overcommit", "Difficulty with details", "Gets bored easily"],
                "warning_signs": ["Feeling trapped by routine", "Multiple unfinished projects", "Avoiding administrative tasks", "Looking for new opportunities constantly"]
            },
            "relationships": {
                "strengths": ["Brings joy and excitement", "Supportive of partner's dreams", "Great at seeing potential", "Enthusiastic about shared activities"],
                "success_tips": ["Follow through on commitments", "Listen to partner's concerns", "Balance excitement with stability", "Show appreciation for routine"],
                "pitfalls": ["May neglect routine relationship needs", "Gets distracted by new interests", "Avoids serious conversations", "Makes promises impulsively"],
                "warning_signs": ["Partner feels neglected", "Avoiding commitment discussions", "Looking for excitement elsewhere", "Breaking promises frequently"]
            }
        },
        "ISTJ": {
            "career": {
                "strengths": ["Reliability and consistency", "Attention to detail", "Planning abilities", "Respect for procedures"],
                "success_tips": ["Excel in structured environments", "Use your organizational skills", "Build expertise gradually", "Seek stable, reputable companies"],
                "pitfalls": ["May resist change", "Struggles with ambiguous situations", "Can be seen as inflexible", "May miss big-picture opportunities"],
                "warning_signs": ["Constant organizational changes", "Unclear expectations", "Being pushed to innovate rapidly", "Lack of clear procedures"]
            },
            "relationships": {
                "strengths": ["Dependable and loyal", "Shows love through actions", "Provides stability", "Committed long-term"],
                "success_tips": ["Express appreciation verbally", "Be open to new experiences", "Communicate your needs", "Show affection regularly"],
                "pitfalls": ["May seem emotionally distant", "Resistant to change", "Difficulty expressing feelings", "Takes things literally"],
                "warning_signs": ["Partner wants more spontaneity", "Feeling emotionally disconnected", "Avoiding new experiences", "Partner seems frustrated with routine"]
            }
        },
        "ISFJ": {
            "career": {
                "strengths": ["Helping others", "Attention to detail", "Reliability", "Team harmony"],
                "success_tips": ["Find service-oriented roles", "Use your people skills", "Seek appreciative environments", "Set boundaries to prevent burnout"],
                "pitfalls": ["May be taken advantage of", "Difficulty saying no", "Avoids self-promotion", "Sensitive to criticism"],
                "warning_signs": ["Constantly helping others", "No recognition for efforts", "Feeling overwhelmed", "Being asked to do everyone's work"]
            },
            "relationships": {
                "strengths": ["Caring and supportive", "Remembers important details", "Creates comfortable home", "Puts others first"],
                "success_tips": ["Express your own needs", "Accept help from others", "Communicate when hurt", "Take time for self-care"],
                "pitfalls": ["May sacrifice own needs", "Holds grudges silently", "Avoids conflict", "Expects appreciation without asking"],
                "warning_signs": ["Feeling unappreciated", "Doing everything for others", "Feeling resentful", "Partner doesn't know you're upset"]
            }
        },
        "ESTJ": {
            "career": {
                "strengths": ["Leadership abilities", "Organizational skills", "Goal achievement", "Decision making"],
                "success_tips": ["Take on management roles", "Focus on results", "Build efficient systems", "Develop people skills"],
                "pitfalls": ["May be seen as controlling", "Impatience with inefficiency", "Difficulty with change", "May neglect relationship building"],
                "warning_signs": ["Team seems reluctant to share ideas", "High stress from constant pressure", "Resistance to your methods", "Feeling isolated from colleagues"]
            },
            "relationships": {
                "strengths": ["Provides stability and security", "Takes care of practical needs", "Loyal and committed", "Clear about expectations"],
                "success_tips": ["Listen to emotions, not just facts", "Show appreciation for differences", "Allow for flexibility", "Express feelings more"],
                "pitfalls": ["May be too controlling", "Focuses on tasks over emotions", "Difficulty with partner's feelings", "Expects efficiency in relationship"],
                "warning_signs": ["Partner feels controlled", "Relationship feels like a business", "Partner stops sharing feelings", "Conflicts over different approaches"]
            }
        },
        "ESFJ": {
            "career": {
                "strengths": ["People skills", "Team harmony", "Service orientation", "Practical application"],
                "success_tips": ["Work in people-focused roles", "Build strong relationships", "Seek appreciative environments", "Use your organizational skills"],
                "pitfalls": ["Takes criticism personally", "May avoid conflict", "Difficulty saying no", "Sensitive to team dysfunction"],
                "warning_signs": ["Constant criticism", "Toxic team environment", "Being taken advantage of", "No appreciation for efforts"]
            },
            "relationships": {
                "strengths": ["Warm and caring", "Creates social connections", "Remembers special occasions", "Makes others feel valued"],
                "success_tips": ["Communicate your needs directly", "Accept that not everyone shows care the same way", "Take time for yourself", "Don't take things personally"],
                "pitfalls": ["May be overly sensitive", "Needs constant reassurance", "Avoids difficult conversations", "Takes responsibility for others' emotions"],
                "warning_signs": ["Feeling unappreciated", "Partner seems distant", "Avoiding conflict", "Feeling responsible for everyone's happiness"]
            }
        },
        "ISTP": {
            "career": {
                "strengths": ["Problem-solving skills", "Practical application", "Independence", "Crisis management"],
                "success_tips": ["Seek hands-on work", "Value flexibility and autonomy", "Focus on practical solutions", "Avoid micromanagement"],
                "pitfalls": ["May seem disengaged", "Struggles with long-term planning", "Avoids office politics", "Difficulty with emotional aspects"],
                "warning_signs": ["Too many meetings", "Rigid schedules", "Emotional workplace drama", "No hands-on work"]
            },
            "relationships": {
                "strengths": ["Loyal and dependable", "Shows love through actions", "Gives space to partner", "Practical problem-solver"],
                "success_tips": ["Express feelings verbally", "Make time for relationship talk", "Show interest in partner's emotions", "Plan some activities together"],
                "pitfalls": ["May seem emotionally distant", "Avoids relationship discussions", "Needs excessive alone time", "Difficulty with emotional support"],
                "warning_signs": ["Partner wants more emotional connection", "Avoiding serious conversations", "Spending all free time alone", "Partner feels neglected"]
            }
        },
        "ISFP": {
            "career": {
                "strengths": ["Creativity and aesthetics", "Helping others individually", "Adaptability", "Values alignment"],
                "success_tips": ["Find meaningful work", "Seek creative outlets", "Work with supportive people", "Avoid high-conflict environments"],
                "pitfalls": ["Avoids competition", "Struggles with criticism", "May procrastinate", "Difficulty with self-promotion"],
                "warning_signs": ["Competitive, cutthroat environment", "Constant criticism", "No creative freedom", "Values conflict with company"]
            },
            "relationships": {
                "strengths": ["Gentle and caring", "Supports partner's individuality", "Loyal and committed", "Creates peaceful environment"],
                "success_tips": ["Communicate needs clearly", "Don't avoid all conflict", "Express appreciation", "Share your creative side"],
                "pitfalls": ["Avoids confrontation", "May be too accommodating", "Sensitive to criticism", "Difficulty expressing needs"],
                "warning_signs": ["Feeling taken for granted", "Partner doesn't know your needs", "Avoiding important discussions", "Feeling emotionally overwhelmed"]
            }
        },
        "ESTP": {
            "career": {
                "strengths": ["Adaptability", "People skills", "Crisis management", "Practical problem-solving"],
                "success_tips": ["Seek variety and action", "Use your networking abilities", "Focus on immediate results", "Avoid too much planning"],
                "pitfalls": ["May seem impulsive", "Struggles with long-term planning", "Bored by routine", "Difficulty with detailed analysis"],
                "warning_signs": ["Too much planning required", "Isolated work environment", "Repetitive tasks", "No variety or excitement"]
            },
            "relationships": {
                "strengths": ["Fun and spontaneous", "Adaptable to partner's needs", "Enjoys shared activities", "Brings excitement"],
                "success_tips": ["Balance fun with serious talk", "Follow through on commitments", "Listen to partner's need for planning", "Show emotional support"],
                "pitfalls": ["May avoid serious conversations", "Impulsive decisions", "Needs constant stimulation", "Difficulty with routine"],
                "warning_signs": ["Partner wants more stability", "Avoiding future planning", "Getting bored with relationship", "Making impulsive relationship decisions"]
            }
        },
        "ESFP": {
            "career": {
                "strengths": ["Enthusiasm and energy", "People skills", "Adaptability", "Team collaboration"],
                "success_tips": ["Work with people", "Seek variety and fun", "Use your communication skills", "Focus on positive team culture"],
                "pitfalls": ["Struggles with criticism", "May avoid conflict", "Difficulty with detailed planning", "Needs approval from others"],
                "warning_signs": ["Isolated work environment", "Constant criticism", "Rigid procedures", "No social interaction"]
            },
            "relationships": {
                "strengths": ["Warm and affectionate", "Makes relationships fun", "Supportive and encouraging", "Good at reading emotions"],
                "success_tips": ["Balance fun with serious discussions", "Don't take criticism personally", "Communicate your needs", "Plan some activities in advance"],
                "pitfalls": ["May avoid difficult conversations", "Needs constant reassurance", "Can be overly emotional", "Difficulty with partner's criticism"],
                "warning_signs": ["Partner seems distant", "Avoiding serious topics", "Feeling criticized constantly", "Relationship lacks depth"]
            }
        }
    }
    
    return guidance_data.get(personality_type, {
        "career": {
            "strengths": ["Unique perspective", "Individual contributions"],
            "success_tips": ["Know your strengths", "Seek supportive environments", "Communicate your value"],
            "pitfalls": ["May struggle with unclear expectations"],
            "warning_signs": ["Feeling misunderstood consistently"]
        },
        "relationships": {
            "strengths": ["Authentic connection", "Individual approach"],
            "success_tips": ["Communicate openly", "Respect differences", "Build on strengths"],
            "pitfalls": ["May have mismatched expectations"],
            "warning_signs": ["Feeling disconnected from others"]
        }
    })


GetPersonalityGuidanceDescription = RichToolDescription(
    description=(
        "Provides comprehensive career and relationship advice based on the user's MBTI personality type. "
        "Returns structured guidance including strengths, success tips, pitfalls to avoid, and warning signs. "
        "Automatically looks up user's personality type from quiz results."
    ),
    use_when="User wants comprehensive life guidance based on their personality type for career or relationship success.",
    side_effects=None,
)


@mcp.tool(description=GetPersonalityGuidanceDescription.model_dump_json())
async def get_personality_guidance(
    user_id: Annotated[str, Field(description="User ID to get guidance for.")],
    guidance_type: Annotated[
        str, 
        Field(description="Type of guidance: 'career', 'relationships', or 'both' (default)")
    ] = "both",
) -> str:
    """Provide comprehensive career and relationship guidance based on personality type."""
    
    if not user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="user_id is required"))
    
    guidance_type = guidance_type.lower().strip()
    if guidance_type not in ["career", "relationships", "both"]:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="guidance_type must be 'career', 'relationships', or 'both'"))
    
    # Look up user's personality type from quiz results
    personality_type = None
    try:
        quiz_rows = await db.get_users_quiz_by_ids(HTTP_CLIENT, [user_id])
        if quiz_rows:
            personality_type = quiz_rows[0].get("type")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(f"[get_personality_guidance] read users_quiz error user_id={user_id}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to load personality type: {e!r}")
        )
    
    if not personality_type:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, 
                message="No personality type found for user. Please complete the quiz first using generate_quiz and submit_quiz_compact."
            )
        )
    
    personality_type = _normalize_mbti_type(personality_type)
    guidance_data = _get_personality_guidance_data(personality_type)
    
    # Build response based on requested guidance type
    response = {
        "personality_type": personality_type,
        "guidance_type": guidance_type
    }
    
    if guidance_type in ["career", "both"]:
        response["career"] = guidance_data.get("career", {})
    
    if guidance_type in ["relationships", "both"]:
        response["relationships"] = guidance_data.get("relationships", {})
    
    logger.info(f"[get_personality_guidance] Generated {guidance_type} guidance for {personality_type} user_id={user_id}")
    
    return json.dumps(response, indent=2)


CheckUserDataStatusDescription = RichToolDescription(
    description=(
        "Check what data we have for a user to avoid redundant requests. "
        "Returns comprehensive status including personality type, profile data, and last quiz date. "
        "Use this to guide conversation flow and prevent asking users to repeat actions they've already completed."
    ),
    use_when="Before prompting user for data or actions, to check what information already exists.",
    side_effects=None,
)


@mcp.tool(description=CheckUserDataStatusDescription.model_dump_json())
async def check_user_data_status(
    user_id: Annotated[str, Field(description="User ID to check data status for.")],
) -> str:
    """Check what data exists for a user to guide conversation flow."""
    
    status = {
        "user_id": user_id,
        "has_personality_type": False,
        "has_profile": False,
        "has_matches": False,
        "personality_type": None,
        "last_quiz_date": None
    }
    
    # Check personality data
    try:
        quiz_rows = await db.get_users_quiz_by_ids(HTTP_CLIENT, [user_id])
        if quiz_rows:
            quiz_data = quiz_rows[0]
            status.update({
                "has_personality_type": True,
                "personality_type": quiz_data.get("type"),
                "last_quiz_date": quiz_data.get("created_at")
            })
    except Exception:
        pass
    
    # Check profile data
    try:
        profile = await db.get_user_profile(HTTP_CLIENT, user_id)
        status["has_profile"] = profile is not None
    except Exception:
        pass
    
    return json.dumps(status)


# --- Run MCP Server ---
async def main():
    # Install asyncio exception handler for better visibility
    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_asyncio_exception_handler)
    except RuntimeError:
        pass
    logger.info("🚀 Starting MCP server on http://0.0.0.0:8086")
    try:
        await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)
    finally:
        await HTTP_CLIENT.aclose()
        logger.info("HTTP client closed; shutdown complete")


if __name__ == "__main__":
    _install_global_exception_handlers()
    # Tame noisy third-party logs unless DEBUG
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("hpack").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    asyncio.run(main())
