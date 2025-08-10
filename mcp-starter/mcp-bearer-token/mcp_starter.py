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
    limits=HTTP_LIMITS,
    timeout=HTTP_TIMEOUT,
    http2=False,
    event_hooks={"response": [_httpx_log_response]},
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


def _extract_b64(data: str) -> str:
    if data.startswith("data:"):
        idx = data.find("base64,")
        if idx != -1:
            return data[idx + 7 :]
    return data


def _approx_raw_size(b64: str) -> int:
    pad = b64.count("=")
    return (len(b64) * 3) // 4 - pad


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


@mcp.tool(description=FindMatchesDescription.model_dump_json())
async def find_matches(
    user_id: Annotated[str, Field(description="The user to find matches for.")],
    limit: Annotated[int, Field(description="Max number of candidates to return.")] = 5,
) -> str:
    try:
        me = await db.get_user_profile(HTTP_CLIENT, user_id)
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[find_matches] read users_profile error user_id={user_id}"
        )
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to load profile: {e!r}")
        )

    if not me:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="save_profile must be called before find_matches",
            )
        )

    if not (me.get("is_looking") is True):
        return json.dumps([])

    try:
        all_profiles = await db.get_all_profiles(HTTP_CLIENT)
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[find_matches] list profiles error user_id={user_id}"
        )
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to load profiles: {e!r}")
        )

    others = [
        p for p in all_profiles
        if (p or {}).get("user_id") != user_id and (p or {}).get("is_looking") is True
    ]

    # fetch quiz confidences for me and others in one call
    ids = [user_id] + [p.get("user_id") for p in others]
    quiz_map: dict[str, dict] = {}
    try:
        quiz_rows = await db.get_users_quiz_by_ids(HTTP_CLIENT, ids)
        for row in quiz_rows:
            if row and "user_id" in row:
                quiz_map[row["user_id"]] = row.get("confidence_by_axis") or {}
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[find_matches] read users_quiz error user_id={user_id}"
        )
        # proceed without confidences

    my_conf = quiz_map.get(user_id)
    candidates: list[dict] = []
    for prof in others:
        other_id = prof.get("user_id")
        conf = quiz_map.get(other_id)
        type_score = _type_fit_score(
            me.get("type", "XXXX"), prof.get("type", "XXXX"), my_conf, conf
        )
        type_score_scaled = (type_score / 4.0) * 60.0
        shared_topics = len(set(me.get("topics", [])) & set(prof.get("topics", [])))
        topic_score = min(20.0, shared_topics * 7.0)
        intent_score = 10.0 if me.get("intent") == prof.get("intent") else 5.0
        has_overlap, overlap_str = _availability_overlap(
            me.get("availability", []), prof.get("availability", [])
        )
        avail_score = 10.0 if has_overlap else 0.0

        total = round(type_score_scaled + topic_score + intent_score + avail_score, 2)
        rationale = _rationale_for_pair(me, prof, overlap_str)
        match_id = f"{user_id}|{other_id}"
        candidates.append(
            {
                "match_id": match_id,
                "user_id": other_id,
                "type": prof.get("type", "XXXX"),
                "score": total,
                "rationale": rationale,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return json.dumps(candidates[: max(1, int(limit))])


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


@mcp.tool(description=CreateRoomDescription.model_dump_json())
async def create_room(
    match_id: Annotated[
        str,
        Field(description="Match identifier returned by find_matches: 'userA|userB'."),
    ],
) -> str:
    try:
        user_a, user_b = match_id.split("|", 1)
    except ValueError:
        raise McpError(
            ErrorData(code=INVALID_PARAMS, message="match_id must be 'userA|userB'")
        )

    try:
        u1 = await db.get_user_profile(HTTP_CLIENT, user_a)
        u2 = await db.get_user_profile(HTTP_CLIENT, user_b)
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(
            f"[create_room] load profiles error match_id={match_id}"
        )
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to load profiles: {e!r}")
        )

    if not u1 or not u2:
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS, message="Both users must have profiles saved"
            )
        )

    if not (u1.get("is_looking") is True and u2.get("is_looking") is True):
        raise McpError(
            ErrorData(
                code=INVALID_PARAMS,
                message="Both users must be actively looking for matches to create a room",
            )
        )

    token_a = secrets.token_urlsafe(24)
    token_b = secrets.token_urlsafe(24)
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    shared_topics = sorted(set(u1.get("topics", [])) & set(u2.get("topics", [])))
    prompts = _generate_starter_prompts(
        u1.get("type", "XXXX"), u2.get("type", "XXXX"), shared_topics
    )

    try:
        row = await db.create_room_row(
            HTTP_CLIENT,
            participants=[user_a, user_b],
            tokens={user_a: token_a, user_b: token_b},
            expires_at=expires_at,
        )
        room_id = str(row.get("room_id") or uuid.uuid4())
        logger.info(f"[create_room] created rooms row room_id={room_id}")
    except (httpx.HTTPError, RuntimeError) as e:
        logger.exception(f"[create_room] rooms error match_id={match_id}")
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"Failed to create room: {e!r}")
        )

    return json.dumps({
        "room_id": room_id,
        "tokens": {user_a: token_a, user_b: token_b},
        "expires_at": expires_at,
        "starter_prompts": prompts,
    })


###############################################
# Diagnostics level tooling (get/set)
###############################################

DiagnosticsDescription = RichToolDescription(
    description="Get or set diagnostics (log) level for the server.",
    use_when="You need more or less verbose server logs during judging.",
    side_effects="Changing level affects subsequent logs.",
)


@mcp.tool(description=DiagnosticsDescription.model_dump_json())
async def diagnostics_level_get() -> str:
    level = logging.getLevelName(logger.getEffectiveLevel())
    return json.dumps({"level": level})


@mcp.tool(description=DiagnosticsDescription.model_dump_json())
async def diagnostics_level_set(
    level: Annotated[
        str, Field(description="One of DEBUG, INFO, WARNING, ERROR, CRITICAL.")
    ]
) -> str:
    lvl = level.upper().strip()
    if lvl not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Invalid level"))
    logging.getLogger().setLevel(lvl)
    logger.setLevel(lvl)
    return json.dumps({"ok": True, "level": lvl})


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
