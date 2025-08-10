import os
from typing import Iterable

import httpx
from urllib.parse import quote


def _get_supabase_base() -> tuple[str, str]:
    """Return (base_url, service_key) or raise RuntimeError if missing."""
    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise RuntimeError(
            "Supabase is not configured; set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"
        )
    supabase_url = supabase_url.rstrip("/")
    return f"{supabase_url}/rest/v1", service_key


def _get_schema() -> str:
    return (os.getenv("SUPABASE_SCHEMA") or "public").strip() or "public"


def _tbl(url_path: str) -> str:
    base, _ = _get_supabase_base()
    return f"{base}/{url_path.lstrip('/')}"


def _common_headers() -> dict[str, str]:
    _, service_key = _get_supabase_base()
    schema = _get_schema()
    return {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Accept-Profile": schema,
    }


def _upsert_headers() -> dict[str, str]:
    headers = _common_headers()
    headers.update(
        {
            "Prefer": "resolution=merge-duplicates, return=representation",
            "Content-Profile": _get_schema(),
        }
    )
    return headers


async def upsert_users_quiz(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    mbti: str,
    confidence_by_axis: dict,
    axis_sums: dict,
    raw_answers: dict | list | str | None = None,
    sanitized_answers: dict | list | None = None,
) -> dict:
    payload_item: dict = {
        "user_id": user_id,
        "type": mbti,
        "confidence_by_axis": confidence_by_axis,
        "axis_sums": axis_sums,
    }
    # Only include optional fields if provided to avoid overwriting with null
    if raw_answers is not None:
        payload_item["raw_answers"] = raw_answers
    if sanitized_answers is not None:
        payload_item["sanitized_answers"] = sanitized_answers

    payload = [payload_item]
    resp = await client.post(
        _tbl("users_quiz?on_conflict=user_id"), headers=_upsert_headers(), json=payload
    )
    resp.raise_for_status()
    return resp.json()[0]


async def get_users_quiz_by_ids(
    client: httpx.AsyncClient, user_ids: Iterable[str]
) -> list[dict]:
    ids_list = list(user_ids)
    if not ids_list:
        return []
    ids = ",".join(f'"{u}"' for u in ids_list)
    encoded = quote(f"({ids})", safe="")
    url = _tbl(f"users_quiz?select=*&user_id=in.{encoded}")
    resp = await client.get(url, headers=_common_headers())
    resp.raise_for_status()
    return resp.json()


async def upsert_users_profile(
    client: httpx.AsyncClient,
    *,
    user_id: str,
    type_: str,
    availability: list[dict],
    topics: list[str],
    intent: str,
    is_looking: bool,
) -> dict:
    payload = [
        {
            "user_id": user_id,
            "type": type_,
            "availability": availability,
            "topics": topics,
            "intent": intent,
            "is_looking": is_looking,
        }
    ]
    resp = await client.post(
        _tbl("users_profile?on_conflict=user_id"),
        headers=_upsert_headers(),
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()[0]


async def get_user_profile(client: httpx.AsyncClient, user_id: str) -> dict | None:
    url = _tbl(f"users_profile?user_id=eq.{quote(user_id, safe='')}&select=*")
    resp = await client.get(url, headers=_common_headers())
    resp.raise_for_status()
    data = resp.json()
    return data[0] if data else None


async def get_all_profiles(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(_tbl("users_profile?select=*"), headers=_common_headers())
    resp.raise_for_status()
    return resp.json()


async def upsert_counterpart(
    client: httpx.AsyncClient, *, user_id: str, counterpart_type: str
) -> dict:
    payload = [{"user_id": user_id, "counterpart_type": counterpart_type}]
    resp = await client.post(
        _tbl("user_counterpart_type?on_conflict=user_id"),
        headers=_upsert_headers(),
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()[0]


async def get_counterpart(client: httpx.AsyncClient, user_id: str) -> str | None:
    url = _tbl(
        f"user_counterpart_type?user_id=eq.{quote(user_id, safe='')}&select=counterpart_type"
    )
    resp = await client.get(url, headers=_common_headers())
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    first = data[0]
    return (first or {}).get("counterpart_type")


async def create_room_row(
    client: httpx.AsyncClient,
    *,
    participants: list[str],
    tokens: dict[str, str],
    expires_at: str,
) -> dict:
    payload = [
        {
            "participants": participants,
            "tokens": tokens,
            "expires_at": expires_at,
        }
    ]
    resp = await client.post(_tbl("rooms"), headers=_upsert_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()[0]


