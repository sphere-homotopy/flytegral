from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


BUFFER_GRAPHQL_ENDPOINT = "https://api.buffer.com"

_CREATE_POST_QUERY = """
mutation CreateFlyTweet($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post { id dueAt } }
    ... on MutationError { message }
  }
}
""".strip()

_POST_STATUS_QUERY = """
query FlyTweetPost($input: PostInput!) {
  post(input: $input) {
    id
    status
    dueAt
    sentAt
    externalLink
  }
}
""".strip()


def _require_datetime(value: str, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return text


def build_create_post_request(
    *,
    text: str,
    scheduled_at: str,
    idempotency_key: str,
    channel_id: str,
) -> dict[str, object]:
    body = str(text or "").strip()
    due_at = _require_datetime(scheduled_at, name="scheduled_at")
    key = str(idempotency_key or "").strip()
    channel = str(channel_id or "").strip()
    if not body:
        raise ValueError("text is required")
    if not key:
        raise ValueError("idempotency_key is required")
    if not channel:
        raise ValueError("channel_id is required")
    return {
        "query": _CREATE_POST_QUERY,
        "variables": {
            "input": {
                "text": body,
                "channelId": channel,
                "schedulingType": "automatic",
                "mode": "customScheduled",
                "dueAt": due_at,
                "source": f"flytegral:{key}",
            }
        },
    }


def parse_create_post_response(payload: Mapping[str, Any]) -> tuple[str, str]:
    errors = payload.get("errors")
    if errors:
        messages = [str(item.get("message", item)) if isinstance(item, Mapping) else str(item) for item in errors]
        raise RuntimeError("; ".join(messages))
    data = payload.get("data")
    result = data.get("createPost") if isinstance(data, Mapping) else None
    if not isinstance(result, Mapping):
        raise RuntimeError("Buffer response missing data.createPost")
    if result.get("__typename") == "PostActionSuccess":
        post = result.get("post")
        if isinstance(post, Mapping) and str(post.get("id", "")).strip():
            return str(post["id"]), str(post.get("dueAt", "") or "")
    raise RuntimeError(str(result.get("message", "Buffer createPost failed")))


def build_post_status_request(post_id: str) -> dict[str, object]:
    value = str(post_id or "").strip()
    if not value:
        raise ValueError("post_id is required")
    return {"query": _POST_STATUS_QUERY, "variables": {"input": {"id": value}}}


def parse_post_status_response(payload: Mapping[str, Any]) -> dict[str, str]:
    errors = payload.get("errors")
    if errors:
        messages = [str(item.get("message", item)) if isinstance(item, Mapping) else str(item) for item in errors]
        raise RuntimeError("; ".join(messages))
    data = payload.get("data")
    post = data.get("post") if isinstance(data, Mapping) else None
    if not isinstance(post, Mapping):
        raise RuntimeError("Buffer response missing data.post")
    return {
        "id": str(post.get("id", "") or ""),
        "status": str(post.get("status", "") or ""),
        "dueAt": str(post.get("dueAt", "") or ""),
        "sentAt": str(post.get("sentAt", "") or ""),
        "externalLink": str(post.get("externalLink", "") or ""),
    }
