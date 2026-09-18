from __future__ import annotations

from datetime import UTC, datetime, timedelta
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

_FIND_MATCHING_POSTS_QUERY = """
query FindFlyTweet(
  $organizationId: OrganizationId!
  $channelId: ChannelId!
  $startDate: DateTime!
  $endDate: DateTime!
) {
  posts(
    first: 100
    input: {
      organizationId: $organizationId
      filter: {
        channelIds: [$channelId]
        startDate: $startDate
        endDate: $endDate
        status: [scheduled, sending, sent]
      }
      sort: [{field: dueAt, direction: asc}]
    }
  ) {
    edges {
      node {
        id
        text
        channelId
        dueAt
        status
        sentAt
        externalLink
      }
    }
  }
}
""".strip()


def _parse_datetime(value: str, *, name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed.astimezone(UTC)


def _require_datetime(value: str, *, name: str) -> str:
    _parse_datetime(value, name=name)
    return str(value).strip()


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
        messages = [
            str(item.get("message", item)) if isinstance(item, Mapping) else str(item)
            for item in errors
        ]
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
        messages = [
            str(item.get("message", item)) if isinstance(item, Mapping) else str(item)
            for item in errors
        ]
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


def build_find_matching_posts_request(
    *,
    organization_id: str,
    channel_id: str,
    scheduled_at: str,
) -> dict[str, object]:
    organization = str(organization_id or "").strip()
    channel = str(channel_id or "").strip()
    due = _parse_datetime(scheduled_at, name="scheduled_at")
    if not organization:
        raise ValueError("organization_id is required")
    if not channel:
        raise ValueError("channel_id is required")
    start = due - timedelta(minutes=2)
    end = due + timedelta(minutes=2)
    return {
        "query": _FIND_MATCHING_POSTS_QUERY,
        "variables": {
            "organizationId": organization,
            "channelId": channel,
            "startDate": start.isoformat().replace("+00:00", "Z"),
            "endDate": end.isoformat().replace("+00:00", "Z"),
        },
    }


def find_matching_post(
    payload: Mapping[str, Any],
    *,
    text: str,
    scheduled_at: str,
    channel_id: str,
) -> dict[str, str] | None:
    errors = payload.get("errors")
    if errors:
        messages = [
            str(item.get("message", item)) if isinstance(item, Mapping) else str(item)
            for item in errors
        ]
        raise RuntimeError("; ".join(messages))
    data = payload.get("data")
    posts = data.get("posts") if isinstance(data, Mapping) else None
    edges = posts.get("edges") if isinstance(posts, Mapping) else None
    if not isinstance(edges, list):
        raise RuntimeError("Buffer response missing data.posts.edges")

    expected_text = str(text or "").strip()
    expected_channel = str(channel_id or "").strip()
    expected_due = _parse_datetime(scheduled_at, name="scheduled_at")
    matches: list[dict[str, str]] = []
    for edge in edges:
        node = edge.get("node") if isinstance(edge, Mapping) else None
        if not isinstance(node, Mapping):
            continue
        if str(node.get("text", "") or "").strip() != expected_text:
            continue
        if str(node.get("channelId", "") or "").strip() != expected_channel:
            continue
        due_text = str(node.get("dueAt", "") or "").strip()
        if not due_text:
            continue
        if _parse_datetime(due_text, name="remote dueAt") != expected_due:
            continue
        matches.append(
            {
                "id": str(node.get("id", "") or ""),
                "status": str(node.get("status", "") or ""),
                "dueAt": due_text,
                "sentAt": str(node.get("sentAt", "") or ""),
                "externalLink": str(node.get("externalLink", "") or ""),
            }
        )
    if len(matches) > 1:
        raise RuntimeError("multiple Buffer posts match one Fly Tweets idempotency candidate")
    return matches[0] if matches else None
