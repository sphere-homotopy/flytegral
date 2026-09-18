from fly_window.publishing.buffer_api import (
    build_create_post_request,
    build_find_matching_posts_request,
    find_matching_post,
    parse_create_post_response,
)


def test_buffer_create_request_preserves_fly_selected_time_and_key():
    request = build_create_post_request(
        text="bzz topology",
        scheduled_at="2026-09-18T11:37:00+00:00",
        idempotency_key="batch-a:0",
        channel_id="channel-123",
    )

    payload = request["variables"]["input"]
    assert payload["text"] == "bzz topology"
    assert payload["channelId"] == "channel-123"
    assert payload["schedulingType"] == "automatic"
    assert payload["mode"] == "customScheduled"
    assert payload["dueAt"] == "2026-09-18T11:37:00+00:00"
    assert payload["source"] == "flytegral:batch-a:0"


def test_buffer_create_response_extracts_post_id():
    response = {
        "data": {
            "createPost": {
                "__typename": "PostActionSuccess",
                "post": {"id": "post-1", "dueAt": "2026-09-18T11:37:00Z"},
            }
        }
    }

    assert parse_create_post_response(response) == ("post-1", "2026-09-18T11:37:00Z")


def test_find_request_scopes_remote_reconciliation_to_channel_and_due_window():
    request = build_find_matching_posts_request(
        organization_id="org-1",
        channel_id="channel-123",
        scheduled_at="2026-09-18T11:37:00+00:00",
    )

    variables = request["variables"]
    assert variables["organizationId"] == "org-1"
    assert variables["channelId"] == "channel-123"
    assert variables["startDate"] < variables["endDate"]
    assert "scheduled" in request["query"]
    assert "sent" in request["query"]


def test_remote_match_requires_same_channel_text_and_exact_due_time():
    response = {
        "data": {
            "posts": {
                "edges": [
                    {
                        "node": {
                            "id": "wrong-text",
                            "text": "other",
                            "channelId": "channel-123",
                            "dueAt": "2026-09-18T11:37:00Z",
                            "status": "scheduled",
                            "sentAt": None,
                            "externalLink": None,
                        }
                    },
                    {
                        "node": {
                            "id": "post-1",
                            "text": "bzz topology",
                            "channelId": "channel-123",
                            "dueAt": "2026-09-18T11:37:00.000Z",
                            "status": "scheduled",
                            "sentAt": None,
                            "externalLink": None,
                        }
                    },
                ]
            }
        }
    }

    match = find_matching_post(
        response,
        text="bzz topology",
        scheduled_at="2026-09-18T11:37:00+00:00",
        channel_id="channel-123",
    )

    assert match is not None
    assert match["id"] == "post-1"
