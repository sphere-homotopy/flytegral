from fly_window.publishing.buffer_api import build_create_post_request, parse_create_post_response


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
