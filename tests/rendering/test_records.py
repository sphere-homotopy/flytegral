from fly_window.rendering.records import build_release_record_requests


def test_release_record_requests_are_fixed_and_never_cherry_pick_heldout_seeds():
    requests = build_release_record_requests()

    assert [request.kind for request in requests] == [
        "development-untrained",
        "development-midpoint",
        "development-final",
        "heldout",
        "heldout",
        "heldout",
    ]
    assert [request.checkpoint_label for request in requests] == [
        "untrained",
        "midpoint",
        "final",
        "final",
        "final",
        "final",
    ]
    assert [request.seed for request in requests] == [9000, 9000, 9000, 10000, 10001, 10002]
    assert [request.output_name for request in requests] == [
        "development-untrained.json",
        "development-midpoint.json",
        "development-final.json",
        "heldout-10000.json",
        "heldout-10001.json",
        "heldout-10002.json",
    ]
    assert len({request.output_name for request in requests}) == 6
