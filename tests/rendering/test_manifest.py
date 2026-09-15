from __future__ import annotations

import hashlib

from fly_window.rendering.release import build_release_manifest


def test_release_manifest_hashes_sources_and_pins_media_properties(tmp_path) -> None:
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_bytes(b'{"held_out_seed_start":10000,"held_out_seed_end":10099}')
    video = tmp_path / "fly-window-x.mp4"
    video.write_bytes(b"deterministic-video-bytes")

    manifest = build_release_manifest(
        evaluation_path=evaluation,
        output_path=video,
        renderer_git_sha="abc123",
        source_run_ids=(111, 222, 333),
        checkpoint_hashes={"midpoint": "midhash", "final": "finalhash"},
        fps=30,
        width=1600,
        height=900,
        duration_seconds=32.0,
    )

    assert manifest["evaluation_sha256"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert manifest["output_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert manifest["renderer_git_sha"] == "abc123"
    assert manifest["source_run_ids"] == [111, 222, 333]
    assert manifest["checkpoint_hashes"] == {"midpoint": "midhash", "final": "finalhash"}
    assert manifest["selected_seeds"] == {
        "development": [9000],
        "heldout": [10000, 10001, 10002],
    }
    assert manifest["resolution"] == {"width": 1600, "height": 900}
    assert manifest["fps"] == 30
    assert manifest["duration_seconds"] == 32.0
    assert manifest["frame_ranges"][0] == {
        "kind": "hook",
        "seed": None,
        "start_frame": 0,
        "end_frame": 90,
    }
    assert manifest["frame_ranges"][-1]["end_frame"] == 960
