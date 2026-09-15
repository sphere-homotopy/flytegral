from fly_window.rendering.media import RELEASE_MEDIA_SPEC


def test_release_media_contract_matches_checkpoint_e_specification():
    spec = RELEASE_MEDIA_SPEC

    assert (spec.width, spec.height) == (1600, 900)
    assert spec.fps == 30
    assert spec.video_codec == "libx264"
    assert spec.pixel_format == "yuv420p"
    assert spec.duration_seconds == 32.0
    assert spec.brain_panel_width == 520
    assert spec.brain_panel_margin == 22
    assert spec.brain_panel_width < spec.width // 2
