from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw

import assemble_release_video as base
from fly_window.rendering.assembly import build_release_assembly_plan, validated_release_metric
from fly_window.rendering.media import RELEASE_MEDIA_SPEC
from fly_window.rendering.release import build_release_manifest


def _normalized_evaluation(raw: dict) -> dict:
    summary = raw.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("biological evaluation must contain a summary object")
    if summary.get("model_label") != "oracle-imitation-biological":
        raise ValueError("unexpected biological model label")
    if int(summary.get("episode_count", -1)) != 100:
        raise ValueError("demo release requires exactly 100 held-out episodes")
    if int(summary.get("seed_start", -1)) != 10000 or int(summary.get("seed_end", -1)) != 10099:
        raise ValueError("demo release requires held-out seeds 10000..10099")

    successes = int(summary["successes"])
    passed_original_gate = successes >= 80
    return {
        "held_out_seed_start": 10000,
        "held_out_seed_end": 10099,
        "models": [summary],
        "release_gate": {
            "passed": passed_original_gate,
            "original_minimum_successes": 80,
            "actual_successes": successes,
            "controls_compared": False,
            "reason": (
                "biological model met the original success threshold; controls were skipped for the time-bounded demo"
                if passed_original_gate
                else "biological model did not meet the original 80/100 success threshold; controls were skipped once the gate was impossible"
            ),
        },
        "source_evaluation": {
            "checkpoint": raw.get("checkpoint"),
            "elapsed_seconds": raw.get("elapsed_seconds"),
            "batch_size": raw.get("batch_size"),
            "flytegral_commit": raw.get("flytegral_commit"),
        },
    }


def _write_demo_cards(clips_dir: Path, metric_display: str) -> None:
    spec = RELEASE_MEDIA_SPEC
    clips_dir.mkdir(parents=True, exist_ok=True)

    def new_card():
        image = Image.new("RGB", (spec.width, spec.height), (5, 8, 13))
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (spec.width - spec.brain_panel_width - 44, 0, spec.width, spec.height),
            fill=(3, 7, 12),
        )
        return image, draw

    hook, draw = new_card()
    title_font = base._font(72, bold=True)
    body_font = base._font(34)
    small_font = base._font(24)
    draw.text((92, 126), "Can a fly brain find the window?", font=title_font, fill=(242, 245, 248))
    draw.text((96, 254), "57,407 MaleCNS neurons", font=body_font, fill=(181, 203, 229))
    draw.text((96, 310), "1,543,613 published connections", font=body_font, fill=(181, 203, 229))
    draw.text((96, 416), "Connectome-constrained simulation", font=base._font(31, bold=True), fill=(255, 177, 79))
    y = 500
    for line in base._wrap(draw, base.CAVEAT, small_font, 950):
        draw.text((96, y), line, font=small_font, fill=(144, 157, 175))
        y += 38
    hook.save(clips_dir / "hook.png")

    end, draw = new_card()
    draw.text((92, 136), "Held-out result", font=title_font, fill=(242, 245, 248))
    draw.text((96, 286), metric_display.replace("successes", "exits"), font=base._font(51, bold=True), fill=(255, 177, 79))
    draw.text((96, 384), "seeds 10000-10099 | no cherry-picking", font=base._font(29), fill=(181, 203, 229))
    draw.text((96, 452), "one epoch oracle imitation | frozen MaleCNS topology", font=base._font(27), fill=(181, 203, 229))
    y = 570
    for line in base._wrap(draw, base.CAVEAT, small_font, 950):
        draw.text((96, y), line, font=small_font, fill=(144, 157, 175))
        y += 38
    end.save(clips_dir / "end-card.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the honest 32-second Fly Window demo even when the original benchmark gate was not met."
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--biological-evaluation", type=Path, required=True)
    parser.add_argument("--release-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--renderer-git-sha", required=True)
    parser.add_argument("--source-run-id", action="append", type=int, default=[])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()

    if not args.biological_evaluation.is_file():
        parser.error(f"biological evaluation not found: {args.biological_evaluation}")
    raw = json.loads(args.biological_evaluation.read_text(encoding="utf-8"))
    evaluation = _normalized_evaluation(raw)
    metric = validated_release_metric(evaluation, require_gate=False)
    checkpoints = base._parse_checkpoints(args.checkpoint)

    args.release_evaluation.parent.mkdir(parents=True, exist_ok=True)
    args.release_evaluation.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ffmpeg = base._resolve_ffmpeg(args.ffmpeg)
    ffprobe = base._resolve_ffprobe(args.ffprobe, ffmpeg)
    _write_demo_cards(args.clips_dir, metric.display)

    plan = build_release_assembly_plan()
    for segment in plan:
        source = args.clips_dir / segment.source_name
        if not source.is_file():
            parser.error(f"release source missing: {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fly-window-demo-release-") as temp_raw:
        temp_dir = Path(temp_raw)
        normalized: list[Path] = []
        for index, segment in enumerate(plan):
            source = args.clips_dir / segment.source_name
            normalized_path = temp_dir / f"{index:02d}-{segment.kind}.mp4"
            base._normalize_segment(
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                source=source,
                output=normalized_path,
                duration_seconds=segment.duration_seconds,
            )
            normalized.append(normalized_path)

        concat_file = temp_dir / "concat.txt"
        concat_file.write_text(
            "".join(
                "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"
                for path in normalized
            ),
            encoding="utf-8",
        )
        base._run(
            [
                ffmpeg,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-t",
                f"{RELEASE_MEDIA_SPEC.duration_seconds:.6f}",
                "-c:v",
                RELEASE_MEDIA_SPEC.video_codec,
                "-pix_fmt",
                RELEASE_MEDIA_SPEC.pixel_format,
                "-r",
                str(RELEASE_MEDIA_SPEC.fps),
                "-preset",
                "medium",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-movflags",
                "+faststart",
                str(args.output),
            ]
        )

    media = base._validate_output(ffprobe, args.output)
    checkpoint_hashes = {label: base._sha256(path) for label, path in checkpoints.items()}
    manifest = build_release_manifest(
        evaluation_path=args.release_evaluation,
        output_path=args.output,
        renderer_git_sha=args.renderer_git_sha,
        source_run_ids=tuple(args.source_run_id),
        checkpoint_hashes=checkpoint_hashes,
        fps=RELEASE_MEDIA_SPEC.fps,
        width=RELEASE_MEDIA_SPEC.width,
        height=RELEASE_MEDIA_SPEC.height,
        duration_seconds=media["duration_seconds"],
    )
    manifest["release_mode"] = "audited-demo-below-original-gate"
    manifest["heldout_metric"] = {
        "successes": metric.successes,
        "episode_count": metric.episode_count,
        "success_rate": metric.success_rate,
        "display": metric.display,
    }
    manifest["release_gate"] = evaluation["release_gate"]
    manifest["media_probe"] = media
    manifest["source_sha256"] = {
        segment.source_name: base._sha256(args.clips_dir / segment.source_name)
        for segment in plan
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
