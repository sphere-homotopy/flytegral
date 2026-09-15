from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from fly_window.rendering.assembly import (
    build_release_assembly_plan,
    validated_release_metric,
)
from fly_window.rendering.media import RELEASE_MEDIA_SPEC
from fly_window.rendering.release import build_release_manifest


CAVEAT = (
    "Every neuron and connection shown comes from the published connectome; "
    "dynamics, sensory encoding, and learning rule are modeled."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def _resolve_ffmpeg(explicit: str | None) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found on PATH and imageio-ffmpeg is not installed"
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _resolve_ffprobe(explicit: str | None, ffmpeg: str) -> str:
    if explicit:
        return explicit
    discovered = shutil.which("ffprobe")
    if discovered:
        return discovered
    ffmpeg_path = Path(ffmpeg)
    suffix = ffmpeg_path.suffix
    sibling = ffmpeg_path.with_name("ffprobe" + suffix)
    if sibling.is_file():
        return str(sibling)
    raise RuntimeError("ffprobe was not found; pass --ffprobe explicitly")


def _probe(ffprobe: str, path: Path) -> dict[str, Any]:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result.stdout)


def _has_audio(probe: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        box = draw.textbbox((0, 0), candidate, font=font)
        if current and box[2] - box[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _write_cards(clips_dir: Path, metric_display: str) -> None:
    from PIL import Image, ImageDraw

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
    title_font = _font(72, bold=True)
    body_font = _font(34)
    small_font = _font(24)
    draw.text((92, 126), "Can a fly brain find the window?", font=title_font, fill=(242, 245, 248))
    draw.text((96, 254), "57,407 MaleCNS neurons", font=body_font, fill=(181, 203, 229))
    draw.text((96, 310), "1,543,613 published connections", font=body_font, fill=(181, 203, 229))
    draw.text((96, 416), "Connectome-constrained simulation", font=_font(31, bold=True), fill=(255, 177, 79))
    y = 500
    for line in _wrap(draw, CAVEAT, small_font, 950):
        draw.text((96, y), line, font=small_font, fill=(144, 157, 175))
        y += 38
    hook.save(clips_dir / "hook.png")

    end, draw = new_card()
    draw.text((92, 136), "Held-out result", font=title_font, fill=(242, 245, 248))
    draw.text((96, 286), metric_display, font=_font(51, bold=True), fill=(255, 177, 79))
    draw.text((96, 384), "seeds 10000–10099 | no cherry-picking", font=_font(29), fill=(181, 203, 229))
    draw.text((96, 452), "biological topology vs untrained + shuffled controls", font=_font(27), fill=(181, 203, 229))
    y = 570
    for line in _wrap(draw, CAVEAT, small_font, 950):
        draw.text((96, y), line, font=small_font, fill=(144, 157, 175))
        y += 38
    end.save(clips_dir / "end-card.png")


def _normalize_segment(
    *,
    ffmpeg: str,
    ffprobe: str,
    source: Path,
    output: Path,
    duration_seconds: float,
) -> None:
    spec = RELEASE_MEDIA_SPEC
    is_image = source.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    duration = f"{duration_seconds:.6f}"
    if is_image:
        command = [
            ffmpeg,
            "-y",
            "-loop",
            "1",
            "-framerate",
            str(spec.fps),
            "-t",
            duration,
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-t",
            duration,
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
        audio_input = "1:a"
    else:
        source_probe = _probe(ffprobe, source)
        command = [ffmpeg, "-y", "-i", str(source)]
        if _has_audio(source_probe):
            audio_input = "0:a"
        else:
            command.extend(
                [
                    "-f",
                    "lavfi",
                    "-t",
                    duration,
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                ]
            )
            audio_input = "1:a"

    video_filter = (
        f"[0:v]scale={spec.width}:{spec.height}:force_original_aspect_ratio=decrease,"
        f"pad={spec.width}:{spec.height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={spec.fps},tpad=stop_mode=clone:stop_duration={duration},"
        f"trim=duration={duration},setpts=PTS-STARTPTS[v];"
        f"[{audio_input}]apad,atrim=duration={duration},asetpts=PTS-STARTPTS[a]"
    )
    command.extend(
        [
            "-filter_complex",
            video_filter,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            duration,
            "-c:v",
            spec.video_codec,
            "-pix_fmt",
            spec.pixel_format,
            "-r",
            str(spec.fps),
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
            str(output),
        ]
    )
    _run(command)


def _validate_output(ffprobe: str, output: Path) -> dict[str, Any]:
    spec = RELEASE_MEDIA_SPEC
    probe = _probe(ffprobe, output)
    video_streams = [
        stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"
    ]
    if len(video_streams) != 1:
        raise ValueError("final release must contain exactly one video stream")
    if len(audio_streams) != 1:
        raise ValueError("final release must contain exactly one audio stream")

    video = video_streams[0]
    if int(video.get("width", 0)) != spec.width or int(video.get("height", 0)) != spec.height:
        raise ValueError("final release resolution does not match media contract")
    if video.get("codec_name") != "h264":
        raise ValueError("final release video codec must be H.264")
    if video.get("pix_fmt") != spec.pixel_format:
        raise ValueError("final release pixel format must be yuv420p")

    rate = video.get("avg_frame_rate") or video.get("r_frame_rate")
    if not isinstance(rate, str) or "/" not in rate:
        raise ValueError("ffprobe did not report a usable frame rate")
    numerator, denominator = rate.split("/", 1)
    fps = float(numerator) / float(denominator)
    if not math.isclose(fps, spec.fps, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"final release fps is {fps}, expected {spec.fps}")

    duration = float(probe.get("format", {}).get("duration", "nan"))
    if not math.isfinite(duration) or not math.isclose(
        duration,
        spec.duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0 / spec.fps,
    ):
        raise ValueError(
            f"final release duration is {duration}, expected {spec.duration_seconds}"
        )
    return {
        "width": spec.width,
        "height": spec.height,
        "fps": fps,
        "codec": video.get("codec_name"),
        "pixel_format": video.get("pix_fmt"),
        "duration_seconds": duration,
        "audio_codec": audio_streams[0].get("codec_name"),
    }


def _parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint entries must be LABEL=PATH")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path)
        if not label or not path.is_file():
            raise ValueError(f"invalid checkpoint entry: {value}")
        checkpoints[label] = path
    required = {"untrained", "midpoint", "final"}
    if set(checkpoints) != required:
        raise ValueError("checkpoints must be exactly untrained, midpoint, and final")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble, validate, and manifest the fixed 32-second Fly Window release video."
    )
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--renderer-git-sha", required=True)
    parser.add_argument("--source-run-id", action="append", type=int, default=[])
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()

    if not args.evaluation.is_file():
        parser.error(f"evaluation not found: {args.evaluation}")
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    metric = validated_release_metric(evaluation)
    checkpoints = _parse_checkpoints(args.checkpoint)

    ffmpeg = _resolve_ffmpeg(args.ffmpeg)
    ffprobe = _resolve_ffprobe(args.ffprobe, ffmpeg)
    _write_cards(args.clips_dir, metric.display)

    plan = build_release_assembly_plan()
    for segment in plan:
        source = args.clips_dir / segment.source_name
        if not source.is_file():
            parser.error(f"release source missing: {source}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fly-window-release-") as temp_raw:
        temp_dir = Path(temp_raw)
        normalized: list[Path] = []
        for index, segment in enumerate(plan):
            source = args.clips_dir / segment.source_name
            normalized_path = temp_dir / f"{index:02d}-{segment.kind}.mp4"
            _normalize_segment(
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
        _run(
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

    media = _validate_output(ffprobe, args.output)
    checkpoint_hashes = {label: _sha256(path) for label, path in checkpoints.items()}
    manifest = build_release_manifest(
        evaluation_path=args.evaluation,
        output_path=args.output,
        renderer_git_sha=args.renderer_git_sha,
        source_run_ids=tuple(args.source_run_id),
        checkpoint_hashes=checkpoint_hashes,
        fps=RELEASE_MEDIA_SPEC.fps,
        width=RELEASE_MEDIA_SPEC.width,
        height=RELEASE_MEDIA_SPEC.height,
        duration_seconds=media["duration_seconds"],
    )
    manifest["heldout_metric"] = {
        "successes": metric.successes,
        "episode_count": metric.episode_count,
        "success_rate": metric.success_rate,
        "display": metric.display,
    }
    manifest["release_gate"] = evaluation["release_gate"]
    manifest["media_probe"] = media
    manifest["source_sha256"] = {
        segment.source_name: _sha256(args.clips_dir / segment.source_name)
        for segment in plan
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
