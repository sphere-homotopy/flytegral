from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

import httpx

from .urls import SourceFile


DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DownloadRecord:
    source_name: str
    url: str
    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _record(source: SourceFile, path: Path) -> DownloadRecord:
    size, digest = _hash_file(path)
    return DownloadRecord(
        source_name=source.name,
        url=source.url,
        path=str(path),
        size_bytes=size,
        sha256=digest,
    )


def _metadata_path(temporary: Path) -> Path:
    return temporary.with_name(f"{temporary.name}.meta.json")


def _read_metadata(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")


def _download_full_stream(
    source: SourceFile,
    temporary: Path,
    client: httpx.Client,
) -> None:
    with client.stream("GET", source.url) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())


def _download_ranges(
    source: SourceFile,
    temporary: Path,
    metadata_path: Path,
    client: httpx.Client,
    *,
    total_size: int,
    etag: str | None,
    chunk_size: int,
) -> None:
    metadata = {
        "url": source.url,
        "etag": etag,
        "size_bytes": total_size,
    }

    existing_metadata = _read_metadata(metadata_path)
    resumable = (
        etag is not None
        and temporary.exists()
        and existing_metadata == metadata
        and temporary.stat().st_size <= total_size
    )

    if not resumable:
        temporary.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        temporary.touch()
        _write_metadata(metadata_path, metadata)

    offset = temporary.stat().st_size
    while offset < total_size:
        end = min(offset + chunk_size - 1, total_size - 1)
        response = client.get(
            source.url,
            headers={"Range": f"bytes={offset}-{end}"},
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError(
                f"Range request for {source.name} returned HTTP {response.status_code}, expected 206"
            )

        response_etag = response.headers.get("ETag")
        if etag is not None and response_etag is not None and response_etag != etag:
            raise ValueError(
                f"ETag changed while downloading {source.name}: {etag!r} -> {response_etag!r}"
            )

        expected_size = end - offset + 1
        payload = response.content
        if len(payload) != expected_size:
            raise ValueError(
                f"Range request for {source.name} returned {len(payload)} bytes, "
                f"expected {expected_size} for bytes={offset}-{end}"
            )

        with temporary.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        offset += len(payload)

    actual_size = temporary.stat().st_size
    if actual_size != total_size:
        raise ValueError(
            f"Completed download size mismatch for {source.name}: {actual_size} != {total_size}"
        )


def download_source(
    source: SourceFile,
    destination_dir: Path,
    client: httpx.Client | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DownloadRecord:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    temporary = destination_dir / f"{source.name}.part"
    metadata_path = _metadata_path(temporary)

    if target.exists():
        return _record(source, target)

    owns_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=15.0),
    )
    try:
        head = active_client.head(source.url)
        head.raise_for_status()
        content_length = head.headers.get("Content-Length")
        accepts_ranges = "bytes" in head.headers.get("Accept-Ranges", "").lower()
        etag = head.headers.get("ETag")

        if content_length is not None and accepts_ranges:
            total_size = int(content_length)
            if total_size < 0:
                raise ValueError(f"Negative Content-Length for {source.name}: {total_size}")
            _download_ranges(
                source,
                temporary,
                metadata_path,
                active_client,
                total_size=total_size,
                etag=etag,
                chunk_size=chunk_size,
            )
        else:
            temporary.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            _download_full_stream(source, temporary, active_client)

        temporary.replace(target)
        metadata_path.unlink(missing_ok=True)
        return _record(source, target)
    finally:
        if owns_client:
            active_client.close()
