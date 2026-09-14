from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import os

import httpx

from .urls import SourceFile


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


def download_source(
    source: SourceFile,
    destination_dir: Path,
    client: httpx.Client | None = None,
) -> DownloadRecord:
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    temporary = destination_dir / f"{source.name}.part"

    if target.exists():
        return _record(source, target)

    owns_client = client is None
    active_client = client or httpx.Client(follow_redirects=True, timeout=None)
    try:
        digest = sha256()
        size = 0
        with active_client.stream("GET", source.url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if owns_client:
            active_client.close()

    return DownloadRecord(
        source_name=source.name,
        url=source.url,
        path=str(target),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )
