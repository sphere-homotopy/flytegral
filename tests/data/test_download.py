import hashlib
import json

import httpx

from fly_window.data.download import download_source
from fly_window.data.urls import SourceFile


TEST_SOURCE = SourceFile("fixture.feather", "https://example.test/fixture.feather")


def _range_response(request: httpx.Request, payload: bytes, etag: str = '"fixture-v1"') -> httpx.Response:
    if request.method == "HEAD":
        return httpx.Response(
            200,
            headers={
                "Content-Length": str(len(payload)),
                "Accept-Ranges": "bytes",
                "ETag": etag,
            },
            request=request,
        )

    raw_range = request.headers.get("Range")
    assert raw_range is not None
    unit, bounds = raw_range.split("=", 1)
    assert unit == "bytes"
    start_text, end_text = bounds.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    chunk = payload[start : end + 1]
    return httpx.Response(
        206,
        content=chunk,
        headers={
            "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{len(payload)}",
            "ETag": etag,
        },
        request=request,
    )


def test_download_source_is_atomic_and_hashes(tmp_path):
    payload = b"male-cns-test"
    requests: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.headers.get("Range")))
        return _range_response(request, payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = download_source(TEST_SOURCE, tmp_path, client=client, chunk_size=5)

    assert record.source_name == TEST_SOURCE.name
    assert record.url == TEST_SOURCE.url
    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / TEST_SOURCE.name).read_bytes() == payload
    assert not (tmp_path / f"{TEST_SOURCE.name}.part").exists()
    assert not (tmp_path / f"{TEST_SOURCE.name}.part.meta.json").exists()
    assert requests == [
        ("HEAD", None),
        ("GET", "bytes=0-4"),
        ("GET", "bytes=5-9"),
        ("GET", "bytes=10-12"),
    ]


def test_download_source_resumes_matching_partial_file(tmp_path):
    payload = b"0123456789abcdef"
    partial = tmp_path / f"{TEST_SOURCE.name}.part"
    partial.write_bytes(payload[:8])
    meta = tmp_path / f"{TEST_SOURCE.name}.part.meta.json"
    meta.write_text(
        json.dumps(
            {
                "url": TEST_SOURCE.url,
                "etag": '"fixture-v1"',
                "size_bytes": len(payload),
            }
        ),
        encoding="utf-8",
    )
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            ranges.append(request.headers["Range"])
        return _range_response(request, payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = download_source(TEST_SOURCE, tmp_path, client=client, chunk_size=4)

    assert ranges == ["bytes=8-11", "bytes=12-15"]
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / TEST_SOURCE.name).read_bytes() == payload


def test_download_source_discards_partial_when_etag_changed(tmp_path):
    payload = b"new-object"
    partial = tmp_path / f"{TEST_SOURCE.name}.part"
    partial.write_bytes(b"old-")
    meta = tmp_path / f"{TEST_SOURCE.name}.part.meta.json"
    meta.write_text(
        json.dumps(
            {
                "url": TEST_SOURCE.url,
                "etag": '"old-etag"',
                "size_bytes": len(payload),
            }
        ),
        encoding="utf-8",
    )
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            ranges.append(request.headers["Range"])
        return _range_response(request, payload, etag='"new-etag"')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = download_source(TEST_SOURCE, tmp_path, client=client, chunk_size=4)

    assert ranges[0] == "bytes=0-3"
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / TEST_SOURCE.name).read_bytes() == payload


def test_download_source_reuses_completed_file_without_http(tmp_path):
    payload = b"already-downloaded"
    target = tmp_path / TEST_SOURCE.name
    target.write_bytes(payload)

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = download_source(TEST_SOURCE, tmp_path, client=client)

    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert calls == 0
