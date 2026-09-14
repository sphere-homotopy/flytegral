import hashlib

import httpx

from fly_window.data.download import download_source
from fly_window.data.urls import SourceFile


TEST_SOURCE = SourceFile("fixture.feather", "https://example.test/fixture.feather")


def test_download_source_is_atomic_and_hashes(tmp_path):
    payload = b"male-cns-test"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        record = download_source(TEST_SOURCE, tmp_path, client=client)

    assert record.source_name == TEST_SOURCE.name
    assert record.url == TEST_SOURCE.url
    assert record.size_bytes == len(payload)
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert (tmp_path / TEST_SOURCE.name).read_bytes() == payload
    assert not (tmp_path / f"{TEST_SOURCE.name}.part").exists()
    assert calls == 1


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
