from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence


_METRIC_FIELDS = ("views", "likes", "reposts", "replies", "bookmarks")


def merge_collected_metrics(
    rows: Sequence[MutableMapping[str, object]],
    metrics_rows: Sequence[Mapping[str, object]],
) -> int:
    """Merge headless X collector output into durable Fly Tweets rows.

    Rows are matched only by their concrete tweet URL. Unknown collector records are
    ignored so a stale/retried metrics file cannot manufacture training examples.
    Existing queue/provenance fields are left untouched.
    """
    by_url: dict[str, MutableMapping[str, object]] = {}
    for row in rows:
        url = str(row.get("tweet_url", "") or "").strip()
        if not url:
            continue
        if url in by_url and by_url[url] is not row:
            raise ValueError(f"duplicate durable tweet_url: {url}")
        by_url[url] = row

    updated = 0
    for metrics in metrics_rows:
        url = str(metrics.get("tweet_url", "") or "").strip()
        target = by_url.get(url)
        if target is None:
            continue
        collected_at = str(metrics.get("collected_at", "") or "").strip()
        if not collected_at:
            raise ValueError(f"collector metrics missing collected_at for {url}")
        for field in _METRIC_FIELDS:
            if field not in metrics:
                raise ValueError(f"collector metrics missing {field} for {url}")
            target[field] = metrics[field]
        target["metrics_collected_at"] = collected_at
        updated += 1

    return updated
