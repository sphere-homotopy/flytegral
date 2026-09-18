from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence


def merge_outbox_rows(
    rows: MutableSequence[dict[str, object]],
    outbox_rows: Sequence[Mapping[str, object]],
) -> int:
    """Merge generated outbox rows into the durable local row store exactly once."""
    existing: dict[str, dict[str, object]] = {}
    for row in rows:
        key = str(row.get("idempotency_key", "") or "").strip()
        if not key:
            raise ValueError("durable row missing idempotency_key")
        if key in existing and existing[key] != row:
            raise ValueError(f"conflicting durable row for idempotency key: {key}")
        existing[key] = row

    added = 0
    for source in outbox_rows:
        candidate = dict(source)
        key = str(candidate.get("idempotency_key", "") or "").strip()
        if not key:
            raise ValueError("outbox row missing idempotency_key")
        previous = existing.get(key)
        if previous is not None:
            if previous != candidate:
                raise ValueError(f"conflicting durable row for idempotency key: {key}")
            continue
        rows.append(candidate)
        existing[key] = candidate
        added += 1

    return added
