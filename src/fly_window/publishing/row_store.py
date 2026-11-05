from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
import re

# These fields belong to the original generation/cadence decision. Transport,
# delivery, metrics and training fields are intentionally NOT part of identity.
_GENERATION_FIELDS = (
    "batch_id",
    "batch_size",
    "tweet_index",
    "text",
    "generation_time",
    "checkpoint_id",
    "git_sha",
    "rng_seed",
    "token_ids",
    "token_logprobs",
    "termination_reason",
    "scheduled_at",
    "cadence_context",
    "cadence_action",
)


def _key(row: Mapping[str, object], *, origin: str) -> str:
    key = str(row.get("idempotency_key", "") or "").strip()
    if not key:
        raise ValueError(f"{origin} row missing idempotency_key")
    return key


def _check_generation_identity(
    durable: Mapping[str, object],
    incoming: Mapping[str, object],
    key: str,
) -> None:
    """Reject altered generation data, never changes in delivery state.

    A missing field in an older/partial record is not evidence of a conflict;
    whenever the field exists in both records its value must be identical.
    """
    # Native v2 reconciliation stores the published rendering in text, while
    # fly_original_text preserves the immutable neural generation snapshot.
    # Only independently verified native publications use that source text.
    native_rendering = (
        durable.get("decoder") == "fly_compositional_style_v2"
        and durable.get("status") == "published"
        and not str(durable.get("buffer_post_id", "") or "").strip()
        and isinstance(durable.get("fly_original_text"), str)
        and re.fullmatch(
            r"https://x\.com/fly_topology/status/[0-9]+",
            str(durable.get("tweet_url", "") or ""),
        ) is not None
    )
    for field in _GENERATION_FIELDS:
        if field not in durable or field not in incoming:
            continue
        durable_value = durable["fly_original_text"] if field == "text" and native_rendering else durable[field]
        if durable_value != incoming[field]:
            raise ValueError(
                f"conflicting durable row for idempotency key: {key} "
                f"(generation field: {field})"
            )


def merge_outbox_rows(
    rows: MutableSequence[dict[str, object]],
    outbox_rows: Sequence[Mapping[str, object]],
) -> int:
    """Add unseen generations once; never roll back durable publication state.

    The outbox is an immutable generation snapshot. Durable rows subsequently
    acquire statuses, Buffer IDs, X URLs, engagement metrics and rewards. Those
    mutable fields must always remain owned by the durable row.

    Validate the entire batch before mutation so a later conflicting key
    cannot leave a partially merged in-memory row store.
    """
    existing: dict[str, dict[str, object]] = {}
    for row in rows:
        key = _key(row, origin="durable")
        if key in existing:
            raise ValueError(f"duplicate durable row for idempotency key: {key}")
        existing[key] = row

    staged: list[dict[str, object]] = []
    for source in outbox_rows:
        candidate = dict(source)
        key = _key(candidate, origin="outbox")
        previous = existing.get(key)
        if previous is not None:
            _check_generation_identity(previous, candidate, key)
            continue
        staged.append(candidate)
        existing[key] = candidate

    rows.extend(staged)
    return len(staged)
