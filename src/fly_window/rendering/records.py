from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReleaseRecordRequest:
    kind: str
    checkpoint_label: str
    seed: int
    output_name: str


def build_release_record_requests() -> tuple[ReleaseRecordRequest, ...]:
    """Return the immutable episode set used by the public release render."""

    return (
        ReleaseRecordRequest(
            kind="development-untrained",
            checkpoint_label="untrained",
            seed=9000,
            output_name="development-untrained.json",
        ),
        ReleaseRecordRequest(
            kind="development-midpoint",
            checkpoint_label="midpoint",
            seed=9000,
            output_name="development-midpoint.json",
        ),
        ReleaseRecordRequest(
            kind="development-final",
            checkpoint_label="final",
            seed=9000,
            output_name="development-final.json",
        ),
        ReleaseRecordRequest(
            kind="heldout",
            checkpoint_label="final",
            seed=10000,
            output_name="heldout-10000.json",
        ),
        ReleaseRecordRequest(
            kind="heldout",
            checkpoint_label="final",
            seed=10001,
            output_name="heldout-10001.json",
        ),
        ReleaseRecordRequest(
            kind="heldout",
            checkpoint_label="final",
            seed=10002,
            output_name="heldout-10002.json",
        ),
    )
