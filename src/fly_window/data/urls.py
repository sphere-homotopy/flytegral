from dataclasses import dataclass


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str


MALECNS_DATASET = "male-cns:v1.0"
_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
ANNOTATIONS_SOURCE = SourceFile(
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    f"{_BASE}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
)
NEUROTRANSMITTER_SOURCE = SourceFile(
    "body-neurotransmitters-male-cns-v1.0.feather",
    f"{_BASE}/body-neurotransmitters-male-cns-v1.0.feather",
)
CONNECTIVITY_SOURCE = SourceFile(
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    f"{_BASE}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
)
