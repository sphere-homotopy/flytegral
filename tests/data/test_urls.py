from fly_window.data.urls import (
    ANNOTATIONS_SOURCE,
    CONNECTIVITY_SOURCE,
    MALECNS_DATASET,
    NEUROTRANSMITTER_SOURCE,
)


def test_malecns_v1_sources_are_pinned():
    assert MALECNS_DATASET == "male-cns:v1.0"
    assert ANNOTATIONS_SOURCE.name == "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    assert NEUROTRANSMITTER_SOURCE.name == "body-neurotransmitters-male-cns-v1.0.feather"
    assert CONNECTIVITY_SOURCE.name == "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    assert ANNOTATIONS_SOURCE.url.startswith(
        "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    )
    assert CONNECTIVITY_SOURCE.url.startswith(
        "https://storage.googleapis.com/flyem-male-cns/v1.0/"
    )
