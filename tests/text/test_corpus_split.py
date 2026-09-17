from fly_window.text.pretraining import split_train_heldout


def test_train_heldout_split_is_deterministic_disjoint_and_complete():
    sequences = [(index, index + 100) for index in range(20)]

    train_a, heldout_a = split_train_heldout(
        sequences,
        heldout_fraction=0.2,
        seed=260917,
    )
    train_b, heldout_b = split_train_heldout(
        sequences,
        heldout_fraction=0.2,
        seed=260917,
    )

    assert train_a == train_b
    assert heldout_a == heldout_b
    assert len(train_a) == 16
    assert len(heldout_a) == 4
    assert set(train_a).isdisjoint(heldout_a)
    assert set(train_a) | set(heldout_a) == set(sequences)


def test_train_heldout_split_rejects_too_small_corpus():
    try:
        split_train_heldout([(1, 2)], heldout_fraction=0.2, seed=1)
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("a one-sequence corpus cannot form a held-out split")


def test_train_heldout_split_keeps_at_least_one_example_on_each_side():
    train, heldout = split_train_heldout(
        [(1, 2), (3, 4)],
        heldout_fraction=0.01,
        seed=5,
    )

    assert len(train) == 1
    assert len(heldout) == 1
