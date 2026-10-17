from fly_window.text.curriculum import generate_curriculum
from fly_window.text.vocabulary import build_v1_vocabulary


def test_curriculum_is_deterministic_for_seed_and_changes_across_seeds():
    vocabulary = build_v1_vocabulary()

    first = generate_curriculum(vocabulary, seed=1234, count=64)
    replay = generate_curriculum(vocabulary, seed=1234, count=64)
    other = generate_curriculum(vocabulary, seed=1235, count=64)

    assert first == replay
    assert first != other


def test_curriculum_sequences_are_bos_eos_bounded_and_never_use_unk():
    vocabulary = build_v1_vocabulary()
    bos = vocabulary.id_for("<BOS>")
    eos = vocabulary.id_for("<EOS>")
    unk = vocabulary.id_for("<UNK>")

    sequences = generate_curriculum(vocabulary, seed=7, count=256)

    assert len(sequences) == 256
    assert all(sequence[0] == bos for sequence in sequences)
    assert all(sequence[-1] == eos for sequence in sequences)
    assert all(unk not in sequence for sequence in sequences)
    assert all(4 <= len(sequence) <= 20 for sequence in sequences)
    assert all(0 <= token_id < len(vocabulary) for sequence in sequences for token_id in sequence)


def test_curriculum_contains_math_and_fly_voice_without_becoming_all_buzz():
    vocabulary = build_v1_vocabulary()
    sequences = generate_curriculum(vocabulary, seed=99, count=512)
    rendered = [" ".join(vocabulary.token_for(token_id) for token_id in sequence) for sequence in sequences]

    assert any("theorem" in sentence or "proof" in sentence for sentence in rendered)
    buzz_count = sum("bzz" in sentence or "buzz" in sentence for sentence in rendered)
    assert 0 < buzz_count < len(rendered) // 4
