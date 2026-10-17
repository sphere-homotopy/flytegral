from fly_window.text.corpus import (
    JOHN_D_COOK_MATH_ACCOUNTS,
    CorpusTweet,
    prepare_corpus_tweet,
    status_belongs_to_account,
)
from fly_window.text.vocabulary import build_v1_vocabulary


def test_john_d_cook_default_math_accounts_are_stable_and_focused():
    assert JOHN_D_COOK_MATH_ACCOUNTS == (
        "ProbFact",
        "DataSciFact",
        "AnalysisFact",
        "AlgebraFact",
        "NetworkFact",
        "LogicPractice",
        "TopologyFact",
        "diff_eq",
        "FunctorFact",
    )


def test_status_belongs_to_account_rejects_reposts_from_other_accounts():
    assert status_belongs_to_account(
        "AnalysisFact", "https://x.com/AnalysisFact/status/123456"
    )
    assert status_belongs_to_account(
        "@AnalysisFact", "https://twitter.com/analysisfact/status/123456?s=20"
    )
    assert not status_belongs_to_account(
        "AnalysisFact", "https://x.com/JohnDCook/status/123456"
    )


def test_prepare_corpus_tweet_normalizes_url_mention_number_and_punctuation():
    vocabulary = build_v1_vocabulary()
    row = CorpusTweet(
        account="AnalysisFact",
        tweet_url="https://x.com/AnalysisFact/status/123",
        tweet_datetime="2026-01-02T03:04:05.000Z",
        text="The theorem is 42. See https://example.com @JohnDCook!",
    )

    prepared = prepare_corpus_tweet(row, vocabulary, max_oov_fraction=0.5)

    assert prepared is not None
    tokens = tuple(vocabulary.token_for(token_id) for token_id in prepared.token_ids)
    assert tokens == (
        "<BOS>",
        "the",
        "theorem",
        "is",
        "<NUMBER>",
        ".",
        "see",
        "<URL>",
        "<USER>",
        "!",
        "<EOS>",
    )


def test_prepare_corpus_tweet_rejects_repost_and_excessive_oov_text():
    vocabulary = build_v1_vocabulary()

    repost = CorpusTweet(
        account="TopologyFact",
        tweet_url="https://x.com/someone_else/status/55",
        tweet_datetime="2026-01-01T00:00:00Z",
        text="topology is beautiful",
    )
    assert prepare_corpus_tweet(repost, vocabulary) is None

    unknown = CorpusTweet(
        account="TopologyFact",
        tweet_url="https://x.com/TopologyFact/status/56",
        tweet_datetime="2026-01-01T00:00:00Z",
        text="quuxblorf snarglefrob theorem glipglop",
    )
    assert prepare_corpus_tweet(unknown, vocabulary, max_oov_fraction=0.25) is None
