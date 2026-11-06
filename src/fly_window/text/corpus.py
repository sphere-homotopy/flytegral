from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from fly_window.text.vocabulary import FlyVocabulary


# Focused seed set from John D. Cook's official technical-account list.
# DSP/CS accounts remain available to the collector as optional extras, but are
# intentionally excluded from the default math-language seed.
JOHN_D_COOK_MATH_ACCOUNTS = (
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

JOHN_D_COOK_OPTIONAL_ACCOUNTS = (
    "dsp_fact",
    "CompSciFact",
    "SciPyTip",
    "TeXtip",
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
_NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?(?!\w)", re.IGNORECASE
)
_TOKEN_RE = re.compile(
    r"<URL>|<USER>|<NUMBER>|\.\.\.|<=>|<->|=>|->|<-|<=|>=|!=|"
    r"[A-Za-z]+(?:-[A-Za-z]+)*|[^\s]"
)
_SPECIAL_RE = re.compile(r"^<[A-Z]+>$")


@dataclass(frozen=True, slots=True)
class CorpusTweet:
    account: str
    tweet_url: str
    tweet_datetime: str
    text: str
    is_reply: bool = False
    is_repost: bool = False


@dataclass(frozen=True, slots=True)
class PreparedCorpusTweet:
    account: str
    tweet_url: str
    tweet_datetime: str
    token_ids: tuple[int, ...]
    oov_fraction: float


def _canonical_handle(handle: str) -> str:
    return handle.strip().lstrip("@").lower()


def status_belongs_to_account(account: str, tweet_url: str) -> bool:
    """Return True only when a status URL is authored by the requested account."""
    handle = _canonical_handle(account)
    parsed = urlparse(tweet_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[1].lower() != "status":
        return False
    return parts[0].lower() == handle and parts[2].isdigit()


def _canonical_tokens(text: str) -> tuple[str, ...]:
    normalized = _URL_RE.sub(" <URL> ", text)
    normalized = _MENTION_RE.sub(" <USER> ", normalized)
    normalized = _NUMBER_RE.sub(" <NUMBER> ", normalized)

    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalized):
        token = raw if _SPECIAL_RE.match(raw) else raw.lower()
        if token.strip():
            tokens.append(token)
    return tuple(tokens)


def prepare_corpus_tweet(
    row: CorpusTweet,
    vocabulary: FlyVocabulary,
    *,
    max_oov_fraction: float = 0.35,
) -> PreparedCorpusTweet | None:
    """Convert one scraped tweet into a fixed-vocabulary teacher-forcing sequence.

    No semantic quality judgment occurs here. Rows are rejected only for provenance
    (reply/repost/not authored by the configured account), emptiness, or excessive
    vocabulary mismatch.
    """
    if not 0.0 <= max_oov_fraction <= 1.0:
        raise ValueError("max_oov_fraction must lie in [0, 1]")
    if row.is_reply or row.is_repost:
        return None
    if not status_belongs_to_account(row.account, row.tweet_url):
        return None

    body = _canonical_tokens(row.text)
    if not body:
        return None

    unk_id = vocabulary.id_for("<UNK>")
    token_ids: list[int] = []
    unknown_count = 0
    for token in body:
        try:
            token_ids.append(vocabulary.id_for(token))
        except KeyError:
            token_ids.append(unk_id)
            unknown_count += 1

    oov_fraction = unknown_count / len(body)
    if oov_fraction > max_oov_fraction:
        return None

    sequence = (
        vocabulary.id_for("<BOS>"),
        *token_ids,
        vocabulary.id_for("<EOS>"),
    )
    return PreparedCorpusTweet(
        account=_canonical_handle(row.account),
        tweet_url=row.tweet_url.split("?", 1)[0],
        tweet_datetime=row.tweet_datetime,
        token_ids=tuple(sequence),
        oov_fraction=oov_fraction,
    )
