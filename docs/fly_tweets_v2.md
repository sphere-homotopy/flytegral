# Fly Tweets v2: semantic generation experiment

Status: **offline experiment / no production rollout**.

## Goal

Make the fly produce short, recognizable mathematical-comic observations rather than uniform-random vocabulary. Preserve the key premise: the actual fly connectome decides creative variables. Deterministic grammar and safety validators may shape the rendering, but an LLM must not silently ghostwrite published tweets.

## Inputs and provenance

- Owner-provided X archive: use `scripts/prepare_owner_math_corpus.py` against the local ZIP. It reads **only** `data/tweets.js` and excludes reposts and replies by default. It never uploads archive content or extracts media.
- Selected original texts are written to `.private/fly_tweets_owner_math.jsonl`, **not tracked by Git**.
- Retain source tweet IDs, author, source URLs and language. Russian tweets require explicit translation and review; do not pretend the original is English.
- Other candidate accounts: `@alcuintweets` and `@mouse_math`. These are *proposed* sources, not automatically collected or approved training data. Filter reposts, replies, non-math material and near-duplicates; retain provenance and review usage rights.

## Export

```powershell
py -3.12 scripts/prepare_owner_math_corpus.py --archive "C:\\Users\\khoru\\Downloads\\twitter-2026-10-26-....zip" --output .private/fly_tweets_owner_math.jsonl --handle sphere_homotopy
```

Archive files and `.private/` must stay outside the Git repository history. Inspect the resulting count and review outliers; keyword matching is a **candidate filter** only.

## Offline translation

`scripts/translate_owner_math_corpus.py` uses a local Argos RU→EN model. It downloads a model only with `--install-model`; after installation inference is local. Install the optional dependency into an isolated environment, never the production Fly venv:

```powershell
py -3.12 -m venv .private/translator-venv
.private/translator-venv/Scripts/python.exe -m pip install argostranslate
.private/translator-venv/Scripts/python.exe scripts/translate_owner_math_corpus.py --input .private/fly_tweets_owner_math.jsonl --output .private/fly_tweets_owner_math_en.jsonl --install-model
```

Use `--limit 30` for a pilot; rerun to resume without replacing existing drafts. An `original_en` tweet is preserved as-is. Russian translations are `machine_draft`, **not approved training data**; review idioms, jokes, symbols and mathematical terminology before marking `reviewed`. Source attribution survives. No archive contents, originals or draft translations are committed or sent to cloud translation APIs.

## Candidate accounts

Configured in `configs/fly_tweets_v2_sources.json`: `@alcuintweets` and `@mouse_math`. Both are **review-only** candidates, not training data. The older X collector accepts `COOK_ACCOUNTS` but its current rows do not reliably identify replies/quote tweets, so ingest **requires explicit `is_reply: false` and `is_quote: false`** from an upgraded or independently reviewed collector. Missing evidence means reject; do not force them into the corpus. `scripts/prepare_candidate_math_corpus.py` also verifies that each status URL belongs to its claimed author, excludes RTs and non-math tweets, and writes source-attributed records to a private file.

```powershell
py -3.12 scripts/prepare_candidate_math_corpus.py --input .private/validated_external_x.jsonl --output .private/external_math_review.jsonl
```

This makes accidental training on `@mouse_math`'s non-mathematical posts or on someone else's retweets less likely. Respect source rights and retain URLs.

## Offline semantic decoder (v0)

The *current* Fly token trajectories are nearly uniform random. This prototype intentionally **does not** pretend the fly understands math or has created original jokes. It is a constrained concept/voice decoder for comparing whether the audience prefers a coherent mathematical-fly contrast over a word salad.

- Inputs: existing Fly outbox `token_ids`, `text`, `rng_seed`, and `idempotency_key`.
- The first mathematically salient sampled token drives the topic; otherwise a deterministic hash of the Fly trace chooses the topic. The trace + seed chooses the voice.
- The fact/world pairs are an explicit, inspectable, math-reviewed seed bank; no pretrained LLM and no direct reuse of other people's tweets.
- Two-pass selection prioritizes unique topics, then unique surface texts. Every record is tagged `publishable: false`; there is **no Buffer/X integration**.
- Production still uses the original generator; changes to v2 cannot silently publish anything.

```powershell
py -3.12 scripts/preview_fly_tweets_v2.py --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --output .private/fly_v2_preview.jsonl --count 24
```

The preview is **not the final learned generator**. Next: train a tiny stateful semantic policy using reviewed English examples from the owner corpus (with manual pairwise funny/boring labels), including return-to-topic memory and style variation. Only promote after a genuine human blind comparison and semantic plagiarism/duplication review.

## Success criteria

Compare the existing generator and the experimental generator on fixed seeds. Metrics: grammatical completeness, unexpected but interpretable relation between mathematical and fly concepts, human pairwise preference, exact/near duplication, and length/EOS behavior. No automatic publishing until approved.
