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

## Success criteria

Compare the existing generator and the experimental generator on fixed seeds. Metrics: grammatical completeness, unexpected but interpretable relation between mathematical and fly concepts, human pairwise preference, exact/near duplication, and length/EOS behavior. No automatic publishing until approved.
