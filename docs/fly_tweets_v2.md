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

If the Argos package registry returns 403, download the same RU→EN package from its official mirror `https://data.argosopentech.com/argospm/v1/translate-ru_en-1_9.argosmodel` into `.private/translate-ru_en-1_9.argosmodel`, then use `--model-file .private/translate-ru_en-1_9.argosmodel` instead of `--install-model` on the first run. Neither this model nor any tweets are committed. Use `--limit 30` for a pilot; rerun without the install option to resume without replacing existing drafts. Every translated record is now checkpointed atomically to the local JSONL file, so a long model run can be interrupted safely; `--limit` counts new translation attempts only. An `original_en` tweet is preserved as-is. Russian translations are `machine_draft`, **not approved training data**; review idioms, jokes, symbols and mathematical terminology before marking `reviewed`. Source attribution survives. No archive contents, originals or draft translations are committed or sent to cloud translation APIs.

## Owner translation pilot

An initial **11-editor-draft** sample has been translated manually from the owner's actual Russian math posts. Drafts remain at `.private/editor_translations_pilot.jsonl`; original Russian tweets remain untouched in `.private/fly_tweets_owner_math.jsonl`. The merged, attributed corpus is at `.private/fly_tweets_owner_math_en.jsonl`. None of these files is tracked or training-approved.

```powershell
py -3.12 scripts/apply_editor_translations.py --input .private/fly_tweets_owner_math.jsonl --edits .private/editor_translations_pilot.jsonl --output .private/fly_tweets_owner_math_en.jsonl
```

The pilot preserves mathematical words, jokes and authorial tone rather than word-by-word translation; it is still marked `editor_draft`, **not** `reviewed`. When Argos is available, `scripts/translate_owner_math_corpus.py` resumes on the remaining Russian posts without overwriting editor drafts. Both automatic and editor drafts require explicit review before training.

## Human review queue

To avoid teaching the fly broken literal translations or copying old jokes, create an editable CSV review queue from locally translated candidates:

```powershell
py -3.12 scripts/build_fly_v2_review_queue.py --input .private/fly_tweets_owner_math_en.jsonl --output .private/fly_v2_review.csv --count 100
```

The CSV retains the original, English draft, source URL, translation state, a blank `review_decision`, and reviewer notes. Editor drafts are presented before machine drafts; long and URL-dependent examples are deprioritized. **Nothing is automatically approved or used by the model.** This is an input for human style/joke and mathematical correctness review. Keep the CSV private and do not sync it into production automatically.

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
