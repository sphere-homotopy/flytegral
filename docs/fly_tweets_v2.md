# Fly Tweets v2: semantic generation experiment

Status: **live native-X pilot since 2026-11-02**; experimental compositional decoder is enabled only for explicitly queued, fly-timed posts. Buffer API quota is exhausted (30-day window). Historical sections below describe earlier offline milestones.

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
- Two-pass selection prioritizes unique topics, then unique surface texts. A Jaccard word-set near-duplicate guard catches rearrangements of the same joke. The test outbox currently yields 15 genuinely distinct topic previews from 117 raw generated rows. Every record is tagged `publishable: false`; there is **no Buffer/X integration**.
- Production still uses the original generator; changes to v2 cannot silently publish anything.

```powershell
py -3.12 scripts/preview_fly_tweets_v2.py --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --output .private/fly_v2_preview.jsonl --count 24
```

The preview is **not the final learned generator**. Next: train a tiny stateful semantic policy using reviewed English examples from the owner corpus (with manual pairwise funny/boring labels), including return-to-topic memory and style variation. Only promote after a genuine human blind comparison and semantic plagiarism/duplication review.

## Compositional semantic decoder (v1)

A second decoder is implemented in `src/fly_window/text/compositional_engine.py`. It **does not assemble one of the 16 v0 complete jokes**. Instead, its output is composed from independently selected components:

1. **Mathematical concept** (16 concise, hand-vetted mathematical facts with trigger words).
2. **Physical observation** (31 fly-world scenes such as windows, lids, experiments and crumbs).
3. **Rhetorical move** (six grammar strategies: report, objection, label, question, misclassification, diagnosis).

Crucially, the three selector streams are **disjoint slices of actual sampled token IDs**: `token_ids[0::3]` for math, `[1::3]` for scene and `[2::3]` for rhetorical move. If a sampled math keyword is present in the generated text, it narrows the math concept. An independent, deterministic hashing function only maps the sampled token streams to finite choices. `rng_seed` and `idempotency_key` do **not** determine creative variables. This is fly-driven choice, **not** learning or comprehension.

Mathematical concepts and scenes have explicit compatibility tags, which exclude irrelevant combinations such as a banana being classified as a topological boundary, a pigeonhole count being called compactness, or a swatter's shadow having zero Lebesgue measure. This compatibility is a curated bridge, not a proof of semantic quality. The six output structures are grammatically fixed but no complete rendered joke is stored. The result is still a rule-based compositional engine with a weak underlying neural choice signal.

```powershell
py -3.12 scripts/preview_fly_compositional.py --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --output .private/fly_v2_composed.jsonl --count 30 --max-chars 240
```

The comparison report is saved beside it as `.private/fly_v2_composed.report.json`. On 117 historical Fly rows, v0 yielded 15 non-near-duplicate captions; v1 yielded **30** captions across 15 topics, 17 concrete scenes and six rhetorical moves (240-character ceiling). The cardinality improvement proves *combinatorial diversity*, **not** that the jokes are funny or original.

All outputs retain source idempotency key, chosen component IDs and `publishable: false`. The experiment cannot publish to Buffer or X.

### Remaining gaps

- Readability and semantic matching have improved, but recurring clauses (`i have been studying...`) sound mechanical. Some legitimate tag-compatible combinations remain boring or weakly connected.
- Actual humor preferences, mathematical correctness and originality require review, ideally pairwise comparison against v0 and original author posts.
- The owner's 392 selected math posts are available for *style calibration*, but machine-draft translations are not reliable positive examples. No corpus-weight learning has been applied here yet.
- Sampling from the existing Fly policy is near uniform; the policy still needs a trained selector that can learn what kinds of combinations work.

## Blinded A/B evaluation and preference fitting

The experimentation code includes `src/fly_window/text/preference_lab.py`, a paired *reviewer feedback* tool. It does not fabricate positive results or use engagement numbers that are missing.

```powershell
py -3.12 scripts/compare_fly_decoders.py make --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --review-csv .private/fly_v2_ab_review.csv --count 30
```

The CSV has **30 blinded pairs** sampled from the same saved Fly trajectory per pair. One text is v0 (pre-written full situation), the other v1 (composition). Position A/B is deterministic but hidden. The metadata sidecar `.private/fly_v2_ab_review.metadata.jsonl` holds the provenance and generator assignments, separately from the review CSV.

Set `preferred` to `A` or `B`, or to `tie` / `skip`; leave it empty if not reviewed. Don't change the other columns. Only explicit A/B choices create preference training examples:

```powershell
py -3.12 scripts/compare_fly_decoders.py train --review-csv .private/fly_v2_ab_review.csv --model-json .private/fly_v2_preference_model.json
```

A lightweight pairwise logistic learner adjusts interpretable *post-decoder* topic, scene, move, decoder, question and length weights. It is not training the 57k-neuron connectome, and it cannot by itself produce new sentences. With zero submitted labels, it reports `trained: false` and does not write a fake model. All evaluated texts and models stay offline; no publication hook is introduced.

### Evaluate the owner-style v2 against compositional v1

To blind-compare **v1 composition vs v2 compact owner-style composition with owner math-topic prior**, generate a second CSV:

```powershell
py -3.12 scripts/compare_fly_decoders.py make --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --owner-corpus .private/fly_tweets_owner_math_en.jsonl --topic-prior-from-owner --review-csv .private/fly_v2_styled_ab_review.csv --count 30
```

The file has two text columns (`text_a`, `text_b`) and a blank `preferred` column. Set `preferred` to A, B, tie, or skip for each comparison; labels can be collected partially. Metadata remains in the adjacent `.metadata.jsonl` file, so the rater does not see the decoder's name. Both alternatives for each row use **exactly the same original Fly token trace**. Learn the post-hoc preference model only from those marked preferences:

```powershell
py -3.12 scripts/compare_fly_decoders.py train --review-csv .private/fly_v2_styled_ab_review.csv --model-json .private/fly_v2_styled_preference_model.json
```

As of this checkpoint, 30 blinded pairs exist with **zero** human responses: no model was trained and no candidate was approved for publication. Keep the data stable while filling out the review CSV (regeneration would clear labels).

## Corpus-derived style calibration (v2)

The third offline renderer, `src/fly_window/text/style_profile.py`, uses only the owner's `original_en` and `editor_draft` texts as stylistic examples. It excludes machine-draft translations, links, texts over 280 characters, and duplicate texts. **The corpus is used for simple style statistics, not as a stock of jokes to reuse.** It never inserts source sentences into generated posts.

On the actual archived short examples (22 eligible): median **78 characters**, **54.5%** contain a question, **4.5%** begin lowercase. The decoder therefore searches several grammatical renderings of the **same Fly-chosen math topic and physical scene**, with question intent drawn from a separate deterministic transform of sampled Fly tokens. It chooses a complete, short rendering by closeness to the owner's median length and observed question share. Unlike v1 it normalizes sentence case and standalone `I`. A corpus fingerprint is included in every trace to make style provenance auditable.

```powershell
py -3.12 scripts/preview_fly_styled.py --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --owner-corpus .private/fly_tweets_owner_math_en.jsonl --output .private/fly_v2_styled.jsonl --count 30
```

The comparison is saved in `.private/fly_v2_styled.report.json`. Same saved 117 historical Fly rows, 30 outputs each:

| Metric | Composition v1 | Styled composition v2 |
|---|---:|---:|
| Mean length | 135.9 | 82.9 |
| Contains question | 10.0% | 56.7% |
| Unique texts | 30 | 30 |
| Topics | 15 | 13 |
| Fly-world scenes | 17 | 20 |

**This is not learned humor.** Both variants use hand-authored mathematical and physical facts, and v2 is still visibly formulaic (`Is this ...?`, `Field note on ...`). Pure surface-style matching is not a sufficient success criterion. Offline only; neither generated texts nor corpus are uploaded to Buffer.

## Mathematical topic prior learned from owner originals

An optional prior `src/fly_window/text/topic_prior.py` counts specific math concepts in the owner's **original Russian/English tweets**, without using the noisy machine-translated text. It is an interpretable topic-frequency model, not a generative language or humor model. An explicit math token generated by the Fly always takes precedence over this fallback prior. Otherwise, a hash of Fly's actual sampled math-token stream is drawn against smoothed source frequencies (`sqrt(count+1)`), so rare topics remain possible.

Of 392 selected owner posts, 131 match at least one of the conservative 16 bilingual patterns. In this exploratory pass, examples include symmetry/group language in 56 posts, isomorphism/homeomorphism in 32, continuity/differentiability in 24 and convergence/limits in 19. These are **approximate regex-based topical families, not research-area labels**; e.g. group theory does not always mean physical symmetry. The other 261 posts are not assumed to be irrelevant, merely unclassified.

Use the same style-preview script with `--topic-prior-from-owner` to enable the prior and save its source counts alongside the generated sample:

```powershell
py -3.12 scripts/preview_fly_styled.py --outbox "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\outbox.jsonl" --owner-corpus .private/fly_tweets_owner_math_en.jsonl --output .private/fly_v2_styled_prior.jsonl --count 30 --topic-prior-from-owner
```

On the historical 117-token-trace input, this variant yields 30 unique captions spanning 13 math topics and 18 scenes. Mean length is 83.9 characters and 53.3% contain questions. **These metrics describe the surface form, not humor quality**, and the sample still contains repetitive sentence openings. This is opt-in/offline, not a production default.

## Observed pilot results (2026-11-01)

- The owner's October 26 X export contains **12,142** posts. The math-topic candidate filter selected **392** originals (370 RU, 22 non-Russian).
- Local `RU → EN` translation pass is **complete as a draft**: 355 machine drafts, 15 editor drafts (11 initial + 4 manual repairs), and 22 already-English originals. These are **not** reviewed or training-approved. Automated translations demonstrably mishandle some humor, formal notation and informal idioms; human correction is required.
- A local `.private/fly_v2_review.csv` contains **120** candidates for selective human review. None is approved by this system.
- Using 117 saved Fly-generated rows, the offline semantic contrast decoder yielded **15** distinct-topic previews after lexical near-duplicate removal. All are `publishable: false`.
- `@alcuintweets` and `@mouse_math` are registered as candidate sources. No complete third-party dataset has been ingested; brief retweet excerpts in the owner's archive do not constitute usable full originals.
- This version is a **seed-bank decoder**, not a newly trained joke-producing model. We must evaluate originality, style, mathematical correctness and whether people actually find the results funny before attempting any production promotion.

## Native X delivery fallback (2026-11-02)

The production Buffer account returned HTTP **429** with `RATE_LIMIT_EXCEEDED`, `window: "30d"`. Buffer's 30-day quota is shared across API keys, so rotating credentials or aggressive retries will not fix it. The already-authorized X profile on the Windows runner is `@fly_topology` (verified by the collector's read-only login check). The native browser delivery fallback is **live as of 2026-11-02**, using the existing X authorization without creating new Buffer credentials.

`scripts/queue_fly_v2_live.py` takes existing generated Fly token trajectories, owner-original corpus, and a durable native-posting queue. It only creates new posts from **future**, not yet Buffer-scheduled, `generated` rows. It preserves actual Fly-selected publication times and never alters a previously queued/posting/published row. Human-authorized one-time smoke posts require an explicit `--smoke-source`; otherwise history cannot be reposted. The unique X queue ID records whether a post was scheduled or a smoke test. State is persisted atomically.

```powershell
py -3.12 scripts/queue_fly_v2_live.py --rows-json "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\rows.json" --owner-corpus .private/fly_tweets_owner_math_en.jsonl --queue-json "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\v2-native-x-queue.json"
```

### Native X UI sender and idempotency

`collector/x-native-publish.js` uses the already authenticated X Playwright Chromium profile and **always verifies** the account's sidebar profile link matches `--expected-handle` before typing. Live submission requires `--allow-live` (otherwise dry-run), a durable queue record, and the due time actually passing.

Before submission, the sender searches the authenticated account's own recent posts for an identical text and canonical `/status/<id>` permalink. It then persists a `publishing` claim **before** clicking the Post button. After clicking, it verifies the text and canonical permalink independently via the author's timeline and only then marks `published`. If the result is ambiguous after clicking, the record becomes `uncertain` and **will not automatically retry**; the next run attempts read-only reconciliation. Single-job lock, minimum 20-minute gap, one post per invocation, six-hour expiry window for overdue cadence posts, and no automatic publication of old outbox rows are enforced. No Buffer API request is made. Profile access may require reauthentication or a UI selector update; errors are logged, never silently treated as sent.

```powershell
cd collector
npm install --omit=dev
node x-native-publish.js --queue-json "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\v2-native-x-queue.json" --profile-dir C:\\tools\\x-stats\\x-browser-profile --expected-handle fly_topology --dry-run
# For the explicit user-approved live rollout, replace --dry-run with --allow-live.
```

The fallback is a **browser-operated posting method**, not the Buffer API. Running a browser against X is more brittle than a supported API, so continue to verify real permalink success and session health. The queue JSON contains public posts but should remain out of Git history.

### Delivery operations and real smoke verification

`scripts/reconcile_native_x_rows.py` copies only **verified** native publications into the durable `rows.json` runtime. It preserves the Fly's original token IDs and text under `fly_original_text` for audit, marks the rendered text `published` and saves the verified X permalink. A one-time historical smoke is inserted as a separate uniquely keyed v2 row instead of rewriting past V1 rows. This allows the existing metrics collector to observe engagement on the v2 tweets. The reconciler is idempotent and never overrides a Buffer-accepted post.

`scripts/run_fly_v2_native.ps1 -AllowLive` is the single-run orchestration entrypoint: reconcile already-published native posts → enqueue only future Fly-selected slots → submit at most one **due** post through authorized X UI → reconcile published statuses. Without `-AllowLive` it uses a dry run. It requires the persistent account's Chromium profile and a working Playwright install under `collector/node_modules`. Because Playwright needs the existing local Windows profile, a native scheduled task must run under the same Windows account.

**Live smoke proof (2026-11-02):** the browser posted "The swatter found the exact place I was sitting. Is this measure zero?" and independently verified `https://x.com/fly_topology/status/2117130124948914482`. The smoke row was durably reconciled. Three more v2 drafts remain awaiting **2026-11-03** Fly-selected times: **02:43, 04:13 and 06:13 Europe/Belgrade** (01:43, 03:13 and 05:13 UTC). The Windows task polls every 15 minutes, so posting can occur in the following polling interval, if the PC is on, the Windows account logged in, and X remains authenticated. They are **queued locally, not yet posted to X**. No additional smoke is scheduled.

The composer is still experimental. Queueing does not itself publish and does not bypass the need for a trustworthy X session, strict account checks, idempotent delivery and actual tweet-link verification. Do not claim a tweet is published unless the X UI returns and independently verifies its canonical permalink.

### Windows scheduled delivery

The local Windows scheduled task is `FlyTweets-V2-Native-X`. It runs under `GEORGII\\khoru` with an interactive logon, repeating every **15 minutes**, executes `scripts/run_fly_v2_native.ps1 -AllowLive`, and never publishes before a Fly-selected due time. Installation is idempotent:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/install_fly_v2_task.ps1
Get-ScheduledTaskInfo -TaskName FlyTweets-V2-Native-X
Get-Content "$env:LOCALAPPDATA\\flytegral\\fly-tweets\\state\\v2-native-x-run.log" -Tail 15
```

The schedule requires the user's PC to be running **with that Windows account logged in** and the saved `C:\\tools\\x-stats\\x-browser-profile` still logged into `@fly_topology`. It is not a cloud queue. If the PC sleeps, future delivery can be delayed; missed cadence slots older than six hours are skipped rather than posted all at once on resume. Browser profile contention with the older X metrics collector causes safe retry, not a second click. The task's 364-day repetition must be renewed before expiry if still needed.

**Legacy Buffer handoff completed (2026-11-02):** the old `main` GitHub Actions orchestrator still checks out `feature/fly-tweets-v1` to produce the raw Fly neural trajectories and cadence. Its source branch now contains a durable `.github/fly-tweets-publication-paused` marker at commit `7b20585112c6c92f58d35cc9cf96a9e4b74a2e9d`; the existing `scripts/sync_fly_tweets_sheet.py --action ingest` was exercised and returned `{"paused":true,"batches":0}`. The direct Buffer fallback in the orchestrator only runs when Sheet ingest *fails*, whereas the paused ingest completes normally. This preserves neural generation while preventing legacy text submission. In the connected `Buffer Queue` Google Sheet, tab `FlyTweets`, three stale future v1 entries `batch-2026-11-01-a1:0`, `:1`, `:2` were also explicitly changed from `generated` to `paused` in `M116:M118`, then reread and verified. Their already independently queued v2 native counterparts are unaffected. This disables the known dual-publication path; it does not guarantee that every external Buffer/App Script trigger has stopped if someone creates other untracked sources. The v1 generation branch also now includes commit `354f1068e1c4c2038bf1a45d672518091056acbf`: its `reconcile_sheet_rows()` preserves a locally **verified native X publication**, including the v2 text, URL, and status, when an older Sheet snapshot says `paused` or `generated`; it never overwrites an actual Buffer-owned post. This prevents the daily Sheet sync from losing the v2 permalink before X metrics collection.

## Success criteria

The owner authorized a **limited live v2 pilot**, and the first post is verified above; this is not a claim that the neural policy has learned humor. Continue comparing v0/v1/v2 on fixed Fly seeds: grammatical completeness, interesting mathematical–fly concepts, genuine human pairwise preference, exact/near duplicates, and length/EOS behavior. The live native queue is permitted only on due Fly-chosen times with explicit X permalink verification. No unreviewed expansion to additional accounts, no bulk replay of older v1 text, and no fabricated quality labels.
