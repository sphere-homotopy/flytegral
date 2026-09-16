# Fly Tweets v1 Design

## Goal

Turn the existing MaleCNS/connectome-based Flytegral codebase into a fully autonomous English-language X/Twitter author that writes 10 tweets per day, publishes them through the existing Google Sheets -> Buffer pipeline with stochastic timing, collects engagement statistics, performs one daily learning update, and then immediately generates the next batch of 10 tweets.

No LLM or other language model may participate in tweet generation, filtering, rewriting, ranking, reward assignment, vocabulary expansion, or daily training.

## Hard product invariants

1. The author is the fly system. External code may encode words into sensory stimulation, decode activity into token logits, schedule posts, collect metrics, and execute deterministic training algorithms, but it may not choose or rewrite tweet content.
2. All runtime operations are invisible to the user in normal operation. No terminal, browser, Python, Playwright, confirmation dialog, or training window may appear on the desktop.
3. The home computer may be offline. Daily training work must remain durable until the self-hosted worker becomes available, then execute automatically.
4. Each successful daily training cycle produces the next 10 tweets immediately from the newly updated checkpoint.
5. Generated tweets and their exact generation provenance must be auditable: checkpoint id/SHA, RNG seed, token sequence, log probabilities, generation timestamp, and batch id are stored.
6. No rejection sampling based on meaning, humor, style, engagement prediction, or human/AI quality judgment. Technical safety gates may reject malformed sequences only before the public-launch gate described below.
7. English only for v1.

## Starting point

Branch `feature/fly-tweets-v1` is based on `feature/fly-window-v1`.

The existing fly policy uses the MaleCNS-derived recurrent graph but resets neural activity on every `forward()` call. Fly Tweets therefore needs a stateful text-specific policy while reusing the graph-loading and sparse recurrent infrastructure.

## Architecture

The system has seven independent units:

1. `FlyTextPolicy`: stateful connectome dynamics plus trainable sensory encoder and token readout.
2. `FlyVocabulary`: fixed 1024-token English/math/fly vocabulary.
3. `FlyPretrainer`: deterministic supervised curriculum for initial language competence.
4. `FlyGenerator`: autoregressive tweet generation and provenance capture.
5. `FlyPublisher`: Google Sheet queue schema plus Buffer scheduling with stochastic timing.
6. `FlyMetrics`: DOM-based X statistics collection and row reconciliation.
7. `FlyDailyTrainer`: reward calculation, one daily policy update, checkpointing, and generation of the next 10 tweets.

GitHub Actions provides durable orchestration. Training itself runs only on the Windows self-hosted runner / home PC.

## Fly text model

### Stateful connectome

For token step t:

- token id `x_t` is converted to a fixed-size sensory code,
- the sensory code drives the MaleCNS input neurons,
- recurrent microsteps update the persistent neural activity `h_t`,
- output-neuron activity is mapped to logits over the 1024-token vocabulary,
- the sampled token is fed back as the next sensory input.

Conceptually:

`h_t = FlyCNS(h_{t-1}, encode(x_t))`

`logits_t = W_out * h_t[out_neurons] + b_out`

`x_{t+1} ~ Categorical(logits_t / temperature)`

The biological recurrent adjacency/weight matrix remains fixed in v1. Trainable components are the word-to-sensory encoder, input gain/bias, output readout, and optional scalar temperature parameters. This preserves the connectome as the nonlinear recurrent substrate while allowing the system to learn a text interface.

### State handling

The policy exposes explicit state operations:

- `initial_state(batch_size)`
- `step(token_ids, state) -> logits, next_state, activity`
- `detach_state(state)` for truncated backpropagation

A tweet starts from a zero/reset state and `<BOS>` token. State persists until `<EOS>` or the hard token/character cap.

## Vocabulary

Vocabulary size is exactly 1024 tokens. It is versioned and immutable inside a trained checkpoint; changing it requires a new major vocabulary version.

Target composition:

- 32 control/special tokens
- 64 punctuation, operators, mathematical symbols, and formatting tokens
- 320 high-frequency English/function/general words
- 480 mathematical/scientific words
- 128 conversational, internet, fly, and meme words

The final build step must assert exactly 1024 unique tokens.

### Mathematical coverage

The math/science block must cover at least:

- analysis and calculus
- linear and abstract algebra
- topology and geometry
- probability and statistics
- number theory and combinatorics
- logic and set theory
- category theory and homological language
- dynamical systems and differential equations
- mathematical physics
- algorithms and theoretical computer science
- meta-mathematical words such as proof, theorem, lemma, conjecture, counterexample, trivial, canonical, invariant, natural, almost surely, generic, finite, infinite, local, global

### Fly/meme layer

The 128-token conversational block deliberately gives the fly a small recognizable voice without making meme tokens dominant. It includes ordinary short-form English and a controlled fly/meme subset such as:

`buzz`, `bzz`, `bzzz`, `fly`, `flies`, `wing`, `wings`, `brain`, `neuron`, `synapse`, `banana`, `fruit`, `tiny`, `human`, `humans`, `lol`, `lmao`, `wtf`, `hmm`, `ok`, `nope`, `yes`, `why`, `actually`, `literally`, `basically`, `apparently`, `proof`, `theorem`, `sigma`, `epsilon`, `topology`, `integral`, `category`, `random`, `chaos`.

Variants like `buzz buzz` are generated compositionally from repeated tokens rather than occupying many dedicated vocabulary slots.

The build process must keep fly/meme-specific tokens below 5 percent of the total vocabulary so they cannot dominate purely by prior frequency.

## Initial training

The initial model must be trained before public posting so the first tweets are not arbitrary token soup.

No synthetic text from an LLM is allowed.

Training occurs in three stages.

### Stage A: deterministic grammar curriculum

A hand-written deterministic grammar generates short grammatical mathematical English from the fixed vocabulary. Examples of permitted structural families:

- `every <noun> has <property>`
- `if <statement> then <statement>`
- `there exists <noun> such that <statement>`
- `why is <noun> <property> ?`
- `<noun> is not <noun>`
- `<noun> preserves <noun>`
- `<noun> converges to <noun>`
- `bzz <short mathematical statement>`

The grammar supplies structure, not a hidden author: templates and lexical classes are static project data, and examples are sampled mechanically.

### Stage B: real English mathematical corpus

Use public, non-LLM English mathematical text. Preprocessing must:

- lowercase/canonicalize according to the fixed vocabulary rules,
- remove URLs and unsupported markup,
- map unsupported tokens to `<UNK>` or drop sentences above the OOV threshold,
- split into tweet-sized sequences,
- retain only English text,
- never call an LLM for cleanup or rewriting.

Teacher forcing trains next-token prediction.

### Stage C: self-prefix robustness

Scheduled sampling gradually replaces some teacher-forced previous tokens with the fly's own sampled tokens. This reduces the train/generate mismatch and makes autoregressive generation less likely to collapse immediately.

## Public-launch quality gate

The gate is mechanical and does not judge semantic quality. Before automatic public posting is enabled, the checkpoint must satisfy all configured thresholds on held-out data and generated probes:

- held-out next-token loss below the configured ceiling,
- at least 95 percent of probe generations terminate with `<EOS>` before the hard cap,
- repeated-token run length never exceeds the configured degeneration limit in at least 95 percent of probes,
- median generated tweet length lies inside the configured range,
- no NaN/Inf logits or state values,
- deterministic replay from saved seed/checkpoint reproduces the exact token trajectory.

After this gate has been passed for the deployed model lineage, generated tweets are not semantically filtered.

## Tweet generation

Each daily batch contains exactly 10 tweets.

Generation rules:

- begin with `<BOS>` and reset neural state,
- sample autoregressively from the fly logits,
- stop on `<EOS>`, hard token cap, or 240 rendered characters,
- use a fixed configurable temperature/topology of sampling that does not depend on text meaning,
- persist the complete trajectory and log probability for later learning,
- never ask another model to rank or choose between candidates.

If a sequence hits a hard technical cap without `<EOS>`, it is stored with a technical termination reason. The publishing policy may withhold only structurally invalid output such as an empty rendered string; it may not replace it with a better candidate.

## Google Sheet contract

Use a dedicated Fly Tweets sheet/tab with at least these columns:

- `batch_id`
- `tweet_index`
- `text`
- `generation_time`
- `checkpoint_id`
- `git_sha`
- `rng_seed`
- `token_ids`
- `token_logprobs`
- `status`
- `scheduled_at`
- `buffer_post_id`
- `tweet_url`
- `published_at`
- `views`
- `likes`
- `reposts`
- `replies`
- `bookmarks`
- `metrics_collected_at`
- `reward`
- `training_consumed_at`
- `last_error`

Rows are append-only for generation provenance. Operational/status fields may be updated in place.

## Scheduling and Buffer

Reuse the existing Google Sheets -> Buffer queue concept.

For each 10-tweet batch, assign ten baseline publication slots distributed across the configured waking-day window in `Europe/Belgrade`.

Each slot receives independent stochastic jitter sampled from a zero-mean clipped normal distribution. Initial configuration:

- sigma: 22 minutes
- absolute clip: 45 minutes
- minimum gap after sorting: 35 minutes

The Apps Script must enforce uniqueness/idempotency by batch id plus tweet index before creating a Buffer post.

Scheduling happens without user interaction.

## Metrics collection

Reuse/adapt the old DOM collector approach that reads visible X tweet metrics from the rendered page, including:

- views
- likes
- reposts
- replies
- bookmarks
- tweet URL
- tweet timestamp
- tweet text

The collector reconciles posts to Fly Sheet rows by tweet URL/status id and updates metrics in place.

Metrics collection runs headlessly/background-only. A visible browser window is a defect.

## Reward

Daily learning uses only tweets with sufficient observation age, initially >= 24 hours.

Raw engagement is normalized to reduce dependence on audience size and exposure. A v1 reward is based on:

- log-scaled views/exposure,
- smoothed like rate,
- reply rate,
- repost rate,
- bookmark rate,
- within-cohort centering/rank normalization.

Replies, reposts, and bookmarks receive larger coefficients than likes. Exact coefficients are configuration, versioned with each training run.

The resulting ten-tweet daily rewards are centered so daily learning changes relative preference rather than blindly maximizing account growth noise.

## Daily learning

Daily update uses the stored generation trajectories. The initial algorithm is a small policy-gradient update on token log probabilities with:

- normalized tweet-level reward,
- conservative learning rate,
- entropy regularization,
- KL/anchor penalty toward the initial language checkpoint,
- gradient clipping,
- maximum update-step budget per day.

The anchor term prevents one viral but degenerate tweet from destroying basic language competence.

A successful daily cycle is atomic at the logical level:

1. fetch latest eligible metrics,
2. compute rewards,
3. train from the currently deployed checkpoint,
4. write a new immutable checkpoint plus training manifest,
5. mark consumed reward rows,
6. generate the next 10 tweets from the new checkpoint,
7. append their rows to the Sheet.

Idempotency keys prevent duplicate consumption or duplicate batches if a job retries.

## GitHub orchestration and offline PC

A scheduled GitHub workflow creates/maintains a durable daily training request keyed by date. The request is not lost if the home computer is offline.

A Windows self-hosted runner executes pending requests whenever it is online. The worker must run non-interactively and without visible windows.

After successful completion, the request is marked complete with the produced checkpoint id and batch id.

If several days accumulated while the PC was offline, process them serially. Do not run concurrent updates against the same model lineage.

## Background-only Windows requirement

Normal operation must produce no visible UI.

Required execution properties:

- self-hosted runner installed/running as a Windows service,
- Python workers launched without console windows,
- Playwright/Chromium always headless for metrics collection,
- no desktop browser automation,
- no modal prompts,
- credentials/configuration read from service environment, encrypted credential storage, repository secrets, or existing Script Properties,
- logs written to files/GitHub artifacts rather than shown on screen.

Any workflow requiring the user to click, confirm, keep a terminal open, or manually start a browser is out of scope for the final autonomous system.

## Failure behavior

- Offline PC: leave work pending and resume automatically later.
- X/DOM layout change: metrics job fails closed, records the error, and does not train on missing/fabricated metrics.
- Buffer/Sheet transient error: retry idempotently; never duplicate posts.
- Training numerical failure: retain previous deployed checkpoint and do not generate a new batch from a failed checkpoint.
- Generation structural failure: record the failed sample and retry only when the rendered output is technically invalid, never because it is unfunny or nonsensical.
- Corrupt/missing checkpoint: stop the daily lineage and surface the failure in GitHub logs; do not silently reset to an unrelated model.

## Testing requirements

Automated tests must cover:

- exact 1024-token unique vocabulary build,
- meme/fly vocabulary proportion bound,
- state persistence across token steps,
- reset behavior between tweets,
- deterministic trajectory replay from seed/checkpoint,
- max-length and EOS handling,
- Google Sheet row serialization,
- stochastic schedule bounds and minimum gap,
- Buffer idempotency keys,
- metrics parser fixtures based on X aria-label text,
- reward calculation and normalization,
- prevention of double-consuming a training row,
- daily-job idempotency,
- offline/pending-job state transitions,
- failure rollback to previous checkpoint.

## Non-goals for v1

- LLM assistance of any kind in runtime content generation.
- Images or video generation.
- Replies to other users.
- Reading the live X timeline as language-model context.
- Direct modification of biological recurrent connectome weights.
- Multiple languages.
- Human approval queue before each post.

## Success criteria

The system is complete when, without user interaction or visible desktop processes, a trained MaleCNS-based text policy can:

1. generate 10 English math/fly tweets,
2. append them with provenance to Google Sheets,
3. schedule them through Buffer with stochastic timing,
4. collect engagement metrics after publication,
5. perform one conservative daily reward-driven update when the home PC is available,
6. checkpoint the updated policy,
7. immediately generate the next 10 tweets,
8. survive PC-offline periods without losing or duplicating work.
