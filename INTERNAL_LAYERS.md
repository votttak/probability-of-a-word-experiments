# Internal-layer surprisal comparisons

This opt-in pipeline compares n-gram surprisal (N) and context-limited LM
surprisal (C) with full-context internal-layer surprisal (L). It preserves the
existing N-vs-C pipeline and writes to separate layer experiment directories.

## Method

- L is a GPT-style logit lens over transformer block outputs. Layer 1 is the
  first block output; the embedding stream is excluded. For GPT-2 small the
  emitted layers are 1 through 12. The reference internal-layers repository
  exposes hidden-state index 0, but this experiment deliberately defines L
  over transformer blocks rather than the uncontextualized embedding stream.
- Layers 1 through 11 are decoded with the model's final layer norm and tied
  vocabulary head. Layer 12 uses the ordinary model logits, without applying
  the final norm twice.
- Tokenization, BOS/EOS framing, 1,022-token overlapping chunks, stride 200,
  and weighted word-boundary correction mirror the installed
  wordsprobability 0.17 full-context scorer.
- Layer 12 is checked against the canonical ordinary surprisal both after
  scoring and during merge. The canonical absolute tolerance is 0.0005.
- N, C, and L use one shared complete-case sample and one shared random-word
  fold assignment. Every predictor includes its current value and three
  text-bounded spillovers.
- The evaluator fits N-only, C-only, and L-only models once, then every N+L
  and C+L pair. It never combines multiple layers in one model.

For each pair, the primary quantities are held-out Gaussian log-density
differences per observation:

- delta N given L = score(N+L) - score(L)
- delta L given N = score(N+L) - score(N)
- delta C given L = score(C+L) - score(L)
- delta L given C = score(C+L) - score(C)

Positive values mean that the added family improves held-out prediction.

## Pilot run

Activate the project environment, then run the canonical 500-word pilot:

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    make -f MakefileLayerPilot

The recipe uses the first 50 words from each of the 10 Natural Stories
passages, GPT-2 small layers 1-12, N contexts 0-4, C contexts 1-4, 10 folds,
and seed 42. It subsets the canonical full N+C+RT table rather than rebuilding
the known-different local RT base or calling the n-gram API.

A faster smoke run is:

    make -f MakefileLayerPilot PILOT_WORDS_PER_TEXT=10 PILOT_FOLDS=5

To select layers, include the final layer so the ordinary-surprisal anchor
remains active:

    make -f MakefileLayerPilot INTERNAL_LAYER_IDS="3 6 9 12"

## Canonical full cluster run

The full target is intentionally fixed to the project's definitive comparison:
Natural Stories, GPT-2 small, N contexts 0-4, C contexts 1-4, and transformer
layers 1-12. It reuses the canonical N+C+RT table and never reruns the n-gram
API, RT preprocessing, or context-limited scoring.

Before launching, transfer or commit every new source, test, Makefile, and
script in this change. The three canonical scratch inputs must already exist:

- `text_rt_data/natural_stories.txt`
  (SHA-256 `04578a7187ec7edb779362f912df97befc74f7945c4d554902e2049041579da4`)
- the definitive `joint_full/...context_1-2-3-4/joint-data.tsv`
  (SHA-256 `d09c6b35e3a7ee8cb7dea83bc4ee5babc59f413ab0067a0d20ca5c4ab3450b1e`)
- `surprisals_rt_data/suprisal-natural_stories-gpt2-small.tsv`
  (SHA-256 `cd86559a6ef0289ae006b04f365f623531ef8ebb9badb61e12538bd8ae886bc9`)

Use the already verified cluster environment; do not recreate it from the
stale environment YAML. From the repository root:

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate probability-of-a-word
    export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
    export HF_HOME=/pub/hofmann-scratch/huggingface_cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

Inspect the complete command graph without requiring a GPU:

    DRY_RUN=1 bash scripts/run_layer_full.sh

Select one genuinely free GPU, start a tmux session, and run:

    tmux new -s n-vs-l-c-vs-l
    CUDA_VISIBLE_DEVICES=<one-free-id> bash scripts/run_layer_full.sh

The runner refuses zero or multiple visible GPUs, validates the pinned inputs,
checks the cached GPT-2 snapshot
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, and records Git status, package
versions, model dtype/device, GPU information, and checksums. It first performs
a full-text layer-12-only GPU anchor gate, then runs all layers. Rerun the
identical command after interruption: completed passage shards are
configuration- and input-fingerprinted and will be reused.

The full workflow then:

1. requires all 10,256 L keys and words to match the canonical table;
2. requires final-layer drift below `0.0005`;
3. creates text-bounded L spillovers;
4. evaluates every N x L and C x L pair with one 10-fold split and seed 42; and
5. validates all tables before atomically publishing
   `validation-complete.json`.

Expected canonical counts are:

- 10,256 internal-layer and merged rows;
- 10,023 shared complete cases and 233 excluded rows;
- 1,080 fold-result rows: `(5 + 4) * 12 * 10`;
- 108 conditional-delta rows: `(5 + 4) * 12`.

Logs are written below `$RUN_ROOT/logs/layer-full`. Checkpoints are below
`$RUN_ROOT/checkpoints/rt/layer_full`; results are below
`$RUN_ROOT/results/rt/layer_full`. The validated completion JSON contains
artifact hashes and final-layer max/mean/p99 diagnostics.

## Outputs and interpretation

Pilot checkpoints are under `checkpoints/rt/layer_pilot`; full checkpoints
use `checkpoints/rt/layer_full`. Each result directory contains:

- fold-results.tsv: one row per pair and fold, with explicit N/C context,
  layer, model scores, and both conditional delta directions.
- conditional-deltas.tsv: fold means and standard errors for each pair.
- summary.tsv: row counts, folds, contexts, layers, response, and controls.

The full run evaluates the entire layer curve, but it does not perform nested
layer selection. A post-hoc test-fold argmax is exploratory. Pre-specify a
layer or add nested cross-validation before making a confirmatory
"best-layer" claim.

## Preparation verification

The local full-text layer-12 rollover check covered all 10,256 words and all
20 overlapping chunks. Against the pinned ordinary-surprisal file, maximum
absolute drift was `0.000383918`, mean drift was `0.0000403282`, and
nearest-rank p99 drift was `0.000174107`; all passed the `0.0005` gate.
The exact merge and layer-12-only 10-fold evaluation produced 10,023 complete
cases, 90 fold rows, and 9 aggregate rows. Their shared M0, N-only, and C-only
scores matched the definitive N-vs-C results to about `1e-14`.
