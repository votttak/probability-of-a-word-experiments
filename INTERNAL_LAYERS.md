# Internal-layer surprisal comparisons

This opt-in pipeline compares n-gram surprisal (N) and context-limited LM
surprisal (C) with full-context internal-layer surprisal (L). It preserves the
existing N-vs-C pipeline and writes to separate layer experiment directories.
The canonical model set is exactly the ten models with completed C
checkpoints.

## Models and reference provenance

Model metadata has one source of truth:
`src/h01_data/internal_layer_models.py`. Repository aliases, Hugging Face
models, and transformer-block depths are:

| Alias | Hugging Face model | Family | Blocks/L layers | Ordinary reference | Anchor gate |
|---|---|---|---:|---|---:|
| `gpt2-small` | `gpt2` | GPT-2 | 12 | tracked canonical | `0.0005` |
| `gpt2-medium` | `gpt2-medium` | GPT-2 | 24 | fresh | `0.0005` |
| `gpt2-large` | `gpt2-large` | GPT-2 | 36 | fresh | `0.0005` |
| `gpt2-xl` | `gpt2-xl` | GPT-2 | 48 | fresh | `0.0005` |
| `pythia-70m` | `EleutherAI/pythia-70m` | Pythia | 6 | fresh | `0.01` |
| `pythia-160m` | `EleutherAI/pythia-160m` | Pythia | 12 | fresh | `0.01` |
| `pythia-410m` | `EleutherAI/pythia-410m` | Pythia | 24 | fresh | `0.01` |
| `pythia-14b` | `EleutherAI/pythia-1.4b` | Pythia | 24 | fresh | `0.01` |
| `pythia-28b` | `EleutherAI/pythia-2.8b` | Pythia | 32 | fresh | `0.01` |
| `pythia-69b` | `EleutherAI/pythia-6.9b` | Pythia | 32 | fresh | `0.01` |

List or query the registry without loading a model:

    python src/h01_data/internal_layer_models.py --list
    python src/h01_data/internal_layer_models.py \
      --model pythia-14b --field layer_ids

Only GPT-2 small may reuse the tracked ordinary-surprisal reference, because
that canonical run was empirically anchored. Every other model generates a
fresh ordinary-surprisal reference with the same runtime and cached model
snapshot immediately before L scoring.

Do not compare new Pythia L scores with the old tracked Pythia references. A
pilot found very large cross-runtime/precision drift, so those files are
provenance-incompatible. The `0.01` Pythia gate is solely a numerical check
between stable-float32 L and a fresh native-FP16 wordsprobability reference; it
is not permission to use an old reference. Canonical Make and runner commands
reject arbitrary tolerance overrides. Pythia 12B is excluded because it has no
completed C checkpoint and does not fit the current single-16-GB-GPU loader.

## Method

- L is a GPT-style logit lens over transformer block outputs. Layer 1 is the
  first block output and layer D is the final block output for a D-block model;
  the embedding stream is excluded. The reference internal-layers repository
  exposes hidden-state index 0, but this experiment deliberately defines L
  only over contextualized transformer blocks.
- Layers 1 through D-1 are decoded with the model's final layer norm and
  vocabulary head. Layer D uses the ordinary model logits, without applying
  the final norm twice.
- Tokenization, BOS/EOS framing, 1,022-token overlapping chunks, stride 200,
  and weighted word-boundary correction mirror the installed
  wordsprobability 0.17 full-context scorer.
- The final layer is checked against the selected ordinary reference both
  after scoring and during merge, using the fixed family-specific gate in the
  table above.
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

## Generic pilot runs

The matched pilot supports either 10 or 50 words per each of the ten Natural
Stories passages. It uses the definitive GPT-2-small joint table for RT,
controls, and N; the selected model's completed full C table; and a fresh
same-runtime ordinary reference for the prefix. It then extracts L, builds an
exact model-specific joint, and runs the paired evaluation.

Keep disposable pilot outputs under ignored `.cache` paths. For a 500-word
Pythia-70M pilot:

    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    PILOT_ROOT="$PWD/.cache/layer-pilot/pythia-70m"
    make -f MakefileLayerPilot \
      MODEL=pythia-70m \
      PILOT_WORDS_PER_TEXT=50 \
      INPUT_CHECKPOINT_DIR="$PWD/checkpoints/rt" \
      CHECKPOINT_DIR="$PILOT_ROOT/checkpoints/rt" \
      RESULTS_DIR="$PILOT_ROOT/results/rt"

A faster 100-word smoke is:

    PILOT_ROOT="$PWD/.cache/layer-pilot/pythia-70m-smoke"
    make -f MakefileLayerPilot \
      MODEL=pythia-70m \
      PILOT_WORDS_PER_TEXT=10 PILOT_FOLDS=5 \
      INPUT_CHECKPOINT_DIR="$PWD/checkpoints/rt" \
      CHECKPOINT_DIR="$PILOT_ROOT/checkpoints/rt" \
      RESULTS_DIR="$PILOT_ROOT/results/rt"

All transformer-block outputs are emitted by default. A selective diagnostic
must still include the registry's final layer so the anchor remains active:

    make -f MakefileLayerPilot \
      MODEL=gpt2-medium PILOT_WORDS_PER_TEXT=10 \
      INTERNAL_LAYER_IDS="6 12 18 24" \
      INPUT_CHECKPOINT_DIR="$PWD/checkpoints/rt" \
      CHECKPOINT_DIR="$PWD/.cache/layer-pilot/gpt2-medium/checkpoints/rt" \
      RESULTS_DIR="$PWD/.cache/layer-pilot/gpt2-medium/results/rt"

These are computation pilots, not reduced statistical substitutes for the
full experiment. Large models still require the cluster GPU and populated
offline cache.

## Canonical full cluster runs

The full design is fixed to Natural Stories, N contexts 0-4, C contexts 1-4,
all transformer blocks of the selected model, 10 folds, and seed 42. The
runner stages and byte-verifies the tracked full text, definitive GPT-2-small
N+C joint, and selected model's C file from the repository into scratch. It
never reruns the n-gram API or RT preprocessing.

Use the already verified cluster environment; do not recreate it from the
stale environment YAML:

    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate probability-of-a-word
    export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
    export HF_HOME=/pub/hofmann-scratch/huggingface_cache
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

Inspect one model's complete graph without GPU computation:

    MODEL=gpt2-medium DRY_RUN=1 bash scripts/run_layer_full.sh

For a real run, select one genuinely free GPU and use tmux:

    tmux new -s layer-gpt2-medium
    MODEL=gpt2-medium CUDA_VISIBLE_DEVICES=<one-free-id> \
      bash scripts/run_layer_full.sh

The single-model runner accepts the model through `MODEL`. GPT-2 small remains
the default when `MODEL` is omitted. For a selected sequential sweep, use
either `MODELS` or positional aliases:

    CUDA_VISIBLE_DEVICES=<one-free-id> \
      MODELS="gpt2-medium gpt2-large gpt2-xl" \
      bash scripts/run_all_layer_full.sh

    CUDA_VISIBLE_DEVICES=<one-free-id> bash scripts/run_all_layer_full.sh \
      pythia-70m pythia-160m pythia-410m

With neither `MODELS` nor arguments, the sweep runs all ten registry models in
scale order. It is sequential and fail-fast; do not add `-j` or expose multiple
GPUs. Run `pythia-69b` alone before including it in an unattended sweep:

    MODEL=pythia-69b CUDA_VISIBLE_DEVICES=<one-free-id> \
      bash scripts/run_layer_full.sh

Pythia 6.9B is the 16 GB single-GPU feasibility boundary. The runner's
mandatory offline model-load check is the first memory smoke, and its
final-layer full-corpus anchor smoke must pass before all-layer scoring begins.
An out-of-memory failure stops before the expensive stage; it is not evidence
against the scientific hypothesis.

For GPT-2 small, the runner requires cached model commit
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. For another single model, an
optional `EXPECTED_MODEL_COMMIT=<hash>` pins the cached snapshot. Every run
records Git status, package versions, alias/Hugging Face name, layer count,
reference policy, dtype/device, GPU information, input hashes, and the loaded
model commit.

The per-model workflow is:

1. stage and verify immutable tracked inputs;
2. reuse the tracked GPT-2-small reference or force a fresh ordinary reference;
3. build and preflight the exact model-specific N+C+RT joint table;
4. run a full-corpus final-layer anchor smoke;
5. score all layers with resumable passage checkpoints;
6. evaluate every N x L and C x L pair on the same folds; and
7. validate all invariants before atomically publishing
   `validation-complete.json`.

All models require 10,256 internal-layer and merged rows, with 10,023 shared
complete cases and 233 exclusions. If D is the model depth and F is the fold
count, expected cardinalities are:

- fold-result rows: `(5 + 4) * D * F`;
- conditional-delta rows: `(5 + 4) * D`.

At the canonical F=10:

| Depth | Models | Fold rows | Delta rows |
|---:|---|---:|---:|
| 6 | Pythia 70M | 540 | 54 |
| 12 | GPT-2 small; Pythia 160M | 1,080 | 108 |
| 24 | GPT-2 medium; Pythia 410M/1.4B | 2,160 | 216 |
| 32 | Pythia 2.8B/6.9B | 2,880 | 288 |
| 36 | GPT-2 large | 3,240 | 324 |
| 48 | GPT-2 XL | 4,320 | 432 |

Logs are written below `$RUN_ROOT/logs/layer-full`. Checkpoints are below
`$RUN_ROOT/checkpoints/rt/layer_full`; model-specific results are below
`$RUN_ROOT/results/rt/layer_full/natural_stories-<model>-layers_1-<D>-logit-lens/`.
Rerun the identical command after interruption. Completed passage shards are
configuration-, model-, precision-, runtime-, and input-fingerprinted and are
reused only when their identity matches.

## Outputs and interpretation

Pilot checkpoints are under the selected `CHECKPOINT_DIR/layer_pilot`; full
checkpoints use `CHECKPOINT_DIR/layer_full`. Each result directory contains:

- `fold-results.tsv`: one row per pair and fold, with explicit N/C context,
  layer, model scores, and both conditional delta directions;
- `conditional-deltas.tsv`: fold means and standard errors for each pair;
- `summary.tsv`: row counts, folds, contexts, layers, response, and controls;
- `validation-complete.json` for a validated full run, including artifact
  hashes and final-layer max/mean/p99 diagnostics.

The full run evaluates the entire layer curve, but it does not perform nested
layer selection. A post-hoc test-fold argmax is exploratory. Pre-specify a
layer or add nested cross-validation before making a confirmatory
"best-layer" claim. Compare models by both absolute layer and relative depth
(`layer / D`); raw layer indices are not commensurate across 6- to 48-block
models.

## Completed GPT-2-small baseline

The local full-text layer-12 rollover check covered all 10,256 words and all
20 overlapping chunks. Against the pinned ordinary-surprisal file, maximum
absolute drift was `0.000383918`, mean drift was `0.0000403282`, and
nearest-rank p99 drift was `0.000174107`; all passed the `0.0005` gate.
The exact merge and layer-12-only 10-fold evaluation produced 10,023 complete
cases, 90 fold rows, and 9 aggregate rows. Their shared M0, N-only, and C-only
scores matched the definitive N-vs-C results to about `1e-14`.

The subsequent canonical cluster run over GPT-2-small layers 1-12 completed
with `validated: true`: 10,256 input rows, 10,023 complete cases, 233
exclusions, 1,080 fold rows, and 108 conditional-delta rows. Its definitive
local completion marker is
`results/rt/layer_full/natural_stories-gpt2-small-layers_1-12-logit-lens/folds_10-seed_42/validation-complete.json`.
