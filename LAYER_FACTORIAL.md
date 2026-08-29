# Internal-layer factorial replication

This experiment isolates the three implementation differences most likely to
explain the gap from Kuribayashi et al.:

1. extraction context: passage or sentence;
2. word score: corrected surprisal or the historical `surprisal_buggy`;
3. decoder: logit lens or tuned lens.

## One experiment switchboard

All experiment choices now live in
`configs/layer_factorial.json`. Edit that one file to select:

- models and reading-time responses (`time`, `paper_time`);
- passage/sentence context, corrected/buggy scores, and logit/tuned lens;
- embedding-layer inclusion and the sentence first-token policy;
- regression mode, lag policy, early-layer threshold, and the
  transformer-only sensitivity analysis;
- extraction jobs, CPU threads, pivot size, canonical inputs, and local
  output roots.

The committed file selects the complete replication grid. Lists are switches:
for example, `"contexts": ["sentence"]` and
`"lens_methods": ["tuned-lens"]` run only that selected cell grid.
Command-line flags remain available as explicit one-run overrides and take
precedence over the JSON values. Every run manifest records the raw config
path/hash, the fully resolved effective settings/hash, and which CLI options
overrode the file.

Inspect values without copying them into shell scripts:

```bash
python src/h01_data/layer_factorial_config.py \
  --get switches.responses
python src/h01_data/layer_factorial_config.py --list-models
```

In the default grid, both score families are produced from the same forward
pass. The four context/decoder extractions are evaluated with both scores,
giving eight cells per reading-time response.

## Settings held fixed

- The experiment registry pins one immutable base-model revision for all four
  extraction cells. Lens base identity and artifact hashes are validated
  separately; the official Pythia lens configs have a null
  `base_model_revision`.
- Layers include the embedding output (layer 0) and every transformer block
  output through layer D.
- Sentence cells use Kuribayashi-style beginning-of-word scoring at sentence
  starts. Passage cells use the ordinary BOS-conditioned first-word score.
- The paper-exact evaluator excludes sentence-initial words, pads sentence
  lags with global predictor means, and fits the paper's additive current,
  lag-1, and lag-2 length/frequency controls.
- Two RT responses are reported: the project's canonical `time` and the
  official Natural Stories `meanItemRT` mapped to `paper_time`.
- The sentence map, keyed paper RT, and `wordfreq==3.1.1` controls are
  committed as small immutable inputs. Cluster runs do not depend on a local
  Natural Stories clone or on a runtime frequency lookup.

The output validator runs before any regression. It requires complete layer
families, identical keys across cells, distinct intermediate tuned/logit
predictions, matching final-layer predictions, and matching model revision,
text, sentence policy, manifest, and tuned-lens hashes.

## Local pivot result

The completed GPT-2-small pivot uses the first two complete sentences from
each of the ten Natural Stories texts: 506 input words and 486 analysis rows
after excluding 20 sentence-initial words.

With `paper_time`, all four tuned-lens cells select layer 1 (depth 8.3%).
The logit-lens cells select layers 4, 5, or 12. Thus 4/8 cells are in the first
20% of model depth.

With the project's `time`, every cell selects layer 11 or 12. Thus 0/8 cells
are in the first 20%.

This pivot supports a specific diagnostic conclusion: tuned-lens decoding and
the RT preprocessing/response jointly recover the early-layer direction.
Sentence-bounded extraction or buggy aggregation alone do not. Because this
is a small one-model sample, it does not establish the cross-model result.

The combined report is at
`results/rt/layer_factorial/pivot-gpt2-small/run/combined/REPORT.md`.
The extraction validation and full run provenance are at
`checkpoints/rt/layer_factorial/pivot-gpt2-small/run/`.

## Local commands

```bash
make -f MakefileLayerFactorial layer_factorial_pivot \
  PYTHON=/home/durnovv/miniconda3/envs/probability-of-a-word/bin/python \
  FACTORIAL_JOBS=1 FACTORIAL_THREADS_PER_JOB=4
```

Run the full 10,256-word experiment with:

```bash
make -f MakefileLayerFactorial layer_factorial_full \
  PYTHON=/home/durnovv/miniconda3/envs/probability-of-a-word/bin/python \
  FACTORIAL_JOBS=1 FACTORIAL_THREADS_PER_JOB=4
```

Do not raise `FACTORIAL_JOBS` on a single GPU. Each extraction loads
a full model, so the four context/decoder cells must run sequentially.

## Supported full models

The supported tuned-lens grid contains the nine-model overlap with the earlier
N/C/L experiment:

- `gpt2-small`, `gpt2-large`, and `gpt2-xl`;
- `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-14b`,
  `pythia-28b`, and `pythia-69b`.

The Pythia aliases load Kuribayashi's deduplicated model repositories, not the
non-deduplicated checkpoints used by the earlier N/C/L extraction. The registry
pins `step143000` as our documented final-checkpoint resolution and records
its immutable commit in every run. This is not a recovered lens-training SHA:
the official Pythia lens configs identify the base repository but have a null
`base_model_revision`. The pre-existing joint TSVs provide only RT rows, word
keys, and controls; their old surprisal columns are not predictors here.

GPT-2 Medium is not included because the official tuned-lens repository has no
GPT-2 Medium lens. Kuribayashi also evaluated Pythia 1B/12B and selected OPT
models, but they are outside the model overlap being diagnosed here. Every
included base-model revision and both tuned-lens files are pinned and checked.

## Cluster preparation

The project convention is a manual `tmux` session on
`mark.inf.ethz.ch`. Preparation may use the network; the actual
experiment is forced offline after an exact cache preflight.

After pulling this commit and activating the project's Conda environment:

```bash
cd "$HOME/probability-of-a-word-experiments"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate probability-of-a-word
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME="$RUN_ROOT/huggingface_cache"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export TUNED_LENS_ROOT="$RUN_ROOT/resources/tuned-lens"
export TUNED_LENS_PYTHONPATH="$RUN_ROOT/python/tuned-lens-0.2.0"
mkdir -p "$TUNED_LENS_PYTHONPATH"
python -m pip install --target "$TUNED_LENS_PYTHONPATH" --no-deps "tuned-lens==0.2.0"
python scripts/stage_layer_factorial_resources.py --all --config configs/layer_factorial.json --lens-root "$TUNED_LENS_ROOT" --hf-home "$HF_HOME"
```

The active environment must provide `wordsprobability==0.17`. The
preflight records the installed GPU stack and scientific Python versions.
Verify every staged resource without network access:

```bash
export PYTHONPATH=$TUNED_LENS_PYTHONPATH:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/stage_layer_factorial_resources.py --all --config configs/layer_factorial.json --verify-only --lens-root "$TUNED_LENS_ROOT" --hf-home "$HF_HOME"
```

The staged lenses come from immutable `AlignmentResearch/tuned-lens`
Space revision `1ac7285852a22309f571c2555efc37375d0c4cda`. Complete
base revisions and artifact hashes live in
`src/h01_data/layer_factorial_models.py`.

## Cluster run

Start a persistent shell, select exactly one GPU, and dry-run the full
nine-model sequence:

```bash
tmux new -s layer-factorial
cd "$HOME/probability-of-a-word-experiments"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate probability-of-a-word
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME="$RUN_ROOT/huggingface_cache"
export HF_HUB_DISABLE_XET=1
export TUNED_LENS_ROOT="$RUN_ROOT/resources/tuned-lens"
export TUNED_LENS_PYTHONPATH="$RUN_ROOT/python/tuned-lens-0.2.0"
export CUDA_VISIBLE_DEVICES=0
export EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
scripts/run_all_layer_factorial_cluster.sh --dry-run
scripts/run_all_layer_factorial_cluster.sh
```

To run or resume only one model:

```bash
scripts/run_layer_factorial_cluster.sh pythia-69b
```

Rerunning the same command is safe. Passage checkpoints are reused only after
their model revision, input, sentence policy, and lens identity pass
validation. A per-model file lock prevents duplicate live runs. Each model
writes:

- checkpoints and `run-manifest.json` under
  `$RUN_ROOT/checkpoints/layer-factorial/MODEL/`;
- its combined report under
  `$RUN_ROOT/results/layer-factorial/MODEL/combined/REPORT.md`;
- timestamped logs under `$RUN_ROOT/logs/layer-factorial/MODEL/`.

With the committed configuration, both `time` and `paper_time` are evaluated
for all eight context/score/lens cells. A reduced switch selection evaluates
only its requested grid. The output validator must pass before any regression
is fitted.

## Cross-model analysis

After all selected per-model combined reports exist, run the strict
cross-model postprocessor:

    python src/h03_paper/analyze_cross_model_layer_factorial.py \
      --config configs/layer_factorial.json \
      --run-root "$RUN_ROOT" \
      --output-dir "$RUN_ROOT/results/layer-factorial/cross_model_analysis"

The same command works on a relocated compact archive when --run-root
points to the extracted directory containing both results/layer-factorial
and checkpoints/layer-factorial. It requires the models and factorial cells
selected by the configuration. An explicit `--models` subset remains available
for testing or partial diagnostics.

For modern manifests, the postprocessor uses the common effective scientific
setup recorded by the runner. This means explicit runner overrides such as
`--contexts sentence` or `--early-layer-threshold 0.1` remain analyzable
without editing the source JSON after the run. It verifies the raw config hash,
the canonical effective-config hash, and the corresponding recorded CLI option,
and requires every selected model to have identical effective scientific
settings. A mixed set of modern and legacy manifests is rejected. A fully
legacy archive with no configuration metadata continues to use the supplied
JSON settings.

The postprocessor independently revalidates every selected layer curve,
recomputes each exact delta-LL argmax, reconciles the stored best-layer tables
and summaries, checks registry depths and pinned revisions, verifies the
archived extraction-validation hashes, and requires shared text, RT,
frequency, and sentence-manifest hashes across models.

It always reports the including-embedding scope when layer 0 is selected. The
`analysis.transformer_only_sensitivity` switch optionally adds the
transformer-only scope. If layer 0 is disabled, transformer-only is the sole
scope:

- **including-embedding**, matching the extraction grid with layer 0 eligible;
- **transformer-only**, which excludes layer 0, recomputes the argmax over
  layers 1 through D, and retains the architectural depth definition
  `layer / D <= analysis.early_layer_threshold`.

## Full nine-model result

The full paper-motivated cell (paper_time, sentence context, tuned lens,
buggy surprisal) selects a layer in the first 20% for 9/9 models when layer 0
is eligible and 8/9 models after re-optimizing over transformer layers only.
The earlier project-style baseline (time, passage context, logit lens,
corrected surprisal) reaches only 3/9 under either scope.

Sentence-bounded context is the robust explanation for this difference. Over
transformer layers, changing passage to sentence context moves the optimum
earlier in 59/72 matched factorial cells, leaves 13 unchanged, and moves none
later. Tuned-lens decoding has a smaller earlyward effect (42 earlier, 17
unchanged, 13 later). Changing corrected to buggy surprisal leaves 60/72
optima unchanged. Changing time to paper_time is sensitive to embedding
eligibility and is not the dominant full-data factor.

The local report generated from the compact cluster archive is at
results/rt/layer_factorial/cross_model_analysis/REPORT.md. Machine-readable
condition, model-response, paired-factor, integrity, and best-layer tables are
written beside it.
