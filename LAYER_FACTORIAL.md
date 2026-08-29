# Internal-layer factorial replication

This experiment isolates the three implementation differences most likely to
explain the gap from Kuribayashi et al.:

1. extraction context: passage or sentence;
2. word score: corrected surprisal or the historical `surprisal_buggy`;
3. decoder: logit lens or tuned lens.

Both score families are produced from the same forward pass. The four
context/decoder extractions are evaluated with both scores, giving eight cells
per reading-time response.

## Settings held fixed

- The tuned-lens artifact pins the exact Hugging Face model revision. All four
  extraction cells use that same revision.
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

The faithful tuned-lens grid is defined for `gpt2-small`,
`gpt2-large`, and `gpt2-xl`. Every base-model revision and
both files in every tuned-lens artifact are pinned and hash-checked.

GPT-2 Medium is not included because the official tuned-lens repository has no
GPT-2 Medium lens. Pythia is not silently substituted: Kuribayashi's tuned-lens
commands use deduplicated Pythia checkpoints, whereas the existing project
aliases refer to non-deduplicated checkpoints.

## Cluster preparation

The project convention is a manual `tmux` session on
`mark.inf.ethz.ch`. Preparation may use the network; the actual
experiment is forced offline after an exact cache preflight.

After pulling this commit and activating the project's Conda environment:

```bash
cd /home/durnovv/projects/probability-of-a-word-experiments
source ~/miniforge3/etc/profile.d/conda.sh
conda activate probability-of-a-word
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME=/pub/hofmann-scratch/huggingface_cache
export TUNED_LENS_ROOT=$RUN_ROOT/resources/tuned-lens
export TUNED_LENS_PYTHONPATH=$RUN_ROOT/python/tuned-lens-0.2.0
mkdir -p "$TUNED_LENS_PYTHONPATH"
python -m pip install --target "$TUNED_LENS_PYTHONPATH" --no-deps "tuned-lens==0.2.0"
python scripts/stage_layer_factorial_resources.py --all --lens-root "$TUNED_LENS_ROOT" --hf-home "$HF_HOME"
```

The active environment must provide `wordsprobability==0.17`. The
preflight records the installed GPU stack and scientific Python versions.
Verify every staged resource without network access:

```bash
export PYTHONPATH=$TUNED_LENS_PYTHONPATH:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python scripts/stage_layer_factorial_resources.py --all --verify-only --lens-root "$TUNED_LENS_ROOT" --hf-home "$HF_HOME"
```

The staged lenses come from immutable `AlignmentResearch/tuned-lens`
Space revision `1ac7285852a22309f571c2555efc37375d0c4cda`. Complete
base revisions and artifact hashes live in
`src/h01_data/layer_factorial_models.py`.

## Cluster run

Start a persistent shell, select exactly one GPU, and dry-run the full
three-model sequence:

```bash
tmux new -s layer-factorial
cd /home/durnovv/projects/probability-of-a-word-experiments
source ~/miniforge3/etc/profile.d/conda.sh
conda activate probability-of-a-word
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME=/pub/hofmann-scratch/huggingface_cache
export TUNED_LENS_ROOT=$RUN_ROOT/resources/tuned-lens
export TUNED_LENS_PYTHONPATH=$RUN_ROOT/python/tuned-lens-0.2.0
export CUDA_VISIBLE_DEVICES=0
export EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)
scripts/run_all_layer_factorial_cluster.sh --dry-run
scripts/run_all_layer_factorial_cluster.sh
```

To run or resume only one model:

```bash
scripts/run_layer_factorial_cluster.sh gpt2-xl
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

Both `time` and `paper_time` are evaluated for all eight
context/score/lens cells. The output validator must pass before any regression
is fitted.
