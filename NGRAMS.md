# N-gram reading-time predictors

This opt-in pipeline adds Infini-gram predictors without changing the original
`Makefile` outputs.

## Setup

Install the additional dependencies:

```bash
python -m pip install -r scripts/requirements-ngram.txt
```

The default backend is the public Infini-gram API and the default index is
`v4_piletrain_llama`. A local multi-terabyte index is not required.

N-GRAM: Counts are requested with Llama-2 token IDs so raw corpus text cannot be
misread as Infini-gram `AND`/`OR` query syntax or rejected by a text filter. The
small tokenizer files (not model weights) are downloaded once from the pinned
`NousResearch/Llama-2-7b-hf` revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d` and cached under
`.cache/huggingface/`. This transport therefore supports Infini-gram indexes
whose documented tokenizer is Llama-2 and whose name ends in `_llama`.

## Run one experiment

```bash
make -f MakefileNgrams \
  DATASET=natural_stories \
  MODEL=pythia-70m \
  NGRAM_CONTEXT_LENGTHS="0 1 2 3 4"
```

`NGRAM_CONTEXT_LENGTHS` counts preceding whitespace-delimited words:

- `0`: unigram
- `1`: bigram (one preceding word)
- `2`: trigram (two preceding words)
- and so on

Run only data generation with `process_ngram_data`, or only the final analysis
and its prerequisites with `get_ngram_llh`. To run every English dataset/model
combination from the original project:

```bash
bash scripts/run_all_ngrams.sh
```

## Scoring semantics

The implementation follows `ngram-reading-time`: it computes natural-log count
ratios and uses Stupid Backoff with alpha `0.4`. If the requested context is
unseen, it backs off to shorter contexts. Therefore, a configured context length
is a maximum and the resulting Stupid-Backoff score is not a normalized
probability distribution.

The context window is measured in project words, while Infini-gram counts the
corresponding Llama-token sequence. Context resets at every input-text line.
Every n-gram predictor is entered into the reading-time regression with its own
three preceding-word spillover values, alongside the existing length/frequency
controls.

## Checkpoints and resuming

The pipeline writes separate files under:

- `checkpoints/rt/ngram_surprisals_rt_data/`
- `checkpoints/rt/merged_ngram_data/`
- `checkpoints/rt/ngram_delta_llh/`
- `checkpoints/rt/ngram_params/`

API counts are persisted in `.cache/ngram_count_cache/<index>.sqlite`, which is
already ignored by the repository. Transient HTTP 403, 408, 429, and server
errors receive finite exponential-backoff retries. If all retries fail, rerun
the same Make command; completed count batches are reused. The final TSV is
written atomically, so a partial file cannot be mistaken for a complete target.

Useful overrides include `NGRAM_INDEX`, `NGRAM_WORKERS`, `NGRAM_TIMEOUT`,
`NGRAM_MAX_RETRIES`, `NGRAM_API_URL`, and `NGRAM_BACKOFF_ALPHA`. If
`NGRAM_TOKENIZER` or `NGRAM_TOKENIZER_REVISION` is changed, it must remain an
exact match for the tokenizer used to build `NGRAM_INDEX`.
