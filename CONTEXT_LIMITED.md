# Context-limited LM reading-time predictors

CONTEXT-LIMITED: This opt-in pipeline adds transformer surprisals computed from
fixed numbers of preceding words without changing the original or n-gram
pipeline outputs.

## Run one experiment

CONTEXT-LIMITED: Run all default stages—fixed-window scoring, keyed merge,
spillover creation, and reading-time analysis—with:

```bash
make -f MakefileContextLimited \
  DATASET=natural_stories \
  MODEL=pythia-70m \
  CONTEXT_LIMITED_CONTEXT_LENGTHS="1 2 4"
```

CONTEXT-LIMITED: `CONTEXT_LIMITED_CONTEXT_LENGTHS` directly counts preceding
whitespace-delimited project words. For example, context `4` retains at most
the four words immediately before the target. Values must be positive. The
reference paper's settings labelled as n-gram orders `2, 3, 5, 7, 10, 20`
correspond here to preceding-word lengths `1, 2, 4, 6, 9, 19`.

CONTEXT-LIMITED: `CONTEXT_LIMITED_BATCH_SIZE` defaults to `8`. Reduce it if a
model runs out of accelerator memory. Batch size affects runtime and memory,
not scores or checkpoint names.

CONTEXT-LIMITED: Run only data production with `process_context_limited_data`,
or the complete analysis and its prerequisites with `get_context_limited_llh`.
To process all English dataset/model combinations from the original project:

```bash
bash scripts/run_all_context_limited.sh
```

## Scoring semantics

CONTEXT-LIMITED: Each input-text line is one passage. Context is continuous
across sentence punctuation, never crosses a line/passage boundary, and never
contains future words. Every target is scored independently for every requested
maximum context length.

CONTEXT-LIMITED: BOS is supplied only when a target has no preceding retained
context (the first word of a passage). A nonempty truncated window starts with
its oldest retained word, including its ordinary leading-space tokenization,
but without a misleading BOS token.

CONTEXT-LIMITED: Word surprisal is in natural-log units and uses the same
corrected word-probability equation as `wordsprobability==0.17`:

```text
sum(target-subtoken NLLs) - start-boundary NLL + end-of-word NLL
```

CONTEXT-LIMITED: The implementation reuses that package's GPT-2/Pythia model
mapping and vocabulary masks. It checks compatibility at runtime because those
model-wrapper details are package internals.

## Checkpoints

CONTEXT-LIMITED: Files are written under separate directories:

- `checkpoints/rt/context_limited_surprisals_rt_data/`
- `checkpoints/rt/merged_context_limited_data/`
- `checkpoints/rt/context_limited_delta_llh/`
- `checkpoints/rt/context_limited_params/`

CONTEXT-LIMITED: Predictor TSVs contain zero-based `text_id`, `word_id`, the raw
project `word`, and one column per configured window, such as
`context_limited_surprisal_context_4`. Files are written atomically, merged by
stable IDs with complete-coverage and word-alignment checks, and expanded to
three text-bounded spillover positions before modeling.

## Runtime expectations

CONTEXT-LIMITED: This is substantially more expensive than ordinary full-text
scoring because each word/window pair needs an independent forward pass. The
model is loaded only once per command, similarly sized windows are batched, KV
caching is disabled, and padding is length-sorted to reduce memory overhead.
