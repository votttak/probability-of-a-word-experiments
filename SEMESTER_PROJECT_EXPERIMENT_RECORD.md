# Semester project experiment record

- Status: 2026-08-16
- Code baseline:
  `f3497cea515bad06cd6ee3b35e41d11e18409aa6` (`f3497ce`) on `main`
- Repository: `votttak/probability-of-a-word-experiments`

This document is the durable lab record for the Natural Stories experiment that
compares n-gram surprisal with context-limited transformer surprisal. It
collects the design decisions, execution history, validation checks, numerical
results, interpretation boundaries, failures, and recommended next steps that
were established during the experiment.

## 1. Executive summary

The primary question was whether a context-limited transformer contains
reading-time information beyond an n-gram that is allowed the same number of
preceding words, and conversely whether the n-gram adds information beyond the
transformer.

The full analysis used 10,023 complete Natural Stories word observations,
10 deterministic cross-validation folds, and the same lexical controls and
folds for every comparison. The clearest result is:

- Context-limited transformer surprisal adds consistent held-out predictive
  density beyond n-gram surprisal.
- At matched context lengths, the remaining contribution of contextual
  n-grams is very small and inconsistent.
- The exception is the context-0 n-gram, which is effectively an external
  corpus-unigram/frequency predictor. It remains strongly complementary to
  every transformer model.
- Every model's descriptively best joint condition contains that unigram
  predictor.
- Larger models have lower mean corpus surprisal, but they do not show greater
  human reading-time alignment in this experiment. Smaller models are often
  descriptively stronger. This size result is exploratory.

Across the corrected ten-model analysis, transformer surprisal made a positive
unique contribution in all 200 model-by-n-gram-by-transformer cells. All 200
means were also greater than 1.96 descriptive fold standard errors. In
contrast, the n-gram contribution was positive in 148/200 cells, greater than
1.96 fold standard errors in 77/200, and was driven mainly by context 0.

The largest selected joint gain was for Pythia 70M with n-gram context 0 and
transformer context 4: `0.049402` nats per observation over the lexical-control
model, or `5.064%` higher geometric-mean predictive density. This ranking was
selected using the same folds and must be described as exploratory.

Important metric warning: these percentages are changes in held-out predictive
density, computed as `100 * (exp(delta) - 1)`. They are not percent of reading
time explained, percent variance explained, or R-squared.

## 2. Research question and notation

Let:

- `M0` be the lexical-control reading-time model.
- `A` be one n-gram surprisal predictor.
- `B` be one context-limited transformer surprisal predictor.

The two paired quantities are:

```text
delta(A | B) = LL(M0 + A + B) - LL(M0 + B)
delta(B | A) = LL(M0 + A + B) - LL(M0 + A)
```

Here `LL` is the mean Gaussian log density assigned to held-out raw reading
times, in natural-log units per observation. A positive delta means that adding
the named predictor improved out-of-sample density.

The experiment tests unique predictive information, not whether either
predictor has a causal effect on reading or implements a particular cognitive
mechanism.

### Context-length convention

Every configured context length counts preceding whitespace-delimited project
words.

| Context value | Preceding words retained | Conventional n-gram name |
|---:|---:|---|
| 0 | 0 | unigram |
| 1 | 1 | bigram |
| 2 | 2 | trigram |
| 3 | 3 | 4-gram |
| 4 | 4 | 5-gram |

Thus a matched context-4 comparison is a conventional 5-gram versus a
transformer supplied with at most four preceding words. Calling it a "4-gram"
would be incorrect under the conventional order convention.

## 3. Data and predictor construction

### 3.1 Natural Stories data

- Corpus: Natural Stories.
- Passage count: 10.
- Full predictor text: 10,256 words.
- Passage lengths: approximately 939 to 1,099 words.
- Input structure: one passage per text-file line.
- Predictor IDs: zero-based `text_id` and `word_id`.
- Established RT/control table: one-based text IDs.
- The joint merger adds one to predictor `text_id` for this Natural
  Stories-specific alignment. `MakefileJointFull` intentionally rejects other
  datasets until their ID convention is verified.

RT preprocessing removes duplicate source rows, marks observations with global
`abs(zscore(log(RT))) > 3` as outliers, removes them, and averages numeric
columns by text, word position, and reference token. The response used below is
raw mean reading time; `centered_time` is not used by the evaluator.

The predictor files align from `(text_id=0, word_id=0, word=If)` through
`(text_id=9, word_id=938, word=Tourette's.)`.

The frequency control is not a raw count. The preprocessing strips surrounding
punctuation, lowercases, Moses-tokenizes/detokenizes, and sums negated values
from `corpora/rt/unigrams.csv`. An out-of-vocabulary word produces a missing
value. Word length is Python character length.

The two upstream Natural Stories inputs are `processed_RTs.tsv` and
`all_stories_gpt3.csv` from `languageMIT/naturalstories`. Text reconstruction
uses the story/token offsets in the latter and writes one reconstructed story
per line. The Make dependency currently tracks `processed_RTs.tsv` but does not
explicitly track `all_stories_gpt3.csv`; both raw inputs should be archived and
checksummed.

### 3.2 N-gram predictor A

Implementation:
[`src/h01_data/get_ngram_surprisals.py`](src/h01_data/get_ngram_surprisals.py)

Final configuration:

| Setting | Value |
|---|---|
| Infini-gram index | `v4_piletrain_llama` |
| Contexts | `0 1 2 3 4` preceding words |
| Backoff | Stupid Backoff |
| Alpha | `0.4` |
| API | `https://api.infini-gram.io/` |
| Workers | 4 |
| Timeout | 30 seconds |
| Maximum retries | 8 |
| Tokenizer | `NousResearch/Llama-2-7b-hf` |
| Tokenizer revision | `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` |

Counts are queried with Llama-2 token IDs rather than raw corpus text, without
BOS/EOS. For each configured maximum word context, the score uses natural-log
count ratios and backs off to shorter contexts when necessary. Failed higher
orders incur the `alpha=0.4` penalty; unavailable passage-start history does
not. Context resets at each passage.

For the longest suffix with a positive count, the score has the form:

```text
surprisal = -log(alpha^b * count(context,target) / count(context))
```

Here `b` is the number of failed higher orders. If every multiword order is
unseen, the scorer uses the backed-off unigram count, clamps a zero target count
to one, and divides by the index token total.

Stupid-Backoff scores are not a normalized probability distribution. The term
"n-gram surprisal" is retained to match the project terminology, but this
normalization caveat must be stated in the report.

The full run required 41,475 distinct Infini-gram count queries. It took
6:14:03 at approximately 1.85 queries/second. Counts are resumably cached in:

```text
$RUN_ROOT/cache/ngram_count_cache/v4_piletrain_llama.sqlite
```

The final TSV is written atomically. A complete full-corpus file has 10,257
lines including its header. The shared cluster `ngram.tsv` SHA-256 is:

```text
17633df556afc67e43ede824ed847dfbf7d70a10e65c22441d75ae646601515d
```

### 3.3 Context-limited transformer predictor B

Implementation:
[`src/h01_data/get_context_limited_surprisals.py`](src/h01_data/get_context_limited_surprisals.py)

Final contexts: `1 2 3 4` preceding words.

Each target is scored independently for every requested context. The context:

- continues across punctuation;
- never crosses a passage boundary;
- never includes future words;
- uses BOS only when no preceding context exists;
- starts a nonempty truncated window with ordinary leading-space
  tokenization and no artificial BOS.

Word surprisal is in natural-log units and uses the
`wordsprobability==0.17` boundary correction:

```text
sum(target-subtoken NLLs) - start-boundary NLL + end-of-word NLL
```

The scorer uses the package's GPT-2/Pythia model mapping and vocabulary masks,
inference mode, `use_cache=False`, and length-sorted batches. Logits and
boundary corrections are evaluated in float32 even when model weights are
FP16. It writes the final TSV atomically, but it has no within-model progress
checkpoint. A failed model starts again from the first word; an already
complete Make target is skipped.

### 3.4 Model coverage

Completed context-limited predictor files:

- GPT-2 small
- GPT-2 medium
- GPT-2 large
- GPT-2 XL
- Pythia 70M
- Pythia 160M
- Pythia 410M
- Pythia 1.4B
- Pythia 2.8B
- Pythia 6.9B

Repository aliases that omit the decimal point are:

| Alias | Actual model |
|---|---|
| `pythia-14b` | Pythia 1.4B |
| `pythia-28b` | Pythia 2.8B |
| `pythia-69b` | Pythia 6.9B |
| `pythia-120b` | Pythia 12B |

Pythia 12B was not run. Its FP16 weights require roughly 23.7 GB before
activation overhead, while every GPU on `mark` has 16 GB. The
`wordsprobability` loader places the whole model on the first visible GPU and
does not shard across GPUs. Reducing batch size or exposing multiple GPUs
cannot solve a model-weight load failure.

### 3.5 Merge and spillover validation

Implementation:
[`src/h01_data/build_joint_pilot_dataset.py`](src/h01_data/build_joint_pilot_dataset.py)

The merger requires unique predictor keys, identical full coverage, exact word
agreement across A and B, finite numeric predictors, and contiguous word IDs
within every passage. It increments predictor `text_id` by one for Natural
Stories, requires one-to-one coverage and exact token agreement with the RT
base, sorts by key, and creates current plus three preceding predictor values
within passage boundaries. The final output is atomic.

## 4. Reading-time model and evaluation

Implementation:
[`src/h02_rt_model/rt_vs_joint_pilot_surprisal.R`](src/h02_rt_model/rt_vs_joint_pilot_surprisal.R)

### 4.1 Response and controls

The response is aggregated raw reading time in milliseconds.

The lexical-control formula is:

```text
word_len * freq
+ prev_word_len * prev_freq
+ prev2_word_len * prev2_freq
+ prev3_word_len * prev3_freq
```

In R, each `*` includes both main effects and their interaction. The model
therefore controls word length, frequency, and their interaction at the current
word and three preceding positions: 13 coefficients including the intercept.

Every n-gram or transformer predictor is also entered at the current word and
three preceding spillover positions. Each grid cell fits one n-gram context
and one transformer context; it does not enter every context simultaneously.
Single-predictor models add four coefficients, while the joint model has 21
coefficients in total.

### 4.2 Analysis sample

| Run | Input rows | Complete rows | Excluded |
|---|---:|---:|---:|
| Pilot | 500 | 454 | 46 |
| Full analysis | 10,256 | 10,023 | 233 |

The full complete-case rate is 97.728%. All 233 excluded rows have a missing
frequency control at the current word or one of the three lag positions.
Missing counts are 52 current, 61 at lag 1, 71 at lag 2, and 80 at lag 3, with
overlap across these sets. This includes the first three words of each of the
10 passages, where lags are undefined. The same complete-case mask is used for
every full comparison.

### 4.3 Cross-validation and score

- Folds: 10.
- Seed: 42.
- Split: deterministic random word-level folds.
- Models: ordinary least squares on the training rows.
- Residual variance: training `mean(residuals^2)`.
- Test score: Gaussian log density of held-out raw RT using the training
  prediction and residual variance.
- Fold result: mean log density over that fold's test observations.
- Published mean: unweighted mean of the 10 fold deltas.
- Published SE: `sd(10 fold deltas) / sqrt(10)`.

R first applies `set.seed(42)`, randomly permutes the shared complete rows, and
then divides the permuted sequence into ten folds. The same sorted rows and
fold assignment are used for all cells and all ten models. This makes paired
differences directly comparable.

The reported fold SE describes across-fold variability. It is not a
coefficient SE, bootstrap SE, formal repeated-CV uncertainty estimate, or a
multiple-comparison-corrected confidence interval.

### 4.4 Converting log-density deltas to percentages

For a mean delta `d`:

```text
predictive-density change (%) = 100 * (exp(d) - 1)
```

Example: `d = 0.04` corresponds to about `4.08%` higher geometric-mean density
assigned to observed held-out RTs.

Correct wording:

> The predictor increased held-out predictive density by X% relative to the
> lexical-control model.

Incorrect wording:

> The predictor explained X% of reading time.

R-squared, percent variance explained, and held-out MSE reduction require
additional saved predictions/residuals and are not available from the present
summary tables.

## 5. Experiment lineage

| Date | Commit | Record |
|---|---|---|
| 2026-08-15 16:42 | `0ca4fe9` | Added the complete n-gram/context-limited pipeline, Makefiles, tests, and documentation. |
| 2026-08-16 12:24 | `b168fff` | Added first full GPT-2-small results with transformer contexts 1, 2, and 4. |
| 2026-08-16 12:59 | `67d97ab` | Corrected the default to include transformer context 3. |
| 2026-08-16 13:18 | `9188997` | Added the definitive GPT-2-small contexts 1, 2, 3, and 4 results. |
| 2026-08-16 15:12 | `f3497ce` | Added the context-only all-model runner and runtime dtype/device logging. |

### 5.1 Joint pilot

The pilot used the first 50 words of each of the 10 passages: 500 input rows
and 454 complete cases. It used transformer contexts 1, 2, and 4.

Matched-context results:

| Context | `delta n-gram given transformer` | `delta transformer given n-gram` |
|---:|---:|---:|
| 1 | -0.005758 +/- 0.007541 | +0.026408 +/- 0.017312 |
| 2 | -0.012396 +/- 0.008686 | +0.007766 +/- 0.024580 |
| 4 | -0.000590 +/- 0.006340 | +0.045903 +/- 0.020155 |

Across its 15 grid cells, the transformer mean was positive in 15/15 but
greater than 1.96 fold SE in only 5/15. The n-gram mean was positive in only
2/15. The pilot was useful for pipeline validation and directional evidence,
but it was too noisy for primary inference and is superseded by the full run.

### 5.2 First full GPT-2-small run

The first full result included transformer contexts `1,2,4` but accidentally
omitted context 3. That result remains in the repository for provenance but is
structurally superseded.

The correction did not materially alter shared cells: between the original and
expanded runs, the maximum absolute difference was approximately
`2.42e-9` across the B means and smaller or comparable for the other pivot
statistics. Therefore the rerun added context 3; it did not reverse an earlier
result.

### 5.3 Definitive full GPT-2-small run

Canonical tracked result:
[`summary.tsv`](results/rt/joint_full/natural_stories-gpt2-small-ngram_v4_piletrain_llama-contexts_0-1-2-3-4-alpha_0_4-context_1-2-3-4-folds_10-seed_42/summary.tsv).
Its parent directory contains the fold table and all four mean/SE pivots.

This is the definitive single-model analysis: 10,256 input rows, 10,023
complete rows, four transformer contexts, five n-gram contexts, 20 grid cells,
and 10 folds per cell.

### 5.4 Ten-model extension

The ten complete context-limited predictor TSVs were copied from cluster
scratch into local `checkpoints/rt/joint_full/`. All-model RT evaluation was
then reproduced locally with a cluster-derived base table and the common
n-gram file.

The ten-model numbers below are a validated local re-analysis, but the
consolidated fold results are not yet a tracked canonical repository artifact.
That distinction must be resolved before final submission.

## 6. Data-integrity and reproduction audit

### 6.1 Predictor validation

For each of the ten context-limited models:

- 10,256 data rows plus one header;
- exact schema containing IDs, word, and contexts 1 to 4;
- zero duplicate `(text_id, word_id)` keys;
- zero ID or word mismatches across models;
- exact key/word alignment with the common n-gram TSV;
- all 410,240 context-limited scores finite and strictly positive;
- global observed range `0.001682` to `46.620449`.

In-memory joint merging succeeded for every model. Every resulting data set had
10,256 rows, five n-gram predictors, four transformer predictors, 10,023
complete cases, and the same 233 exclusions.

### 6.2 Critical local-versus-cluster base-table warning

An important provenance problem was found during the all-model analysis:

- The pre-existing local
  `checkpoints/rt/merged_data/natural_stories-gpt2-small.tsv` did not contain
  the same `time` values as the completed cluster experiment.
- All 10,256 `time` entries differed; the maximum discrepancy was about
  200 ms and the mean absolute discrepancy was about 15.68 ms.
- The cause has not yet been established. It may reflect a different cached
  RT preprocessing state or source revision, but this is not proven.

Using that local base produces the wrong reproduction. The corrected
ten-model analysis instead extracted the RT and control columns from the
cluster-generated GPT-2-small `joint-data.tsv`, combined them with the common
cluster n-gram predictor and each model's context-limited file, and held the
rows/folds fixed.

Validation of the correction:

- Corrected GPT-2-small aggregate means match the saved cluster result within
  `1.01e-15`.
- Corrected GPT-2-small SEs match within `2.11e-16`.
- Fold-result rows match within `1.01e-11`.

The verified reconstructed base temporarily resides at:

```text
/tmp/joint-allmodels-clusterbase.RKERrM/cluster-base.tsv
```

Its SHA-256 is:

```text
d56c8dbc527d2c05947d57cc13cc2bdb3124808d9108f1b064eba76a737747b3
```

The validated explicit-model aggregate temporarily resides at:

```text
/tmp/joint-allmodels-clusterbase.RKERrM/all-model-conditional-deltas.tsv
```

Its SHA-256 is:

```text
d3e94161c6e8b4fa5cb2dbbdadae24098652613f54154acada6bcb8044066662
```

Do not use the discrepant local merged base for paper results. Before `/tmp` is
cleared, archive the corrected base and aggregate outside `/tmp`. Before final
submission, persist one checksummed canonical base, explain or eliminate the
discrepancy, and regenerate tracked all-model outputs from that base.

## 7. Quantitative results

### 7.1 GPT-2-small: each predictor alone versus controls

These are held-out gains over `M0`.

N-gram only:

| N-gram context | Mean delta +/- fold SE | Predictive-density gain |
|---:|---:|---:|
| 0 | 0.039651 +/- 0.003076 | 4.045% |
| 1 | 0.027258 +/- 0.002585 | 2.763% |
| 2 | 0.017066 +/- 0.001814 | 1.721% |
| 3 | 0.007725 +/- 0.001330 | 0.776% |
| 4 | 0.004770 +/- 0.000879 | 0.478% |

Context-limited GPT-2-small only:

| Transformer context | Mean delta +/- fold SE | Predictive-density gain |
|---:|---:|---:|
| 1 | 0.036147 +/- 0.003445 | 3.681% |
| 2 | 0.032256 +/- 0.002677 | 3.278% |
| 3 | 0.031069 +/- 0.002351 | 3.156% |
| 4 | 0.030027 +/- 0.002720 | 3.048% |

Every value in both tables was positive in all 10 folds.

The decline across n-gram contexts does not mean that shorter contexts are
intrinsically better language models. It means that, in this RT regression and
with these controls, the selected surprisal predictor gave a larger held-out
density gain.

### 7.2 GPT-2-small: matched unique contributions

| Preceding-word context | N-gram given transformer | Ratio to fold SE; positive folds | Transformer given n-gram | Ratio to fold SE; positive folds |
|---:|---:|---:|---:|---:|
| 1 | +0.000302 +/- 0.000449 | 0.67; 7/10 | +0.009191 +/- 0.001505 | 6.11; 10/10 |
| 2 | -0.000418 +/- 0.000220 | -1.90; 4/10 | +0.014772 +/- 0.002467 | 5.99; 10/10 |
| 3 | +0.000768 +/- 0.000469 | 1.64; 6/10 | +0.024112 +/- 0.002116 | 11.40; 10/10 |
| 4 | +0.001127 +/- 0.000504 | 2.24; 7/10 | +0.026384 +/- 0.002676 | 9.86; 10/10 |

At every matched context, GPT-2-small carries substantially more unique
held-out RT information than the contextual n-gram. The 5-gram/context-4
n-gram has a small positive residual, while the trigram/context-2 residual is
slightly negative.

The strongest GPT-2-small joint cell is n-gram context 0 plus transformer
context 4:

```text
total delta over controls = 0.047214 nats/observation
predictive-density gain   = 4.835%
```

Within that joint cell:

```text
delta n-gram(0) given transformer(4) = 0.017187
delta transformer(4) given n-gram(0) = 0.007563
```

This supports a complementary lexical/unigram interpretation, not a
higher-order contextual n-gram advantage.

### 7.3 Corrected ten-model matched-context summary

Across the ten models:

| Matched context | Mean n-gram given transformer [range] | Models above 1.96 fold SE | Mean transformer given n-gram [range] | Models above 1.96 fold SE |
|---:|---:|---:|---:|---:|
| 1 | +0.000239 [-0.000418, +0.002343] | 2/10 | +0.005262 [+0.002969, +0.009191] | 10/10 |
| 2 | -0.000393 [-0.000506, -0.000289] | 0/10; 5/10 below -1.96 | +0.012386 [+0.009318, +0.017662] | 10/10 |
| 3 | +0.000468 [+0.000179, +0.000768] | 0/10 | +0.020641 [+0.016251, +0.027390] | 10/10 |
| 4 | +0.000822 [+0.000297, +0.001127] | 5/10 | +0.022563 [+0.015877, +0.033037] | 10/10 |

The transformer-only gain over lexical controls ranges from `0.020351` to
`0.036702` nats/observation, equivalent to `2.056%` to `3.738%` higher
predictive density.

Across the full 200-cell grid:

- `delta transformer given n-gram` is positive in 200/200 cells.
- It is greater than 1.96 fold SE in 200/200 cells.
- Its range is `0.001611` to `0.033037`.
- `delta n-gram given transformer` is positive in 148/200 cells.
- It is greater than 1.96 fold SE in 77/200 and below -1.96 in 9/200.
- Its range is `-0.000506` to `0.023792`.
- All 40 cells containing n-gram context 0 are above 1.96 fold SE.

These counts are descriptive summaries over a large exploratory grid, not 200
independent, multiplicity-corrected hypothesis tests.

### 7.4 Best selected joint cell by model

`B-only` reports the best transformer-only context and its density gain over
controls. The remaining columns report the best joint cell selected from the
20 combinations for that model.

| Model | Best B-only (context; density gain) | Best joint A,B | Total delta vs controls | Total density gain | `delta A given B` | `delta B given A` |
|---|---|---:|---:|---:|---:|---:|
| Pythia 70M | C4; 3.738% | 0,4 | 0.049402 | 5.064% | 0.012700 | 0.009750 |
| GPT-2 small | C1; 3.681% | 0,4 | 0.047214 | 4.835% | 0.017187 | 0.007563 |
| Pythia 160M | C2; 3.428% | 0,4 | 0.047141 | 4.827% | 0.016570 | 0.007490 |
| GPT-2 medium | C2; 2.999% | 0,4 | 0.046588 | 4.769% | 0.019041 | 0.006937 |
| Pythia 410M | C1; 3.382% | 0,4 | 0.046154 | 4.724% | 0.020168 | 0.006503 |
| GPT-2 large | C1; 3.294% | 0,4 | 0.045436 | 4.648% | 0.020456 | 0.005785 |
| GPT-2 XL | C1; 3.295% | 0,4 | 0.045415 | 4.646% | 0.020979 | 0.005764 |
| Pythia 1.4B | C1; 3.489% | 0,3 | 0.045160 | 4.620% | 0.019610 | 0.005509 |
| Pythia 6.9B | C1; 3.222% | 0,2 | 0.045023 | 4.605% | 0.018279 | 0.005371 |
| Pythia 2.8B | C1; 3.241% | 0,2 | 0.044949 | 4.597% | 0.017599 | 0.005298 |

Every selected best joint cell uses n-gram context 0. The best transformer
context is not sharply identified: some runner-up differences are extremely
small. For example, the Pythia 1.4B best-versus-runner-up joint gap is only
about `2.54e-5` nats per observation.

Pythia 70M is numerically best in this selected grid. Its advantage over the
other selected model scores is only about `0.219%` to `0.446%` in predictive
density. Because models and cells were ranked on the same folds, this is not a
confirmatory model-selection result.

### 7.5 Descriptive properties of the predictor values

Mean word surprisal by model and retained transformer context:

| Model | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| GPT-2 small | 7.208745 | 6.017979 | 5.614213 | 5.350185 |
| GPT-2 medium | 6.865610 | 5.878597 | 5.437200 | 5.156729 |
| GPT-2 large | 6.623083 | 5.812762 | 5.375310 | 5.084912 |
| GPT-2 XL | 6.605409 | 5.789720 | 5.334566 | 5.045455 |
| Pythia 70M | 7.230104 | 6.449293 | 6.047631 | 5.802079 |
| Pythia 160M | 6.974405 | 6.139736 | 5.706008 | 5.442879 |
| Pythia 410M | 6.797835 | 5.910673 | 5.452783 | 5.154528 |
| Pythia 1.4B | 6.678502 | 5.780362 | 5.303092 | 5.002808 |
| Pythia 2.8B | 6.606913 | 5.743610 | 5.254303 | 4.944549 |
| Pythia 6.9B | 6.597241 | 5.694886 | 5.200186 | 4.892537 |

For every model, mean surprisal falls monotonically from C1 to C4. The C1-to-C4
mean reduction ranges from 19.751% to 25.840%, with a ten-model mean of
23.949%. This is an average: individual word surprisal can rise when context is
added.

Within a model:

- adjacent-context Pearson correlations range from `0.899` to `0.973`;
- C1-versus-C4 correlations range from `0.819` to `0.884`.

Across models at a fixed context, median pairwise correlations range from
`0.964` to `0.972`. The predictors therefore overlap strongly but are not
identical.

Mean surprisal decreases monotonically with nominal parameter count within
both GPT-2 and Pythia at every context. Spearman rank correlation is `-1` for
each within-family series. However, at matched contexts 2 to 4, the unique RT
contribution of transformer surprisal decreases descriptively with size:

- GPT-2 Pearson correlation with log parameter count: approximately
  `-0.982`, `-0.959`, and `-0.982` for C2, C3, and C4.
- Pythia: approximately `-0.953`, `-0.958`, and `-0.964`.

This contrast is scientifically interesting: lower average corpus surprisal
does not imply greater human-RT prediction. It remains exploratory because
there are few sizes, model size is confounded with training/model differences,
and the grid was not preregistered.

## 8. Interpretation for the semester paper

### 8.1 Primary supported claim

When lexical controls, spillover structure, analysis rows, and
cross-validation folds are held fixed, context-limited transformer surprisal
provides robust additional held-out RT predictive density beyond n-gram
surprisal. This holds across all ten completed transformer models and all
tested n-gram/transformer context combinations.

### 8.2 Secondary supported claim

Matched higher-order n-grams contribute little additional contextual signal
once transformer surprisal is known. Context-2 n-grams are slightly negative
on average, and contexts 1, 3, and 4 are near zero to modestly positive.

### 8.3 Complementary unigram claim

The strongest n-gram contribution comes from context 0. It adds information
even though the baseline already contains a frequency control. This should be
interpreted as a complementary lexical-frequency signal from a different
corpus/resource, not as evidence that a no-context model captures additional
sequential context.

### 8.4 Exploratory scaling claim

Smaller models, especially Pythia 70M, often have larger RT-density gains even
though larger models have lower mean surprisal. This may be consistent with a
mismatch between next-word optimization and human processing, but the present
experiment does not establish a mechanism.

### 8.5 Claims that are not supported

Do not claim:

- that X% of RT or RT variance was explained;
- a causal effect of context or model size;
- that Pythia 70M is universally the best psycholinguistic model;
- formal significance from `mean / fold-SE`;
- that the selected context is a precisely estimated cognitive window;
- participant-level generalization;
- story-level generalization;
- that Stupid-Backoff scores are normalized probabilities;
- that larger LMs are generally worse language models.

## 9. Threats to validity

1. **Word-level rather than grouped cross-validation.** Adjacent words and
   words from the same passage can appear in both train and test sets. This can
   leak local/story structure and does not test generalization to a new story.

2. **Overlapping training folds.** The 10 fold estimates share most training
   observations. Their standard error is descriptive rather than a calibrated
   repeated-CV inferential uncertainty estimate.

3. **Aggregated RT.** The response is one aggregated mean per word. The
   analysis has no participant or item random effects and does not use
   `nItem` weighting.

4. **Linear Gaussian model on raw RT.** The evaluator uses OLS and Gaussian
   density on raw milliseconds. It does not model log RT or the distributional
   and hierarchical structure commonly used in reading-time analyses.

5. **Complete-case filtering.** Missing frequency controls and spillover
   boundaries remove 233 observations. The mask is shared across models, but
   exclusion may still bias the analyzed vocabulary/positions.

6. **Multiplicity and selection.** There are 20 cells per model and ten
   models. Best-cell and best-model rankings are selected on the evaluation
   folds themselves.

7. **N-gram interpretation.** Context 0 is a frequency-like signal, and Stupid
   Backoff is not normalized. Its strong performance should not be described
   as higher-order contextual prediction.

8. **Family confounds.** GPT-2 and Pythia differ in training data, tokenizer,
   architecture details, and scale. Cross-family size trends cannot be
   attributed solely to parameter count.

9. **Unpinned revisions and numeric precision.** Output paths identify aliases
   and contexts but not Hugging Face revision, Torch/Transformers version, or
   dtype. Runtime logs partly mitigate this. The n-gram SQLite key also omits
   tokenizer revision and API URL.

10. **Base-table provenance discrepancy.** The local and cluster RT base
    mismatch must be resolved before final archival analysis.

## 10. Cluster and environment record

### 10.1 Hosts and hardware

- Cluster user: `durnovv`.
- Initial host: `donald.inf.ethz.ch`, 8 x RTX 2080 Ti, 11,264 MiB each.
- Work moved because all `donald` GPUs became occupied by `jcoquet`.
- Final host: `mark.inf.ethz.ch`, 8 x Tesla P100 PCIe, 16,384 MiB each.
- Driver on `mark`: `575.57.08`.
- Driver-reported CUDA: `12.9`.
- A warning that group ID `527522` had no name was harmless.

Cake and `nvidia-smi` were used to inspect occupancy. One free GPU was selected
explicitly with `CUDA_VISIBLE_DEVICES`.

### 10.2 Software

Verified cluster environment:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| R | 4.5.3 |
| Torch | 2.12.1+cu126 |
| Torch CUDA runtime | 12.6 |
| Transformers | 5.12.1 |
| `wordsprobability` | 0.17 |

The GPU test reported CUDA available and a Tesla P100 device. The Python test
suite passed 39 tests, and `Rscript tests/test_joint_pilot_r.R` passed.

Miniforge was installed under `~/miniforge3`. The original environment creation
failed because `wikitokenizer` is no longer available from PyPI. The successful
recovery omitted the unused `wikitokenizer`, created/activated the rest of the
environment, then installed R/rpy2 and remaining Python/ngram requirements.
The environment manifest should be fixed and locked before final
reproducibility release.

### 10.3 Storage and logs

```bash
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME=/pub/hofmann-scratch/huggingface_cache
```

Important locations:

```text
$RUN_ROOT/cache/
$RUN_ROOT/checkpoints/rt/
$RUN_ROOT/results/rt/
$RUN_ROOT/joint-full.log
$RUN_ROOT/joint-full-context-1-2-3-4.log
$RUN_ROOT/logs/context-limited-all-models/
```

Long work ran in tmux session `ngram-vs-context-limited`. Attach with
`tmux attach -t ngram-vs-context-limited`; detach with Ctrl-b, then `d`.

### 10.4 Runtime failures and resolutions

1. **GPU full-context scorer failure**

   The original `wordsprobability` stage failed while converting a CUDA tensor
   to NumPy:

   ```text
   TypeError: can't convert cuda:0 device type tensor to numpy
   ```

   The base preprocessing/full-context stage was run with the GPU hidden:

   ```bash
   CUDA_VISIBLE_DEVICES= make -f MakefileJointFull process_data \
     CACHE_DIR="$RUN_ROOT/cache" \
     CHECKPOINT_DIR="$RUN_ROOT/checkpoints/rt" \
     RESULTS_DIR="$RUN_ROOT/results/rt"
   ```

   Context-limited scoring later used the GPU normally.

2. **Long silent context-limited jobs**

   The scorer reports model load and final completion but no word-level
   progress. A one-hour silence after successful model loading was normal,
   particularly for Pythia 2.8B and larger. `nvidia-smi` showing the Python
   process using memory/utilization was the diagnostic. Arrow-key escape text
   such as `^[[B` in the terminal/log was harmless.

3. **Model memory**

   Batch-size reduction reduces activation memory, not model-weight memory.
   Pythia 6.9B completed in FP16 with batch 1 on a free 16 GB P100. Pythia 12B
   cannot fit under the current single-GPU loader.

4. **Git and scratch**

   Scratch output under `/pub/...` is outside the Git checkout. `git fetch`
   retrieves commits, not uncommitted scratch files. Use `scp` for scratch
   outputs. An early result commit made on `mark` was fetched into the laptop
   repository and fast-forward merged before pushing from the laptop.

## 11. Canonical commands

Run from `~/probability-of-a-word-experiments` after activating the environment.

### 11.1 Activate and validate

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate probability-of-a-word

python --version
Rscript --version
CUDA_VISIBLE_DEVICES=0 python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

python -m unittest discover -s tests -p 'test_*.py'
Rscript tests/test_joint_pilot_r.R
```

### 11.2 Full joint run

After the CPU base stage described above:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 make -f MakefileJointFull \
  CACHE_DIR="$RUN_ROOT/cache" \
  CHECKPOINT_DIR="$RUN_ROOT/checkpoints/rt" \
  RESULTS_DIR="$RUN_ROOT/results/rt" \
  2>&1 | tee -a "$RUN_ROOT/joint-full.log"
```

The n-gram stage is CPU/API-bound; the context-limited stage uses the selected
GPU. Reusing the exact SQLite cache avoids repeating the six-hour API query
pass.

### 11.3 Context-only multi-model sweep

Runner:
[`scripts/run_all_joint_full_context_surprisals.sh`](scripts/run_all_joint_full_context_surprisals.sh)

Example for the ten feasible models:

```bash
export RUN_ROOT=/pub/hofmann-scratch/students/durnovv/probability-of-a-word
export HF_HOME=/pub/hofmann-scratch/huggingface_cache

CUDA_VISIBLE_DEVICES=0 MODELS="gpt2-small gpt2-medium gpt2-large gpt2-xl \
pythia-70m pythia-160m pythia-410m pythia-14b pythia-28b pythia-69b" \
bash scripts/run_all_joint_full_context_surprisals.sh
```

Default batch sizes:

| Batch | Models |
|---:|---|
| 8 | GPT-2 small/medium; Pythia 70M/160M/410M |
| 4 | GPT-2 large; Pythia 1.4B |
| 2 | GPT-2 XL |
| 1 | Pythia 2.8B/6.9B/12B |

Completed target files are skipped on rerun. The runner refuses accidental CPU
fallback unless it is explicitly allowed.

### 11.4 Verify row counts and results

```bash
find "$RUN_ROOT/checkpoints/rt/joint_full" \
  -path '*context_1-2-3-4/context-limited.tsv' \
  -exec wc -l {} \;
```

Every complete predictor should report 10,257 lines.

For a completed joint analysis:

```bash
cat "$RESULT_DIR/summary.tsv"
cat "$RESULT_DIR/delta-a-given-b-mean.tsv"
cat "$RESULT_DIR/delta-b-given-a-mean.tsv"
cat "$RESULT_DIR/delta-a-given-b-se.tsv"
cat "$RESULT_DIR/delta-b-given-a-se.tsv"
```

Expected full summary:

```text
input_rows               10256
complete_case_rows       10023
excluded_rows            233
folds                    10
seed                     42
context_limited_contexts 1,2,3,4
```

### 11.5 Copy scratch artifacts to the laptop

From the laptop repository:

```bash
scp -r \
  'durnovv@mark.inf.ethz.ch:/pub/hofmann-scratch/students/durnovv/probability-of-a-word/checkpoints/rt/joint_full/*context_1-2-3-4' \
  checkpoints/rt/joint_full/

scp -r \
  'durnovv@mark.inf.ethz.ch:/pub/hofmann-scratch/students/durnovv/probability-of-a-word/results/rt/joint_full/*' \
  results/rt/joint_full/
```

If a commit exists only in the cluster checkout:

```bash
git fetch durnovv@mark.inf.ethz.ch:probability-of-a-word-experiments main
git merge --ff-only FETCH_HEAD
```

The initial GitHub SSH problems were operational rather than experimental:
the generated key was first saved under the literal name `key`, then moved to
`~/.ssh/id_ed25519`; after the public key was registered, authentication
succeeded as GitHub user `votttak`. A later "Repository not found" error was
fixed by creating/correcting the remote repository URL.

## 12. Artifact map

### Core documentation and orchestration

- [`NGRAMS.md`](NGRAMS.md): n-gram semantics and cache behavior.
- [`CONTEXT_LIMITED.md`](CONTEXT_LIMITED.md): fixed-context scorer semantics.
- [`JOINT_PILOT.md`](JOINT_PILOT.md): paired delta definitions.
- [`MakefileJointFull`](MakefileJointFull): full Natural Stories pipeline.
- [`MakefileJointPilot`](MakefileJointPilot): 500-word pilot.
- [`scripts/run_all_joint_full_context_surprisals.sh`](scripts/run_all_joint_full_context_surprisals.sh):
  context-only all-model runner.

### Core implementation

- [`src/h01_data/get_ngram_surprisals.py`](src/h01_data/get_ngram_surprisals.py)
- [`src/h01_data/get_context_limited_surprisals.py`](src/h01_data/get_context_limited_surprisals.py)
- [`src/h01_data/build_joint_pilot_dataset.py`](src/h01_data/build_joint_pilot_dataset.py)
- [`src/h02_rt_model/rt_vs_joint_pilot_surprisal.R`](src/h02_rt_model/rt_vs_joint_pilot_surprisal.R)

### Result-file meanings

Within a joint result directory:

| File | Meaning |
|---|---|
| `summary.tsv` | sample, folds, seed, response, contexts, controls |
| `fold-results.tsv` | all model scores and paired deltas for every cell/fold |
| `delta-a-given-b-mean.tsv` | mean unique n-gram contribution |
| `delta-b-given-a-mean.tsv` | mean unique transformer contribution |
| `delta-a-given-b-se.tsv` | fold SE of unique n-gram contribution |
| `delta-b-given-a-se.tsv` | fold SE of unique transformer contribution |

The old `context_1-2-4` directory is provenance, not the preferred analysis.
Use the `context_1-2-3-4` directory.

### Current local artifact status

At the code baseline recorded here:

- final GPT-2-small `context_1-2-3-4` results are tracked;
- ten full context-limited predictor directories are present locally but
  untracked;
- pilot checkpoints/results are present locally but untracked;
- corrected consolidated ten-model RT results exist only as a validated local
  re-analysis under `/tmp` and should be made durable.

Individual result TSVs do not contain a `model` column; model identity is
encoded only by the parent directory. A consolidated multi-model artifact must
include an explicit model column.

Do not delete or overwrite the untracked data while resolving provenance.

## 13. Recommended next steps

Priority order:

1. **Archive the temporary corrected analysis now.** Copy the checksummed
   cluster-derived base, consolidated delta table, best-cell table, and
   per-model fold outputs out of
   `/tmp/joint-allmodels-clusterbase.RKERrM` before temporary storage is
   cleared.

2. **Resolve and document the RT base mismatch.** Identify why the local and
   cluster `time` columns differ. Choose one source deliberately and record raw
   input and processed-file checksums.

3. **Make the ten-model analysis reproducible.** Refactor the full Makefile so
   all transformer models reuse one model-neutral RT/control base and one common
   n-gram file. Currently `MODEL` unnecessarily binds the context LM, the
   full-context base file, and a duplicated n-gram path.

4. **Persist canonical all-model outputs.** Save a versioned consolidated
   fold-level table and per-model summaries together with the exact code
   commit, environment versions, HF revisions, dtype, GPU, and input hashes.

5. **Run grouped or blocked validation.** Prefer leave-one-story-out or
   story-grouped folds; also consider contiguous blocked folds. Compare these
   with the current random-word estimates.

6. **Use a richer RT model if data permit.** Analyze participant-level trials
   with participant/item random effects, or at minimum test log RT and
   heteroskedastic/robust alternatives.

7. **Add interpretable prediction metrics.** Save held-out predictions so MSE,
   MAE, cross-validated R-squared, and percentage MSE reduction can be reported
   alongside log density.

8. **Define primary comparisons before another sweep.** For example, predefine
   matched contexts and the unigram-plus-context-4 cell, and control or clearly
   label multiplicity for model/context selection.

9. **Pin the environment.** Remove obsolete `wikitokenizer`, lock dependency
   versions, pin Hugging Face model revisions, and include precision metadata
   in output identity. Also archive the raw corpus inputs and model stderr
   metadata.

10. **Treat Pythia 12B as optional.** Run it only on a GPU with at least 32 GB
    or after implementing explicit sharding/offload. It is not necessary for
    the main ten-model conclusion.

11. **Prepare figures.** Useful paper figures are:
    - heatmaps of `delta transformer given n-gram`;
    - matched-context conditional deltas with fold variability;
    - model-size versus unique RT contribution within family;
    - unigram-only, transformer-only, and joint predictive-density gains.

## 14. Draft paper-ready wording

### Methods draft

> We evaluated whether fixed-context transformer surprisal and corpus n-gram
> surprisal contributed independent information about Natural Stories reading
> times. The analysis included 10,023 complete word observations. Baseline OLS
> models predicted raw word-level reading time from word length, unigram
> frequency, and their interaction at the current word and three preceding
> words. For each n-gram/transformer context pair, predictor surprisals and
> three spillover positions were added individually and jointly. We used the
> same deterministic ten-fold word-level split for every comparison and scored
> held-out observations with Gaussian log density using residual variance
> estimated on the training fold. Unique contribution was the paired
> difference between the joint model and the corresponding single-predictor
> model, expressed in mean held-out log density per observation.

### Results draft

> Context-limited transformer surprisal contributed positive held-out
> predictive density beyond n-gram surprisal in all 200 evaluated
> model-by-context cells. In matched-window comparisons, the mean unique
> transformer contribution increased from 0.0053 nats per observation with one
> preceding word to 0.0226 with four preceding words. The corresponding unique
> contribution of matched contextual n-grams was small, ranging in
> across-model mean from -0.0004 to 0.0008. A context-free corpus unigram
> predictor remained complementary: every model's descriptively best joint
> condition combined transformer surprisal with n-gram context 0. The largest
> selected joint gain was 0.0494 nats per observation, equivalent to 5.06%
> higher geometric-mean predictive density than the lexical-control model.

### Required qualification

> These uncertainty summaries are descriptive across ten overlapping
> cross-validation folds, and model/context rankings are exploratory. The
> percentages describe predictive-density ratios, not variance explained.
> Because folds were assigned at the word level, confirmatory analysis should
> additionally use story-grouped or blocked validation.
