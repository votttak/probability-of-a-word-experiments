# Semester project experiment summary

- Status: 2026-08-16
- Code baseline: `f3497ce`
- Full technical record:
  [`SEMESTER_PROJECT_EXPERIMENT_RECORD.md`](SEMESTER_PROJECT_EXPERIMENT_RECORD.md)

## Research question

Does context-limited transformer surprisal predict human reading time beyond an
n-gram with the same number of preceding words? Conversely, does the n-gram
add information beyond the transformer?

Let `M0` be the lexical-control model, `A` the n-gram predictor, and `B` the
context-limited transformer:

```text
delta(A | B) = LL(M0 + A + B) - LL(M0 + B)
delta(B | A) = LL(M0 + A + B) - LL(M0 + A)
```

A positive delta means better held-out prediction after adding that predictor.

## Method

- Data: 10 Natural Stories passages, 10,256 word rows.
- Analysis sample: 10,023 complete rows; 233 excluded because current or
  spillover frequency controls were missing.
- N-gram contexts: 0, 1, 2, 3, and 4 preceding words, using Infini-gram
  `v4_piletrain_llama` and Stupid Backoff with alpha 0.4.
- Transformer contexts: 1, 2, 3, and 4 preceding words.
- Models: GPT-2 small/medium/large/XL and Pythia
  70M/160M/410M/1.4B/2.8B/6.9B.
- Pythia 12B was skipped because it cannot fit the current single 16 GB GPU
  loader.
- Controls: word length, unigram frequency, and their interaction at the
  current word and three preceding positions.
- Each surprisal predictor was also entered at the current word and three
  spillover positions.
- Evaluation: deterministic 10-fold word-level cross-validation, seed 42,
  using OLS on raw mean RT and Gaussian held-out log density.

Context values count preceding words. Therefore n-gram context 4 is
conventionally a 5-gram; context 0 is a unigram/frequency-like predictor.

Reported percentages use:

```text
100 * (exp(mean log-density delta) - 1)
```

They are changes in geometric-mean held-out predictive density. They are not
percent RT explained, variance explained, or R-squared.

## Main results

Across the ten completed transformer models, the matched-context results were:

| Context | Mean n-gram given transformer [model range] | Models above 1.96 fold SE | Mean transformer given n-gram [model range] | Models above 1.96 fold SE |
|---:|---:|---:|---:|---:|
| 1 | +0.000239 [-0.000418, +0.002343] | 2/10 | +0.005262 [+0.002969, +0.009191] | 10/10 |
| 2 | -0.000393 [-0.000506, -0.000289] | 0/10 | +0.012386 [+0.009318, +0.017662] | 10/10 |
| 3 | +0.000468 [+0.000179, +0.000768] | 0/10 | +0.020641 [+0.016251, +0.027390] | 10/10 |
| 4 | +0.000822 [+0.000297, +0.001127] | 5/10 | +0.022563 [+0.015877, +0.033037] | 10/10 |

The transformer contribution was positive and greater than 1.96 descriptive
fold standard errors in all 200 model-by-context cells. The n-gram contribution
was positive in 148/200 cells and greater than 1.96 fold standard errors in
77/200. Its reliable contribution was concentrated in n-gram context 0.

Important reference values:

- Transformer-only gains over lexical controls ranged from `2.056%` to
  `3.738%` in predictive density.
- The unigram n-gram alone added `0.039651` nats/observation, or `4.045%`.
- Every model's descriptively best joint condition included the unigram
  predictor.
- Best selected joint condition: Pythia 70M, n-gram context 0 plus transformer
  context 4, with a `0.049402`-nat gain or `5.064%` higher predictive density.
- GPT-2-small's best joint condition was also context 0 plus context 4:
  `0.047214` nats or `4.835%`.

The original 500-word pilot had 454 complete cases, was noisy, and is not
primary evidence. The first full GPT-2-small run omitted transformer context 3;
the corrected 1/2/3/4 run is definitive. Shared cells between the two full runs
agree within `2.42e-9`.

## Interpretation

The primary supported conclusion is that fixed-context transformer surprisal
contains substantial held-out RT information that matched higher-order
n-grams do not contain. Once transformer surprisal is included, contextual
n-gram gains are generally very small.

The unigram result is different: it remains strongly complementary even though
the baseline already has a frequency control. It should be interpreted as an
additional lexical-frequency signal from another corpus, not as contextual
n-gram evidence.

Larger models produced lower mean corpus surprisal, but usually smaller unique
RT gains at contexts 2-4. Smaller models, especially Pythia 70M, were
descriptively strongest. This scaling pattern is exploratory and does not show
that small models are universally more human-like.

## Essential limitations

- Folds were assigned by word, so neighboring words and the same stories occur
  in both training and test data. Story-grouped or blocked validation is still
  needed.
- Fold `SE = sd(delta)/sqrt(10)` is descriptive because training folds overlap;
  it is not a formal confidence interval or p-value.
- RT is aggregated by word. The models have no participant/item random effects
  and no observation-count weighting.
- Best cells and models were chosen from a large exploratory grid, creating
  multiplicity and selection concerns.
- Stupid-Backoff n-gram scores are not normalized probabilities.
- Model families differ in tokenizer, training data, and other properties, so
  cross-family size comparisons are confounded.

## Critical reproducibility warning

The pre-existing local merged RT checkpoint does not reproduce the completed
cluster experiment: its `time` values differ on all 10,256 rows, with mean
absolute difference 15.68 ms and maximum difference about 200 ms. The
corrected ten-model analysis used the RT/control base reconstructed from the
copied cluster GPT-2-small `joint-data.tsv`; that reconstruction matches saved
GPT-2-small means within `1.01e-15` and fold rows within `1.01e-11`.

The corrected multi-model outputs currently live under:

```text
/tmp/joint-allmodels-clusterbase.RKERrM/
```

They must be archived before `/tmp` is cleared. The full GPT-2-small
context-1/2/3/4 result is already tracked in
[`results/rt/joint_full`](results/rt/joint_full).

## Next steps

1. Archive the corrected base and all-model fold/results tables immediately.
2. Resolve the local-versus-cluster RT mismatch and record input checksums.
3. Make the Makefile reuse one canonical RT/control base and one common n-gram
   file across transformer models.
4. Run story-grouped or blocked cross-validation.
5. Save held-out predictions to report MSE, MAE, and cross-validated R-squared.
6. Treat the transformer-over-n-gram result as primary, the unigram
   complementarity as secondary, and model-size rankings as exploratory.
