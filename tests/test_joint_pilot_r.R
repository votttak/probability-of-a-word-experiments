#!/usr/bin/env Rscript

source('src/h02_rt_model/rt_vs_joint_pilot_surprisal.R')

synthetic <- data.frame(
  ngram_surprisal_context_10=1:4,
  ngram_surprisal_context_2=1:4,
  context_limited_surprisal_context_4=1:4,
  context_limited_surprisal_context_1=1:4
)
ngram <- discover_predictors(synthetic, 'ngram_surprisal_context_')
context <- discover_predictors(
  synthetic, 'context_limited_surprisal_context_'
)
stopifnot(identical(ngram$context, c(2L, 10L)))
stopifnot(identical(context$context, c(1L, 4L)))

formula_text <- paste(deparse(make_model_formula(c(
  'ngram_surprisal_context_2',
  'context_limited_surprisal_context_1'
))), collapse=' ')
stopifnot(grepl('word_len * freq', formula_text, fixed=TRUE))
stopifnot(grepl('prev3_ngram_surprisal_context_2', formula_text))
stopifnot(grepl('prev3_context_limited_surprisal_context_1', formula_text))

fold_rows <- expand.grid(
  ngram_context=c(0, 1),
  context_limited_context=c(2, 4),
  fold=1:2
)
fold_rows$delta_a_given_b_mean <- with(
  fold_rows, ngram_context + context_limited_context + fold
)
pivot <- make_pivot(
  fold_rows, 'delta_a_given_b_mean', c(0, 1), c(2, 4), 'mean'
)
stopifnot(identical(pivot$context_limited_context, c(2, 4)))
stopifnot(all.equal(pivot$ngram_context_0, c(3.5, 5.5)))
stopifnot(all.equal(pivot$ngram_context_1, c(4.5, 6.5)))
