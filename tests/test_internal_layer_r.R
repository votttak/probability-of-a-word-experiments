#!/usr/bin/env Rscript

source('src/h02_rt_model/rt_vs_internal_layer_surprisal.R')

synthetic <- data.frame(
  internal_layer_surprisal_layer_12=1:4,
  internal_layer_surprisal_layer_6=1:4
)
layers <- discover_predictors(
  synthetic, 'internal_layer_surprisal_layer_'
)
stopifnot(identical(layers$context, c(6L, 12L)))
layer_zero_rejected <- tryCatch(
  {
    validate_internal_layers(data.frame(context=0L))
    FALSE
  },
  error=function(error) grepl('transformer layer 1', conditionMessage(error))
)
stopifnot(layer_zero_rejected)

fold_rows <- expand.grid(
  predictor_family=c('ngram', 'context_limited'),
  predictor_context=c(1, 2),
  layer=c(1, 3),
  fold=1:2,
  stringsAsFactors=FALSE
)
fold_rows$comparison <- paste0(
  fold_rows$predictor_family, '_vs_internal_layer'
)
fold_rows$delta_predictor_given_layer_mean <- with(
  fold_rows, predictor_context + layer + fold
)
fold_rows$delta_layer_given_predictor_mean <- with(
  fold_rows, layer - predictor_context + fold
)
aggregated <- aggregate_conditional_deltas(fold_rows)
selected <- aggregated[
  aggregated$predictor_family == 'ngram' &
  aggregated$predictor_context == 1 & aggregated$layer == 3,
]
stopifnot(nrow(selected) == 1)
stopifnot(all.equal(selected$delta_predictor_given_layer_mean, 5.5))
stopifnot(all.equal(selected$delta_layer_given_predictor_mean, 3.5))
