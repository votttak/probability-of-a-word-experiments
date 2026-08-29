#!/usr/bin/env Rscript

source('src/h02_rt_model/rt_vs_internal_layer_kuribayashi_replication.R')


layer_1 <- 'internal_layer_surprisal_layer_1'
reduced <- make_kuribayashi_reduced_formula(layer_1)
full <- make_kuribayashi_full_formula(layer_1)
reduced_variables <- all.vars(reduced)
full_variables <- all.vars(full)

stopifnot('prev_internal_layer_surprisal_layer_1' %in% reduced_variables)
stopifnot('prev2_internal_layer_surprisal_layer_1' %in% reduced_variables)
stopifnot(!layer_1 %in% reduced_variables)
stopifnot(!'prev3_internal_layer_surprisal_layer_1' %in% reduced_variables)
stopifnot(identical(setdiff(full_variables, reduced_variables), layer_1))
stopifnot(all(lexical_control_variables %in% reduced_variables))


set.seed(8128)
n <- 240L
synthetic <- data.frame(
  time=numeric(n),
  word_len=runif(n, 1, 12),
  freq=rnorm(n),
  prev_word_len=runif(n, 1, 12),
  prev_freq=rnorm(n),
  prev2_word_len=runif(n, 1, 12),
  prev2_freq=rnorm(n),
  prev3_word_len=runif(n, 1, 12),
  prev3_freq=rnorm(n)
)

for (layer in 1:2) {
  variable <- paste0('internal_layer_surprisal_layer_', layer)
  synthetic[[variable]] <- rnorm(n)
  synthetic[[paste0('prev_', variable)]] <- rnorm(n)
  synthetic[[paste0('prev2_', variable)]] <- rnorm(n)
  synthetic[[paste0('prev3_', variable)]] <- rnorm(n)
}

synthetic$time <- with(synthetic,
  250 + 1.7 * word_len - 2.1 * freq +
  4.5 * internal_layer_surprisal_layer_1 +
  0.8 * prev_internal_layer_surprisal_layer_1 -
  0.4 * prev2_internal_layer_surprisal_layer_1 + rnorm(n, sd=1.2)
)
synthetic$prev3_internal_layer_surprisal_layer_2[[17]] <- NA_real_

layers <- validate_replication_layers(discover_predictors(
  synthetic, 'internal_layer_surprisal_layer_'
))
prepared <- prepare_kuribayashi_replication_data(
  synthetic, layers$variable
)
stopifnot(nrow(prepared) == n - 1L)

reduced_fit <- fit_in_sample_model(
  make_kuribayashi_reduced_formula(layer_1), prepared
)
full_fit <- fit_in_sample_model(
  make_kuribayashi_full_formula(layer_1), prepared
)
stopifnot(nobs(reduced_fit$model) == nobs(full_fit$model))
stopifnot(length(coef(full_fit$model)) == length(coef(reduced_fit$model)) + 1L)
stopifnot(full_fit$log_likelihood > reduced_fit$log_likelihood)


temporary_dir <- tempfile('kuribayashi-replication-test-')
dir.create(temporary_dir)
input_fname <- file.path(temporary_dir, 'input.tsv')
layer_fname <- file.path(temporary_dir, 'layer-results.tsv')
best_fname <- file.path(temporary_dir, 'best-layer.tsv')
summary_fname <- file.path(temporary_dir, 'summary.tsv')
write.table(
  synthetic, input_fname, quote=FALSE, sep='\t', row.names=FALSE, na='NA'
)
result <- run_kuribayashi_replication(
  input_fname, layer_fname, best_fname, summary_fname, model='synthetic'
)

stopifnot(file.exists(layer_fname), file.exists(best_fname), file.exists(summary_fname))
stopifnot(identical(result$layers$layer, 1:2))
stopifnot(sum(result$layers$is_best_layer) == 1L)
stopifnot(result$best_layer$layer[[1]] == 1L)
stopifnot(result$layers$analysis_rows[[1]] == n - 1L)
stopifnot(all(result$layers$analysis_rows == nrow(result$analysis_data)))
stopifnot(all(is.finite(result$layers$delta_ll)))
stopifnot(all.equal(
  result$layers$ppp_x1000,
  1000 * result$layers$delta_ll / result$layers$input_rows
))
stopifnot(result$layers$relative_depth_block[[1]] == 0)
stopifnot(result$layers$relative_depth_block[[2]] == 1)

summary_values <- setNames(result$summary$value, result$summary$key)
stopifnot(summary_values[['evaluation']] == 'in-sample Gaussian OLS log likelihood')
stopifnot(grepl('formula bridge', summary_values[['replication_scope']]))
stopifnot(grepl('text-bounded', summary_values[['spillover_boundary_policy']]))

unlink(temporary_dir, recursive=TRUE)
