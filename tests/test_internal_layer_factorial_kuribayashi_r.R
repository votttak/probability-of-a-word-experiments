#!/usr/bin/env Rscript

source('src/h02_rt_model/rt_vs_internal_layer_factorial_kuribayashi.R')


set.seed(20260825)
n_sentences <- 60L
words_per_sentence <- 6L
n <- n_sentences * words_per_sentence
synthetic <- data.frame(
  model=rep('synthetic-gpt', n),
  context_unit=rep('sentence', n),
  first_token_policy=rep('bow', n),
  sentence_first_token_policy=rep('bow', n),
  lens_method=rep('tuned-lens', n),
  include_embedding_layer=rep(TRUE, n),
  lag_boundary=rep('text', n),
  lag_padding=rep('missing', n),
  text_id=rep(0L, n),
  sentence_id=rep(seq.int(0L, n_sentences - 1L), each=words_per_sentence),
  sentence_word_id=rep(seq.int(0L, words_per_sentence - 1L), n_sentences),
  time=numeric(n),
  length=sample(1:12, n, replace=TRUE),
  log_gmean_freq=rnorm(n, -9, 2),
  word_len=runif(n, 1, 12),
  freq=rnorm(n),
  prev_word_len=runif(n, 1, 12),
  prev_freq=rnorm(n),
  prev2_word_len=runif(n, 1, 12),
  prev2_freq=rnorm(n),
  prev3_word_len=runif(n, 1, 12),
  prev3_freq=rnorm(n),
  stringsAsFactors=FALSE
)

for (prefix in factorial_score_specs$predictor_prefix) {
  for (layer in 0:2) {
    variable <- paste0(prefix, layer)
    synthetic[[variable]] <- rnorm(n)
    synthetic[[paste0('prev_', variable)]] <- rnorm(n)
    synthetic[[paste0('prev2_', variable)]] <- rnorm(n)
    synthetic[[paste0('prev3_', variable)]] <- rnorm(n)
  }
}

corrected_0 <- 'internal_layer_surprisal_layer_0'
buggy_1 <- 'internal_layer_surprisal_buggy_layer_1'
synthetic$time <- with(synthetic,
  300 + 1.5 * length - 1.2 * log_gmean_freq +
  6.0 * internal_layer_surprisal_layer_0 +
  0.3 * internal_layer_surprisal_layer_1 +
  0.1 * internal_layer_surprisal_layer_2 +
  0.2 * internal_layer_surprisal_buggy_layer_0 +
  5.0 * internal_layer_surprisal_buggy_layer_1 +
  0.1 * internal_layer_surprisal_buggy_layer_2 +
  rnorm(n, sd=1.0)
)
synthetic$paper_time <- synthetic$time + rnorm(n, sd=0.05)


metadata <- read_factorial_metadata(synthetic)
stopifnot(metadata$model == 'synthetic-gpt')
stopifnot(metadata$include_embedding_layer)
predictors <- discover_factorial_predictors(
  synthetic, metadata$include_embedding_layer
)
stopifnot(nrow(predictors) == 6L)
stopifnot(identical(unique(predictors$context), 0:2))
stopifnot(identical(
  unique(predictors$score_kind), c('corrected', 'buggy')
))

reduced <- make_paper_exact_reduced_formula(corrected_0)
full <- make_paper_exact_full_formula(corrected_0)
reduced_variables <- all.vars(reduced)
full_variables <- all.vars(full)
stopifnot(paste0('prev_', corrected_0) %in% reduced_variables)
stopifnot(paste0('prev2_', corrected_0) %in% reduced_variables)
stopifnot(!corrected_0 %in% reduced_variables)
stopifnot(identical(setdiff(full_variables, reduced_variables), corrected_0))
stopifnot(all(paper_exact_control_variables %in% reduced_variables))
stopifnot(!grepl('\\*', deparse(reduced)))

prepared <- prepare_paper_exact_data(synthetic, predictors$variable)
stopifnot(nrow(prepared$data) == n - n_sentences)
ordered <- prepared$ordered_data
corrected_mean <- mean(ordered[[corrected_0]])
sentence_starts <- ordered$sentence_word_id == 0L
sentence_seconds <- ordered$sentence_word_id == 1L
stopifnot(all(
  ordered[[paste0('prev_', corrected_0)]][sentence_starts] == corrected_mean
))
stopifnot(all(
  ordered[[paste0('prev2_', corrected_0)]][sentence_starts] == corrected_mean
))
stopifnot(all(
  ordered[[paste0('prev2_', corrected_0)]][sentence_seconds] == corrected_mean
))
stopifnot(all(
  ordered$length_prev_1[sentence_starts] == mean(ordered$length)
))
stopifnot(all(
  ordered$log_gmean_freq_prev_2[
    ordered$sentence_word_id <= 1L
  ] == mean(ordered$log_gmean_freq)
))


temporary_dir <- tempfile('factorial-kuribayashi-test-')
dir.create(temporary_dir)
input_fname <- file.path(temporary_dir, 'merged.tsv')
layer_fname <- file.path(temporary_dir, 'layer-results.tsv')
best_fname <- file.path(temporary_dir, 'best-layers.tsv')
summary_fname <- file.path(temporary_dir, 'summary.tsv')
write.table(
  synthetic, input_fname, quote=FALSE, sep='\t', row.names=FALSE, na='NA'
)

# Paper-exact is the API and CLI default.
parsed_default <- parse_factorial_cli_args(c(
  input_fname, layer_fname, best_fname, summary_fname
))
stopifnot(parsed_default$analysis_mode == 'paper-exact')
stopifnot(parsed_default$response_column == 'time')
parsed_bridge <- parse_factorial_cli_args(c(
  input_fname, layer_fname, best_fname, summary_fname,
  '--response-column', 'paper_time',
  '--analysis-mode', 'project-bridge'
))
stopifnot(parsed_bridge$analysis_mode == 'project-bridge')
stopifnot(parsed_bridge$response_column == 'paper_time')

result <- run_factorial_kuribayashi_evaluation(
  input_fname, layer_fname, best_fname, summary_fname,
  response_column='paper_time'
)

stopifnot(file.exists(layer_fname), file.exists(best_fname), file.exists(summary_fname))
stopifnot(nrow(result$layers) == 6L)
stopifnot(nrow(result$best_layers) == 2L)
stopifnot(all(
  result$layers$analysis == 'kuribayashi_paper_exact_L_nesting'
))
stopifnot(all(result$layers$analysis_mode == 'paper-exact'))
stopifnot(all(result$layers$response_column == 'paper_time'))
stopifnot(all(result$layers$model == 'synthetic-gpt'))
stopifnot(all(result$layers$context_unit == 'sentence'))
stopifnot(all(result$layers$first_token_policy == 'bow'))
stopifnot(all(result$layers$sentence_first_token_policy == 'bow'))
stopifnot(all(result$layers$lens_method == 'tuned-lens'))
stopifnot(all(result$layers$include_embedding_layer))
stopifnot(all(result$layers$lag_boundary == 'text'))
stopifnot(all(result$layers$lag_padding == 'missing'))
stopifnot(all(result$layers$analysis_lag_boundary == 'sentence'))
stopifnot(all(result$layers$analysis_lag_padding == 'global-mean'))
stopifnot(identical(
  result$layers$score_kind,
  c('corrected', 'corrected', 'corrected', 'buggy', 'buggy', 'buggy')
))
stopifnot(identical(result$layers$layer, c(0L, 1L, 2L, 0L, 1L, 2L)))
stopifnot(identical(
  result$layers$is_embedding_layer,
  c(TRUE, FALSE, FALSE, TRUE, FALSE, FALSE)
))
stopifnot(sum(result$layers$is_best_layer) == 2L)
best_by_kind <- setNames(
  result$best_layers$layer, result$best_layers$score_kind
)
stopifnot(best_by_kind[['corrected']] == 0L)
stopifnot(best_by_kind[['buggy']] == 1L)
stopifnot(nrow(result$analysis_data) == n - n_sentences)
stopifnot(all.equal(
  result$analysis_data$time,
  result$analysis_data$paper_time,
  check.attributes=FALSE
))
stopifnot(all(result$layers$analysis_rows == n - n_sentences))
stopifnot(all(result$layers$input_rows == n))
stopifnot(all(is.finite(result$layers$delta_ll)))
stopifnot(all.equal(
  result$layers$ppp_x1000,
  1000 * result$layers$delta_ll / result$layers$input_rows
))
stopifnot(all(result$layers$min_layer == 0L))
stopifnot(all(result$layers$max_layer == 2L))
stopifnot(all.equal(
  result$layers$relative_depth_block[1:3], c(0, 0.5, 1)
))

# One factorial cell must agree exactly with the paper-exact primitive.
direct <- evaluate_paper_exact_layer(
  layer=0L, variable=corrected_0, df=result$analysis_data,
  model='synthetic-gpt', min_layer=0L, max_layer=2L, input_rows=n
)
factorial_cell <- result$layers[
  result$layers$score_kind == 'corrected' & result$layers$layer == 0L,
]
stopifnot(all.equal(factorial_cell$ll_reduced, direct$ll_reduced))
stopifnot(all.equal(factorial_cell$ll_full, direct$ll_full))
stopifnot(all.equal(factorial_cell$delta_ll, direct$delta_ll))

summary_values <- setNames(result$summary$value, result$summary$key)
stopifnot(summary_values[['analysis_mode']] == 'paper-exact')
stopifnot(summary_values[['response_column']] == 'paper_time')
stopifnot(summary_values[['design']] == 'corrected_by_buggy_factorial')
stopifnot(summary_values[['complete_case_rows']] == as.character(
  n - n_sentences
))
stopifnot(summary_values[['best_layer_corrected']] == '0')
stopifnot(summary_values[['best_layer_buggy']] == '1')
stopifnot(summary_values[['include_embedding_layer']] == 'TRUE')
stopifnot(summary_values[['lag_boundary']] == 'text')
stopifnot(summary_values[['analysis_lag_boundary']] == 'sentence')
stopifnot(summary_values[['analysis_lag_padding']] == 'global-mean')
stopifnot(grepl('mean-pad', summary_values[['sample_policy']]))
stopifnot(grepl('sentence position > 0', summary_values[['target_filter']]))
stopifnot(grepl('paper_time', summary_values[['response']]))

# The optional bridge mode must still use the pre-existing formulas and t-3
# canonical sample guard, while accepting the complete embedding-to-final range.
bridge_layer_fname <- file.path(temporary_dir, 'bridge-layer-results.tsv')
bridge_best_fname <- file.path(temporary_dir, 'bridge-best-layers.tsv')
bridge_summary_fname <- file.path(temporary_dir, 'bridge-summary.tsv')
bridge_result <- run_factorial_kuribayashi_evaluation(
  input_fname, bridge_layer_fname, bridge_best_fname, bridge_summary_fname,
  analysis_mode='project-bridge'
)
stopifnot(all(bridge_result$layers$analysis == 'kuribayashi_L_nesting'))
stopifnot(all(bridge_result$layers$analysis_mode == 'project-bridge'))
stopifnot(nrow(bridge_result$analysis_data) == n)
stopifnot(all(bridge_result$layers$analysis_lag_boundary == 'text'))
stopifnot(all(bridge_result$layers$analysis_lag_padding == 'missing'))
stopifnot(all(bridge_result$layers$is_embedding_layer ==
              (bridge_result$layers$layer == 0L)))


missing_buggy <- synthetic[, !grepl(
  'internal_layer_surprisal_buggy_layer_', colnames(synthetic), fixed=TRUE
)]
missing_buggy_rejected <- tryCatch(
  {
    discover_factorial_predictors(missing_buggy, TRUE)
    FALSE
  },
  error=function(error) grepl('No predictors found', conditionMessage(error))
)
stopifnot(missing_buggy_rejected)

mismatched_layers <- synthetic[, !grepl(
  'buggy_layer_2', colnames(synthetic), fixed=TRUE
)]
mismatched_layers_rejected <- tryCatch(
  {
    discover_factorial_predictors(mismatched_layers, TRUE)
    FALSE
  },
  error=function(error) grepl(
    'complete consecutive|same layer range', conditionMessage(error)
  )
)
stopifnot(mismatched_layers_rejected)

embedding_metadata_mismatch <- synthetic
embedding_metadata_mismatch$include_embedding_layer <- FALSE
embedding_mismatch_rejected <- tryCatch(
  {
    mismatch_metadata <- read_factorial_metadata(embedding_metadata_mismatch)
    discover_factorial_predictors(
      embedding_metadata_mismatch,
      mismatch_metadata$include_embedding_layer
    )
    FALSE
  },
  error=function(error) grepl('range 1..D', conditionMessage(error), fixed=TRUE)
)
stopifnot(embedding_mismatch_rejected)

mixed_metadata <- synthetic
mixed_metadata$model[[2]] <- 'different-model'
mixed_metadata_rejected <- tryCatch(
  {
    read_factorial_metadata(mixed_metadata)
    FALSE
  },
  error=function(error) grepl(
    'multiple experiment values', conditionMessage(error)
  )
)
stopifnot(mixed_metadata_rejected)

missing_response_rejected <- tryCatch(
  {
    run_factorial_kuribayashi_evaluation(
      input_fname,
      file.path(temporary_dir, 'missing-response-layers.tsv'),
      file.path(temporary_dir, 'missing-response-best.tsv'),
      file.path(temporary_dir, 'missing-response-summary.tsv'),
      response_column='does_not_exist'
    )
    FALSE
  },
  error=function(error) grepl('missing response column', conditionMessage(error))
)
stopifnot(missing_response_rejected)

bad_position <- synthetic
bad_position$sentence_word_id[[2]] <- 0L
bad_position_rejected <- tryCatch(
  {
    prepare_paper_exact_data(bad_position, predictors$variable)
    FALSE
  },
  error=function(error) grepl('position range', conditionMessage(error))
)
stopifnot(bad_position_rejected)

same_output_rejected <- tryCatch(
  {
    validate_factorial_paths(
      input_fname, c(layer_fname, layer_fname, summary_fname)
    )
    FALSE
  },
  error=function(error) grepl('must be distinct', conditionMessage(error))
)
stopifnot(same_output_rejected)

unlink(temporary_dir, recursive=TRUE)
