#!/usr/bin/env Rscript

# Standalone factorial evaluation for corrected and historical-buggy internal
# layer scores.  Paper-exact mode reconstructs Kuribayashi et al.'s Natural
# Stories controls, target filtering, mean-padded sentence lags, and nested OLS
# contrast.  Project-bridge mode delegates to the existing replication helper.

source('src/h02_rt_model/rt_vs_internal_layer_kuribayashi_replication.R')


factorial_score_specs <- data.frame(
  score_kind=c('corrected', 'buggy'),
  predictor_prefix=c(
    'internal_layer_surprisal_layer_',
    'internal_layer_surprisal_buggy_layer_'
  ),
  stringsAsFactors=FALSE
)

factorial_metadata_columns <- c(
  'model', 'context_unit', 'lens_method', 'first_token_policy',
  'sentence_first_token_policy', 'include_embedding_layer',
  'lag_boundary', 'lag_padding'
)

factorial_analysis_modes <- c('paper-exact', 'project-bridge')

factorial_default_score_kinds <- factorial_score_specs$score_kind

paper_exact_control_variables <- c(
  'length', 'log_gmean_freq',
  'length_prev_1', 'log_gmean_freq_prev_1',
  'length_prev_2', 'log_gmean_freq_prev_2'
)

paper_exact_control_formula <- paste(
  paper_exact_control_variables, collapse=' + '
)


read_scalar_metadata <- function(df, column) {
  if (!column %in% colnames(df)) {
    stop(paste('Input is missing required metadata column:', column))
  }
  raw_values <- as.character(df[[column]])
  invalid <- is.na(raw_values) | trimws(raw_values) == ''
  if (any(invalid)) {
    stop(paste('Metadata column contains missing or empty values:', column))
  }
  values <- unique(trimws(raw_values))
  if (length(values) != 1L) {
    stop(paste('Input contains multiple experiment values for', column))
  }
  values[[1]]
}


parse_boolean_metadata <- function(value, column) {
  normalized <- tolower(trimws(as.character(value)))
  if (normalized %in% c('true', 't', '1')) {
    return(TRUE)
  }
  if (normalized %in% c('false', 'f', '0')) {
    return(FALSE)
  }
  stop(paste(column, 'must be true or false'))
}


read_factorial_metadata <- function(df) {
  values <- lapply(
    factorial_metadata_columns,
    function(column) read_scalar_metadata(df, column)
  )
  names(values) <- factorial_metadata_columns
  values$include_embedding_layer <- parse_boolean_metadata(
    values$include_embedding_layer, 'include_embedding_layer'
  )

  if (!values$context_unit %in% c('passage', 'sentence')) {
    stop('context_unit must be passage or sentence')
  }
  if (!values$first_token_policy %in% c('bos', 'bow') ||
      !values$sentence_first_token_policy %in% c('bos', 'bow')) {
    stop('first-token policy metadata must be bos or bow')
  }
  if (values$first_token_policy != values$sentence_first_token_policy) {
    stop('first_token_policy and sentence_first_token_policy disagree')
  }
  if (values$context_unit == 'passage' &&
      values$first_token_policy != 'bos') {
    stop('Passage context requires first_token_policy=bos')
  }
  if (!values$lens_method %in% c('logit-lens', 'tuned-lens')) {
    stop('lens_method must be logit-lens or tuned-lens')
  }
  if (!values$lag_boundary %in% c('text', 'sentence')) {
    stop('lag_boundary must be text or sentence')
  }
  if (!values$lag_padding %in% c('missing', 'global-mean')) {
    stop('lag_padding must be missing or global-mean')
  }
  values
}


validate_factorial_layer_family <- function(layers, include_embedding_layer) {
  expected_start <- if (include_embedding_layer) 0L else 1L
  if (include_embedding_layer && max(layers$context) < 1L) {
    stop('Embedding-inclusive evaluation requires transformer layers 0..D')
  }
  expected <- seq.int(expected_start, max(layers$context))
  if (!identical(layers$context, expected)) {
    stop(paste0(
      'Factorial evaluation requires the complete consecutive layer range ',
      expected_start, '..D'
    ))
  }
  layers
}


validate_factorial_score_kinds <- function(score_kinds) {
  if (!is.character(score_kinds) || length(score_kinds) == 0L ||
      any(is.na(score_kinds))) {
    stop('score_kinds must contain at least one score kind')
  }
  score_kinds <- trimws(score_kinds)
  if (any(score_kinds == '')) {
    stop('score_kinds must not contain empty values')
  }
  if (anyDuplicated(score_kinds)) {
    stop('score_kinds must contain unique values')
  }
  unsupported <- setdiff(score_kinds, factorial_score_specs$score_kind)
  if (length(unsupported) > 0L) {
    stop(paste(
      'Unsupported score kind(s):', paste(unsupported, collapse=', '),
      '; supported values are',
      paste(factorial_score_specs$score_kind, collapse=', ')
    ))
  }
  score_kinds
}


parse_factorial_score_kinds <- function(value) {
  if (length(value) != 1L || is.na(value) || trimws(value) == '') {
    stop('score-kinds must be a non-empty comma-separated list')
  }
  if (grepl(',\\s*$', value)) {
    stop('score-kinds must not contain empty values')
  }
  validate_factorial_score_kinds(strsplit(value, ',', fixed=TRUE)[[1]])
}


select_factorial_score_specs <- function(score_kinds) {
  score_kinds <- validate_factorial_score_kinds(score_kinds)
  factorial_score_specs[
    match(score_kinds, factorial_score_specs$score_kind), , drop=FALSE
  ]
}


discover_factorial_predictors <- function(
    df, include_embedding_layer,
    score_kinds=factorial_default_score_kinds) {
  score_specs <- select_factorial_score_specs(score_kinds)
  families <- vector('list', nrow(score_specs))
  for (index in seq_len(nrow(score_specs))) {
    spec <- score_specs[index, , drop=FALSE]
    layers <- validate_factorial_layer_family(
      discover_predictors(df, spec$predictor_prefix[[1]]),
      include_embedding_layer
    )
    layers$score_kind <- spec$score_kind[[1]]
    layers$predictor_prefix <- spec$predictor_prefix[[1]]
    families[[index]] <- layers
  }

  reference_layers <- families[[1]]$context
  same_layer_range <- vapply(
    families,
    function(family) identical(family$context, reference_layers),
    logical(1)
  )
  if (!all(same_layer_range)) {
    stop('Requested score-kind predictors must have the same layer range')
  }

  predictors <- do.call(rbind, families)
  rownames(predictors) <- NULL
  predictors[, c(
    'score_kind', 'predictor_prefix', 'context', 'variable'
  ), drop=FALSE]
}


resolve_column_alias <- function(df, canonical, alternatives=character()) {
  available <- c(canonical, alternatives)
  available <- available[available %in% colnames(df)]
  if (length(available) == 0L) {
    stop(paste(
      'Paper-exact mode requires column',
      paste(c(canonical, alternatives), collapse=' or ')
    ))
  }
  available[[1]]
}


coerce_numeric <- function(values) {
  if (is.numeric(values)) {
    return(values)
  }
  suppressWarnings(as.numeric(values))
}


parse_is_first <- function(values) {
  if (is.logical(values)) {
    return(values)
  }
  normalized <- tolower(trimws(as.character(values)))
  result <- rep(NA, length(normalized))
  result[normalized %in% c('true', 't', '1')] <- TRUE
  result[normalized %in% c('false', 'f', '0')] <- FALSE
  result
}


sort_and_validate_sentence_rows <- function(df) {
  text_column <- resolve_column_alias(df, 'text_id', 'article')
  sentence_column <- resolve_column_alias(df, 'sentence_id', 'sent_id')
  position_column <- resolve_column_alias(
    df, 'sentence_word_id', 'tokenN_in_sent'
  )

  for (column in c(text_column, sentence_column)) {
    values <- as.character(df[[column]])
    if (any(is.na(values)) || any(trimws(values) == '')) {
      stop(paste('Sentence grouping column is missing or empty:', column))
    }
  }

  position <- coerce_numeric(df[[position_column]])
  if (any(!is.finite(position)) || any(position != floor(position)) ||
      any(position < 0)) {
    stop('Sentence positions must be finite non-negative integers')
  }
  if (all(c('sentence_word_id', 'tokenN_in_sent') %in% colnames(df))) {
    alternate <- coerce_numeric(df$tokenN_in_sent)
    if (any(!is.finite(alternate)) || any(alternate != position)) {
      stop('sentence_word_id and tokenN_in_sent disagree')
    }
  }

  if ('is_first' %in% colnames(df)) {
    supplied_is_first <- parse_is_first(df$is_first)
    if (any(is.na(supplied_is_first)) ||
        any(supplied_is_first != (position == 0))) {
      stop('is_first disagrees with the manifest-derived sentence position')
    }
  }

  df$.factorial_sentence_position <- as.integer(position)
  ordering <- order(
    df[[text_column]], df[[sentence_column]],
    df$.factorial_sentence_position
  )
  df <- df[ordering, , drop=FALSE]
  rownames(df) <- NULL

  group_key <- interaction(
    df[[text_column]], df[[sentence_column]], drop=TRUE, lex.order=TRUE
  )
  positions_by_sentence <- split(df$.factorial_sentence_position, group_key)
  complete_sequences <- vapply(
    positions_by_sentence,
    function(positions) identical(positions, seq.int(0L, length(positions) - 1L)),
    logical(1)
  )
  if (!all(complete_sequences)) {
    stop('Each sentence must contain exactly one ordered position range 0..N-1')
  }
  df
}


mean_padded_lags <- function(values, sentence_position) {
  if (any(!is.finite(values))) {
    stop('Paper-exact mean padding requires complete finite source values')
  }
  n <- length(values)
  value_mean <- mean(values)
  previous_1 <- c(value_mean, values[seq_len(max(n - 1L, 0L))])
  previous_2 <- if (n == 1L) {
    value_mean
  } else {
    c(value_mean, value_mean, values[seq_len(max(n - 2L, 0L))])
  }
  previous_1[sentence_position == 0L] <- value_mean
  previous_2[sentence_position <= 1L] <- value_mean
  list(previous_1=previous_1, previous_2=previous_2, mean=value_mean)
}


prepare_paper_exact_data <- function(df, layer_variables) {
  required <- c('time', 'length', 'log_gmean_freq', layer_variables)
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0L) {
    stop(paste('Input is missing required paper-exact columns:',
               paste(missing, collapse=', ')))
  }
  df <- sort_and_validate_sentence_rows(df)
  for (column in required) {
    df[[column]] <- coerce_numeric(df[[column]])
  }
  padding_sources <- c('length', 'log_gmean_freq', layer_variables)
  for (column in padding_sources) {
    if (any(!is.finite(df[[column]]))) {
      stop(paste('Paper-exact padding source is not complete and finite:', column))
    }
  }

  length_lags <- mean_padded_lags(
    df$length, df$.factorial_sentence_position
  )
  df$length_prev_1 <- length_lags$previous_1
  df$length_prev_2 <- length_lags$previous_2
  frequency_lags <- mean_padded_lags(
    df$log_gmean_freq, df$.factorial_sentence_position
  )
  df$log_gmean_freq_prev_1 <- frequency_lags$previous_1
  df$log_gmean_freq_prev_2 <- frequency_lags$previous_2

  predictor_means <- numeric(length(layer_variables))
  names(predictor_means) <- layer_variables
  for (variable in layer_variables) {
    lags <- mean_padded_lags(
      df[[variable]], df$.factorial_sentence_position
    )
    df[[paste0('prev_', variable)]] <- lags$previous_1
    df[[paste0('prev2_', variable)]] <- lags$previous_2
    predictor_means[[variable]] <- lags$mean
  }

  fit_columns <- unique(c(
    'time', paper_exact_control_variables,
    unlist(lapply(
      layer_variables,
      function(variable) c(
        variable, paste0('prev_', variable), paste0('prev2_', variable)
      )
    ), use.names=FALSE)
  ))
  fit_data <- df[, fit_columns, drop=FALSE]
  complete <- complete.cases(fit_data)
  finite <- apply(fit_data, 1, function(row) all(is.finite(row)))
  sentence_initial <- df$.factorial_sentence_position == 0L
  positive_time <- is.finite(df$time) & df$time > 0
  keep <- !sentence_initial & positive_time & complete & finite

  list(
    data=df[keep, , drop=FALSE],
    ordered_data=df,
    predictor_means=predictor_means,
    sentence_initial_rows=sum(sentence_initial),
    nonpositive_time_rows=sum(!positive_time & !sentence_initial),
    incomplete_rows=sum(!complete | !finite)
  )
}


make_paper_exact_reduced_formula <- function(variable) {
  as.formula(paste(
    'time ~',
    paste(c(
      paste0('prev_', variable), paste0('prev2_', variable),
      paper_exact_control_variables
    ), collapse=' + ')
  ))
}


make_paper_exact_full_formula <- function(variable) {
  as.formula(paste(
    'time ~',
    paste(c(
      paste0('prev_', variable), paste0('prev2_', variable),
      paper_exact_control_variables, variable
    ), collapse=' + ')
  ))
}


evaluate_paper_exact_layer <- function(layer, variable, df, model,
                                       min_layer, max_layer, input_rows) {
  reduced_formula <- make_paper_exact_reduced_formula(variable)
  full_formula <- make_paper_exact_full_formula(variable)
  reduced <- fit_in_sample_model(reduced_formula, df)
  full <- fit_in_sample_model(full_formula, df)
  if (nobs(reduced$model) != nobs(full$model) ||
      nobs(full$model) != nrow(df)) {
    stop(paste('Reduced and full models used different rows for layer', layer))
  }

  coefficient_table <- summary(full$model)$coefficients
  if (!variable %in% rownames(coefficient_table)) {
    stop(paste('Full model has no current-word L coefficient for layer', layer))
  }
  delta_ll <- full$log_likelihood - reduced$log_likelihood
  if (delta_ll < -1e-8) {
    stop(paste('Nested full model reduced likelihood for layer', layer))
  }
  delta_ll <- max(delta_ll, 0)
  current_l <- coefficient_table[variable, ]
  relative_depth_block <- if (max_layer == min_layer) {
    NA_real_
  } else {
    (layer - min_layer) / (max_layer - min_layer)
  }

  data.frame(
    analysis='kuribayashi_paper_exact_L_nesting',
    model=model,
    layer=layer,
    min_layer=min_layer,
    max_layer=max_layer,
    relative_depth_block=relative_depth_block,
    layer_fraction=layer / max_layer,
    input_rows=input_rows,
    analysis_rows=nrow(df),
    excluded_rows=input_rows - nrow(df),
    ll_reduced=reduced$log_likelihood,
    ll_full=full$log_likelihood,
    delta_ll=delta_ll,
    delta_ll_per_input_word=delta_ll / input_rows,
    delta_ll_per_analysis_word=delta_ll / nrow(df),
    ppp_x1000=1000 * delta_ll / input_rows,
    current_l_estimate=unname(current_l[['Estimate']]),
    current_l_std_error=unname(current_l[['Std. Error']]),
    current_l_t=unname(current_l[['t value']]),
    current_l_p=unname(current_l[['Pr(>|t|)']]),
    is_final_layer=layer == max_layer,
    stringsAsFactors=FALSE
  )
}


append_factorial_metadata <- function(row, metadata, analysis_mode,
                                      response_column,
                                      score_kind, predictor_prefix) {
  row$analysis_mode <- analysis_mode
  row$response_column <- response_column
  for (column in factorial_metadata_columns) {
    row[[column]] <- metadata[[column]]
  }
  row$analysis_lag_boundary <- if (analysis_mode == 'paper-exact') {
    'sentence'
  } else {
    metadata$lag_boundary
  }
  row$analysis_lag_padding <- if (analysis_mode == 'paper-exact') {
    'global-mean'
  } else {
    metadata$lag_padding
  }
  row$score_kind <- score_kind
  row$predictor_prefix <- predictor_prefix
  row$is_embedding_layer <- row$layer == 0L
  leading <- c(
    'analysis', 'analysis_mode', 'response_column',
    factorial_metadata_columns,
    'analysis_lag_boundary', 'analysis_lag_padding',
    'score_kind', 'predictor_prefix'
  )
  row[, c(leading, setdiff(colnames(row), leading)), drop=FALSE]
}


select_factorial_best_layers <- function(
    layer_results, score_kinds=factorial_default_score_kinds) {
  score_kinds <- validate_factorial_score_kinds(score_kinds)
  layer_results <- layer_results[
    layer_results$score_kind %in% score_kinds, , drop=FALSE
  ]
  score_order <- match(layer_results$score_kind, score_kinds)
  layer_results <- layer_results[
    order(score_order, layer_results$layer), , drop=FALSE
  ]
  rownames(layer_results) <- NULL
  layer_results$is_best_layer <- FALSE
  best_rows <- vector('list', length(score_kinds))
  for (index in seq_along(score_kinds)) {
    score_kind <- score_kinds[[index]]
    candidates <- which(layer_results$score_kind == score_kind)
    if (length(candidates) == 0L) {
      stop(paste('No evaluated layers for score kind', score_kind))
    }
    # which.max preserves the source analysis's first-layer tie behavior.
    best_index <- candidates[[which.max(layer_results$delta_ll[candidates])]]
    layer_results$is_best_layer[[best_index]] <- TRUE
    best_rows[[index]] <- layer_results[best_index, , drop=FALSE]
  }
  list(layers=layer_results, best=do.call(rbind, best_rows))
}


factorial_score_kind_scope <- function(score_kinds) {
  score_kinds <- validate_factorial_score_kinds(score_kinds)
  if (length(score_kinds) == 1L) {
    return(paste('the', score_kinds[[1]], 'score kind'))
  }
  paste0(
    'the requested score kinds (', paste(score_kinds, collapse=', '), ')'
  )
}


make_factorial_summary <- function(metadata, analysis_mode, response_column,
                                   predictors,
                                   best_layers, input_rows, analysis_rows,
                                   preparation,
                                   score_kinds=factorial_default_score_kinds) {
  score_kinds <- validate_factorial_score_kinds(score_kinds)
  best_by_kind <- split(best_layers, best_layers$score_kind)
  score_scope <- factorial_score_kind_scope(score_kinds)
  if (analysis_mode == 'paper-exact') {
    reduced_template <- paste(
      'time ~ prev_L + prev2_L +', paper_exact_control_formula
    )
    full_template <- paste(reduced_template, '+ L')
    analysis <- 'kuribayashi_paper_exact_L_nesting'
    controls <- paper_exact_control_formula
    sample_policy <- paste(
      'sort by text/sentence/token; mean-pad predictor and control lags at',
      'sentence boundaries; exclude sentence-initial and nonpositive-time',
      'targets; share one complete finite sample across', score_scope,
      'and all layers'
    )
    lag_policy <- 'global-mean padding at sentence boundaries, matching source code'
  } else {
    reduced_template <- paste(
      'time ~', lexical_control_formula, '+ prev_L + prev2_L'
    )
    full_template <- paste(reduced_template, '+ L')
    analysis <- 'kuribayashi_L_nesting'
    controls <- lexical_control_formula
    sample_policy <- paste(
      'one shared complete finite sample across', score_scope,
      'and all layers, including current through t-3 L columns'
    )
    lag_policy <- 'consume the merged input project spillover columns'
  }

  keys <- c(
    'analysis', 'analysis_mode', 'response_column', 'design',
    factorial_metadata_columns,
    'analysis_lag_boundary', 'analysis_lag_padding',
    'score_kinds', 'input_rows', 'complete_case_rows', 'excluded_rows',
    'internal_layers', 'min_layer', 'max_layer'
  )
  values <- c(
    analysis, analysis_mode, response_column,
    paste0(paste(score_kinds, collapse='_by_'), '_factorial'),
    unlist(metadata[factorial_metadata_columns], use.names=FALSE),
    if (analysis_mode == 'paper-exact') 'sentence' else metadata$lag_boundary,
    if (analysis_mode == 'paper-exact') 'global-mean' else metadata$lag_padding,
    paste(score_kinds, collapse=','),
    input_rows, analysis_rows, input_rows - analysis_rows,
    paste(unique(predictors$context), collapse=','),
    min(predictors$context), max(predictors$context)
  )

  for (score_kind in score_kinds) {
    best <- best_by_kind[[score_kind]]
    keys <- c(
      keys, paste0('best_layer_', score_kind),
      paste0('best_delta_ll_', score_kind),
      paste0('best_ppp_x1000_', score_kind)
    )
    values <- c(
      values, best$layer[[1]], best$delta_ll[[1]], best$ppp_x1000[[1]]
    )
  }

  keys <- c(
    keys, 'response', 'evaluation', 'ranking_metric', 'ppp_definition',
    'reduced_formula_template', 'full_formula_template', 'controls',
    'sample_policy', 'lag_boundary_policy', 'target_filter',
    'best_layer_tie_policy', 'relative_depth_block', 'layer_fraction'
  )
  values <- c(
    values, paste('raw reading time (ms) from column', response_column),
    'in-sample Gaussian OLS log likelihood', 'delta_ll',
    '1000 * (ll_full - ll_reduced) / all input words',
    reduced_template, full_template, controls, sample_policy, lag_policy,
    if (analysis_mode == 'paper-exact') {
      'time > 0, sentence position > 0, and is_first=false by construction'
    } else {
      'complete and finite canonical project sample only'
    },
    'lowest layer wins an exact delta_ll tie within each score kind',
    '(layer - min_layer) / (max_layer - min_layer)', 'layer / D'
  )

  if (analysis_mode == 'paper-exact') {
    keys <- c(
      keys, 'sentence_initial_rows_excluded',
      'nonpositive_time_rows_excluded'
    )
    values <- c(
      values, preparation$sentence_initial_rows,
      preparation$nonpositive_time_rows
    )
  }
  data.frame(key=keys, value=as.character(values), stringsAsFactors=FALSE)
}


canonical_path <- function(fname) {
  file.path(
    normalizePath(dirname(fname), winslash='/', mustWork=FALSE),
    basename(fname)
  )
}


validate_factorial_paths <- function(input_fname, output_fnames) {
  paths <- vapply(c(input_fname, output_fnames), canonical_path, character(1))
  if (anyDuplicated(paths)) {
    stop('Input and all three output paths must be distinct')
  }
  invisible(paths)
}


run_factorial_kuribayashi_evaluation <- function(
    input_fname, layer_output_fname, best_output_fname, summary_fname,
    analysis_mode='paper-exact', response_column='time',
    score_kinds=factorial_default_score_kinds) {
  if (!analysis_mode %in% factorial_analysis_modes) {
    stop(paste(
      'analysis_mode must be one of', paste(factorial_analysis_modes, collapse=', ')
    ))
  }
  score_kinds <- validate_factorial_score_kinds(score_kinds)
  validate_factorial_paths(
    input_fname, c(layer_output_fname, best_output_fname, summary_fname)
  )
  df_raw <- read.csv(input_fname, header=TRUE, sep='\t', check.names=FALSE)
  input_rows <- nrow(df_raw)
  if (input_rows == 0L) {
    stop('Factorial input has no rows')
  }
  if (length(response_column) != 1L || is.na(response_column) ||
      trimws(response_column) == '') {
    stop('response_column must be one non-empty column name')
  }
  if (!response_column %in% colnames(df_raw)) {
    stop(paste('Input is missing response column:', response_column))
  }
  # Both validated model implementations use the internal response name `time`.
  # Alias the selected input response once instead of changing either formula.
  df_raw$time <- df_raw[[response_column]]
  metadata <- read_factorial_metadata(df_raw)
  predictors <- discover_factorial_predictors(
    df_raw, metadata$include_embedding_layer, score_kinds
  )

  if (analysis_mode == 'paper-exact') {
    preparation <- prepare_paper_exact_data(df_raw, predictors$variable)
    df <- preparation$data
  } else {
    df <- prepare_kuribayashi_replication_data(df_raw, predictors$variable)
    preparation <- list()
  }
  analysis_rows <- nrow(df)
  if (analysis_rows <= 16L) {
    stop('Factorial replication sample is too small for the full model')
  }

  rows <- vector('list', nrow(predictors))
  min_layer <- min(predictors$context)
  max_layer <- max(predictors$context)
  for (index in seq_len(nrow(predictors))) {
    predictor <- predictors[index, , drop=FALSE]
    if (analysis_mode == 'paper-exact') {
      row <- evaluate_paper_exact_layer(
        layer=predictor$context[[1]], variable=predictor$variable[[1]],
        df=df, model=metadata$model, min_layer=min_layer,
        max_layer=max_layer, input_rows=input_rows
      )
    } else {
      row <- evaluate_kuribayashi_layer(
        layer=predictor$context[[1]], variable=predictor$variable[[1]],
        df=df, model=metadata$model, min_layer=min_layer,
        max_layer=max_layer, input_rows=input_rows
      )
    }
    rows[[index]] <- append_factorial_metadata(
      row, metadata, analysis_mode, response_column,
      predictor$score_kind[[1]],
      predictor$predictor_prefix[[1]]
    )
  }

  layer_results <- do.call(rbind, rows)
  score_order <- match(layer_results$score_kind, score_kinds)
  layer_results <- layer_results[
    order(score_order, layer_results$layer), , drop=FALSE
  ]
  rownames(layer_results) <- NULL
  selected <- select_factorial_best_layers(layer_results, score_kinds)
  layer_results <- selected$layers
  best_layers <- selected$best
  rownames(best_layers) <- NULL
  summary <- make_factorial_summary(
    metadata, analysis_mode, response_column, predictors, best_layers,
    input_rows, analysis_rows, preparation, score_kinds
  )

  # All computation/validation precedes publication.  The shared writer stages
  # each TSV beside its destination and atomically renames it into place.
  write_tsv_atomic(layer_results, layer_output_fname)
  write_tsv_atomic(best_layers, best_output_fname)
  write_tsv_atomic(summary, summary_fname)
  invisible(list(
    layers=layer_results, best_layers=best_layers, summary=summary,
    analysis_data=df, metadata=metadata, predictors=predictors,
    preparation=preparation, response_column=response_column,
    score_kinds=score_kinds
  ))
}


parse_factorial_cli_args <- function(args) {
  usage <- paste(
    'Usage: rt_vs_internal_layer_factorial_kuribayashi.R',
    'INPUT LAYER_RESULTS BEST_LAYERS SUMMARY',
    '[--analysis-mode paper-exact|project-bridge]',
    '[--response-column COLUMN]',
    '[--score-kinds corrected,buggy]'
  )
  if (length(args) < 4L) {
    stop(usage)
  }
  paths <- args[1:4]
  options <- args[-seq_len(4L)]
  mode <- 'paper-exact'
  response_column <- 'time'
  score_kinds <- factorial_default_score_kinds
  seen <- character()
  index <- 1L
  while (index <= length(options)) {
    option <- options[[index]]
    if (option %in% c(
        '--analysis-mode', '--response-column', '--score-kinds')) {
      if (index == length(options)) {
        stop(usage)
      }
      key <- sub('^--', '', option)
      value <- options[[index + 1L]]
      index <- index + 2L
    } else if (grepl(
        '^--(analysis-mode|response-column|score-kinds)=', option)) {
      key <- sub('^--([^=]+)=.*$', '\\1', option)
      value <- sub('^--[^=]+=', '', option)
      index <- index + 1L
    } else {
      stop(usage)
    }
    if (key %in% seen) {
      stop(paste('Duplicate command-line option:', key))
    }
    seen <- c(seen, key)
    if (key == 'analysis-mode') {
      mode <- value
    } else if (key == 'response-column') {
      response_column <- value
    } else {
      score_kinds <- parse_factorial_score_kinds(value)
    }
  }
  if (!mode %in% factorial_analysis_modes) {
    stop(usage)
  }
  if (is.na(response_column) || trimws(response_column) == '') {
    stop(usage)
  }
  list(
    paths=paths, analysis_mode=mode, response_column=response_column,
    score_kinds=score_kinds
  )
}


main <- function(args=commandArgs(trailingOnly=TRUE)) {
  parsed <- parse_factorial_cli_args(args)
  run_factorial_kuribayashi_evaluation(
    parsed$paths[[1]], parsed$paths[[2]], parsed$paths[[3]],
    parsed$paths[[4]], analysis_mode=parsed$analysis_mode,
    response_column=parsed$response_column,
    score_kinds=parsed$score_kinds
  )
}


if (sys.nframe() == 0) {
  main()
}
