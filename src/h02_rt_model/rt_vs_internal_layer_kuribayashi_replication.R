#!/usr/bin/env Rscript

# Formula-bridge analysis for Kuribayashi et al. (2025): compare a model with
# spillover L at t-1 and t-2 against the same model plus current-word L.
# The project's lexical M0 and canonical all-layer sample stay fixed so the
# changed L nesting is the only difference from the primary project design.

source('src/h02_rt_model/rt_vs_joint_pilot_surprisal.R')


kuribayashi_lag_terms <- function(variable) {
  c(paste0('prev_', variable), paste0('prev2_', variable))
}


make_kuribayashi_reduced_formula <- function(variable) {
  as.formula(paste(
    'time ~', lexical_control_formula,
    '+', paste(kuribayashi_lag_terms(variable), collapse=' + ')
  ))
}


make_kuribayashi_full_formula <- function(variable) {
  layer_terms <- c(kuribayashi_lag_terms(variable), variable)
  as.formula(paste(
    'time ~', lexical_control_formula,
    '+', paste(layer_terms, collapse=' + ')
  ))
}


validate_replication_layers <- function(layers) {
  if (any(layers$context < 1L)) {
    stop('Internal-layer indices must start at transformer layer 1')
  }
  expected <- seq.int(min(layers$context), max(layers$context))
  if (!identical(layers$context, expected) || layers$context[[1]] != 1L) {
    stop('Replication requires the complete consecutive layer range 1..D')
  }
  layers
}


canonical_layer_sample_columns <- function(layer_variables) {
  unique(c(
    'time', lexical_control_variables,
    unlist(lapply(layer_variables, spillover_terms), use.names=FALSE)
  ))
}


prepare_kuribayashi_replication_data <- function(df, layer_variables) {
  required <- canonical_layer_sample_columns(layer_variables)
  missing <- setdiff(required, colnames(df))
  if (length(missing) > 0) {
    stop(paste('Input is missing required columns:', paste(missing, collapse=', ')))
  }

  for (column in required) {
    if (!is.numeric(df[[column]])) {
      df[[column]] <- suppressWarnings(as.numeric(df[[column]]))
    }
  }
  numeric_data <- df[, required, drop=FALSE]
  complete <- complete.cases(numeric_data)
  finite <- apply(numeric_data, 1, function(row) all(is.finite(row)))
  df[complete & finite, , drop=FALSE]
}


fit_in_sample_model <- function(formula, df) {
  model <- lm(formula, data=df)
  design <- model.matrix(model)
  if (model$rank < ncol(design)) {
    stop(paste('Rank-deficient in-sample model:', deparse(formula)))
  }
  likelihood <- as.numeric(logLik(model))
  if (!is.finite(likelihood)) {
    stop(paste('Non-finite in-sample likelihood:', deparse(formula)))
  }
  list(model=model, log_likelihood=likelihood)
}


evaluate_kuribayashi_layer <- function(layer, variable, df, model,
                                        min_layer, max_layer,
                                        input_rows) {
  reduced <- fit_in_sample_model(
    make_kuribayashi_reduced_formula(variable), df
  )
  full <- fit_in_sample_model(make_kuribayashi_full_formula(variable), df)
  if (nobs(reduced$model) != nobs(full$model) || nobs(full$model) != nrow(df)) {
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
    analysis='kuribayashi_L_nesting',
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


run_kuribayashi_replication <- function(input_fname, layer_output_fname,
                                         best_output_fname, summary_fname,
                                         model='unspecified') {
  df_raw <- read.csv(input_fname, header=TRUE, sep='\t', check.names=FALSE)
  input_rows <- nrow(df_raw)
  layers <- validate_replication_layers(discover_predictors(
    df_raw, 'internal_layer_surprisal_layer_'
  ))
  df <- prepare_kuribayashi_replication_data(df_raw, layers$variable)
  analysis_rows <- nrow(df)
  if (analysis_rows <= 16L) {
    stop('Replication sample is too small for the full model')
  }

  rows <- vector('list', nrow(layers))
  for (index in seq_len(nrow(layers))) {
    rows[[index]] <- evaluate_kuribayashi_layer(
      layer=layers$context[[index]],
      variable=layers$variable[[index]],
      df=df,
      model=model,
      min_layer=min(layers$context),
      max_layer=max(layers$context),
      input_rows=input_rows
    )
  }
  layer_results <- do.call(rbind, rows)
  layer_results <- layer_results[order(layer_results$layer), , drop=FALSE]
  best_index <- which.max(layer_results$delta_ll)
  layer_results$is_best_layer <- seq_len(nrow(layer_results)) == best_index
  best_layer <- layer_results[best_index, , drop=FALSE]

  reduced_template <- paste(
    'time ~', lexical_control_formula, '+ prev_L + prev2_L'
  )
  full_template <- paste(reduced_template, '+ L')
  summary <- data.frame(
    key=c(
      'analysis', 'model', 'replication_scope', 'input_rows',
      'complete_case_rows', 'excluded_rows', 'internal_layers',
      'best_layer', 'best_delta_ll', 'best_ppp_x1000', 'response',
      'evaluation', 'ranking_metric', 'ppp_definition',
      'reduced_formula_template', 'full_formula_template',
      'lexical_baseline', 'sample_policy', 'spillover_boundary_policy',
      'layer_decoder', 'relative_depth_block', 'layer_fraction',
      'paper_difference_controls', 'paper_difference_predictor_context'
    ),
    value=c(
      'kuribayashi_L_nesting', model,
      paste(
        'formula bridge: replicate the paper nested L contrast while holding',
        'the project lexical M0 and canonical sample fixed'
      ),
      input_rows, analysis_rows, input_rows - analysis_rows,
      paste(layers$context, collapse=','),
      best_layer$layer[[1]], best_layer$delta_ll[[1]],
      best_layer$ppp_x1000[[1]], 'raw reading time (ms)',
      'in-sample Gaussian OLS log likelihood', 'delta_ll',
      '1000 * (ll_full - ll_reduced) / input_rows',
      reduced_template, full_template, lexical_control_formula,
      paste(
        'one shared complete finite sample across all layers, including',
        'current through t-3 L columns to match the primary analysis'
      ),
      'existing project spillovers are text-bounded, not sentence-bounded',
      'logit lens', '(layer - 1) / (D - 1)', 'layer / D',
      paste(
        'paper uses additive current-through-t-2 length and log_gmean_freq;',
        'this bridge retains the project current-through-t-3 length*freq M0'
      ),
      paste(
        'paper predictors reset/pad at sentence boundaries; project L uses',
        'long overlapping context and text-bounded spillovers'
      )
    ),
    stringsAsFactors=FALSE
  )

  write_tsv_atomic(layer_results, layer_output_fname)
  write_tsv_atomic(best_layer, best_output_fname)
  write_tsv_atomic(summary, summary_fname)
  invisible(list(
    layers=layer_results, best_layer=best_layer, summary=summary,
    analysis_data=df
  ))
}


main <- function(args=commandArgs(trailingOnly=TRUE)) {
  if (length(args) < 4) {
    stop(paste(
      'Usage: rt_vs_internal_layer_kuribayashi_replication.R',
      'INPUT LAYER_RESULTS BEST_LAYER SUMMARY [MODEL]'
    ))
  }
  model <- if (length(args) >= 5) args[[5]] else 'unspecified'
  run_kuribayashi_replication(
    args[[1]], args[[2]], args[[3]], args[[4]], model=model
  )
}


if (sys.nframe() == 0) {
  main()
}
