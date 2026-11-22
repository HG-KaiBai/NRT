suppressPackageStartupMessages({
  library(partykit)
  library(icenReg)
  library(survival)
})

parse_numeric_columns <- function(data) {
  columns <- c("L", "R", grep("^x_", names(data), value = TRUE))
  for (column in intersect(columns, names(data))) {
    values <- trimws(as.character(data[[column]]))
    values[tolower(values) %in% c("inf", "+inf", "infinity", "+infinity")] <- "Inf"
    data[[column]] <- as.numeric(values)
  }
  data
}

interval_logrank_trafo <- function(y, x, start = NULL, weights, offset,
                                   estfun = TRUE, object = FALSE, ...) {
  if (is.null(weights)) weights <- rep(1, NROW(y))
  active <- weights > 0
  response <- y[active, , drop = FALSE]
  if (!(survival::is.Surv(response) && identical(attr(response, "type"), "interval"))) {
    stop("Response must be an interval Surv object.")
  }

  curve <- icenReg::ic_np(response[, 1:2])
  survival_left <- 1 - icenReg::getFitEsts(curve, q = response[, 1])
  survival_right <- 1 - icenReg::getFitEsts(curve, q = response[, 2])
  slog_left <- ifelse(survival_left <= 0, 0, survival_left * log(survival_left))
  slog_right <- ifelse(survival_right <= 0, 0, survival_right * log(survival_right))
  denominator <- survival_left - survival_right
  score <- (slog_left - slog_right) / denominator

  # The derivative is the continuous limit when the NPMLE is flat at both ends.
  flat <- !is.finite(score) | abs(denominator) < sqrt(.Machine$double.eps)
  score[flat] <- ifelse(
    survival_left[flat] > 0,
    1 + log(survival_left[flat]),
    0
  )
  result <- rep(0, length(weights))
  result[active] <- score
  list(estfun = matrix(as.double(result), ncol = 1), converged = TRUE)
}

fit_node_curves <- function(data, terminal_nodes) {
  node_ids <- sort(unique(terminal_nodes))
  curves <- setNames(vector("list", length(node_ids)), as.character(node_ids))
  for (node_id in node_ids) {
    bounds <- as.matrix(data[terminal_nodes == node_id, c("L", "R"), drop = FALSE])
    curves[[as.character(node_id)]] <- tryCatch(
      icenReg::ic_np(bounds),
      error = function(error) NULL
    )
  }
  curves
}

evaluate_node_survival <- function(curve, grid) {
  if (is.null(curve)) return(rep(1, length(grid)))
  cdf <- tryCatch(
    as.numeric(icenReg::getFitEsts(curve, q = grid)),
    error = function(error) rep(0, length(grid))
  )
  survival <- 1 - pmin(pmax(cdf, 0), 1)
  if (length(grid) > 0L && grid[[1L]] == 0) survival[[1L]] <- 1
  cummin(survival)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) stop("The first argument must be 'fit' or 'predict'.")
mode <- arguments[[1L]]

if (mode == "fit") {
  if (length(arguments) != 8L) {
    stop("fit expects: input.csv model.rds mincriterion minsplit minbucket maxdepth seed")
  }
  input_path <- arguments[[2L]]
  model_path <- arguments[[3L]]
  mincriterion <- as.numeric(arguments[[4L]])
  minsplit <- as.integer(arguments[[5L]])
  minbucket <- as.integer(arguments[[6L]])
  maxdepth <- as.numeric(arguments[[7L]])
  seed <- as.integer(arguments[[8L]])
  set.seed(seed)

  original_data <- parse_numeric_columns(read.csv(input_path, check.names = FALSE))
  feature_names <- sort(grep("^x_", names(original_data), value = TRUE))
  if (length(feature_names) < 1L) stop("No x_ feature columns were supplied.")
  fit_data <- original_data

  # This matches the exact-event and right-tail handling in LTRCtrees::ICtree.
  exact <- fit_data$L == fit_data$R
  if (any(exact)) {
    distinct_times <- sort(unique(c(fit_data$L, fit_data$R[is.finite(fit_data$R)])))
    positive_gaps <- diff(distinct_times)
    positive_gaps <- positive_gaps[positive_gaps > 0]
    epsilon <- if (length(positive_gaps) > 0L) {
      min(positive_gaps) / 20
    } else {
      max(1, max(abs(distinct_times))) * 1e-7
    }
    fit_data$R[exact] <- fit_data$R[exact] + epsilon
  }
  if (any(is.infinite(fit_data$R))) {
    finite_right <- fit_data$R[is.finite(fit_data$R)]
    if (length(finite_right) == 0L) stop("All right endpoints are infinite.")
    fit_data$R[is.infinite(fit_data$R)] <- max(finite_right) * 100
  }

  formula <- as.formula(sprintf(
    "Surv(L, R, type = 'interval2') ~ %s", paste(feature_names, collapse = " + ")
  ))
  control <- partykit::ctree_control(
    mincriterion = mincriterion,
    minsplit = minsplit,
    minbucket = minbucket,
    maxdepth = maxdepth
  )
  fit <- partykit::ctree(
    formula = formula,
    data = fit_data,
    ytrafo = interval_logrank_trafo,
    control = control
  )
  terminal_nodes <- predict(fit, type = "node")
  node_curves <- fit_node_curves(original_data, terminal_nodes)
  saveRDS(
    list(
      fit = fit,
      feature_names = feature_names,
      node_curves = node_curves,
      control = list(
        mincriterion = mincriterion,
        minsplit = minsplit,
        minbucket = minbucket,
        maxdepth = maxdepth
      )
    ),
    model_path
  )
} else if (mode == "predict") {
  if (length(arguments) != 5L) {
    stop("predict expects: model.rds input.csv grid.csv output.csv")
  }
  artifact <- readRDS(arguments[[2L]])
  data <- parse_numeric_columns(read.csv(arguments[[3L]], check.names = FALSE))
  grid <- scan(arguments[[4L]], what = numeric(), quiet = TRUE, sep = ",")
  missing_features <- setdiff(artifact$feature_names, names(data))
  if (length(missing_features) > 0L) stop("Prediction data are missing fitted features.")
  terminal_nodes <- predict(
    artifact$fit,
    newdata = data[, artifact$feature_names, drop = FALSE],
    type = "node"
  )
  output <- matrix(NA_real_, nrow = length(terminal_nodes), ncol = length(grid))
  for (index in seq_along(terminal_nodes)) {
    curve <- artifact$node_curves[[as.character(terminal_nodes[[index]])]]
    output[index, ] <- evaluate_node_survival(curve, grid)
  }
  write.table(
    output, file = arguments[[5L]], sep = ",", row.names = FALSE,
    col.names = FALSE, quote = FALSE
  )
} else {
  stop(sprintf("Unknown mode: %s", mode))
}
