suppressPackageStartupMessages({
  library(ICcforest)
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

turnbull_components <- function(prediction) {
  intervals <- prediction$T_bull_Intervals
  if (is.null(intervals)) intervals <- prediction$Tbull_ints
  probabilities <- prediction$p_hat
  if (is.null(intervals) || is.null(probabilities)) return(NULL)
  intervals <- as.matrix(intervals)
  if (nrow(intervals) == 2L && ncol(intervals) != 2L) {
    lower <- as.numeric(intervals[1L, ])
    upper <- as.numeric(intervals[2L, ])
  } else if (ncol(intervals) == 2L) {
    lower <- as.numeric(intervals[, 1L])
    upper <- as.numeric(intervals[, 2L])
  } else {
    return(NULL)
  }
  probabilities <- as.numeric(probabilities)
  if (length(lower) != length(probabilities)) return(NULL)
  keep <- is.finite(lower) & is.finite(upper) & is.finite(probabilities) & probabilities > 0
  if (!any(keep)) return(NULL)
  ordering <- order(lower[keep], upper[keep])
  list(
    lower = lower[keep][ordering],
    upper = upper[keep][ordering],
    probabilities = probabilities[keep][ordering]
  )
}

survival_on_grid <- function(prediction, grid) {
  components <- turnbull_components(prediction)
  if (is.null(components)) {
    cdf <- as.numeric(getFitEsts(prediction, q = grid))
    return(1 - pmin(pmax(cdf, 0), 1))
  }
  cdf <- rep(0, length(grid))
  for (index in seq_along(components$probabilities)) {
    width <- components$upper[[index]] - components$lower[[index]]
    if (width <= 0) {
      fraction <- as.numeric(grid >= components$upper[[index]])
    } else {
      fraction <- pmin(pmax(
        (grid - components$lower[[index]]) / width, 0
      ), 1)
    }
    cdf <- cdf + components$probabilities[[index]] * fraction
  }
  1 - pmin(pmax(cdf, 0), 1)
}

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) stop("The first argument must be 'fit' or 'predict'.")
mode <- arguments[[1L]]

if (mode == "fit") {
  if (length(arguments) != 9L) {
    stop("fit expects: input.csv model.rds ntree tune tune_ntree step_factor inf_replacement seed")
  }
  input_path <- arguments[[2L]]
  model_path <- arguments[[3L]]
  ntree <- as.integer(arguments[[4L]])
  tune <- as.logical(arguments[[5L]])
  tune_ntree <- as.integer(arguments[[6L]])
  step_factor <- as.numeric(arguments[[7L]])
  inf_replacement <- as.numeric(arguments[[8L]])
  set.seed(as.integer(arguments[[9L]]))

  data <- parse_numeric_columns(read.csv(input_path, check.names = FALSE))
  feature_names <- sort(grep("^x_", names(data), value = TRUE))
  if (length(feature_names) < 1L) stop("No x_ feature columns were supplied.")
  data$R[is.infinite(data$R)] <- inf_replacement
  formula <- as.formula(sprintf(
    "Surv(L, R, type = 'interval2') ~ %s", paste(feature_names, collapse = " + ")
  ))
  control <- partykit::ctree_control(
    teststat = "quad", testtype = "Univ", mincriterion = 0,
    saveinfo = FALSE,
    minsplit = max(2L, floor(nrow(data) * 0.15)),
    minbucket = max(1L, floor(nrow(data) * 0.06))
  )
  if (tune) {
    mtry <- tuneICRF(
      formula = formula, data = data, ntreeTry = tune_ntree,
      stepFactor = step_factor, suppress = TRUE, trace = FALSE,
      plot = FALSE, doBest = FALSE, control = control
    )
    mtry <- max(1L, min(as.integer(mtry), length(feature_names)))
  } else {
    mtry <- max(1L, floor(sqrt(length(feature_names))))
  }
  fit <- ICcforest(
    formula = formula, data = data, ntree = ntree, mtry = mtry,
    suppress = TRUE, trace = FALSE, control = control
  )
  saveRDS(
    list(fit = fit, feature_names = feature_names, inf_replacement = inf_replacement),
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
  predictions <- predict(
    artifact$fit, newdata = data[, artifact$feature_names, drop = FALSE],
    type = "prob", suppress = TRUE
  )
  output <- matrix(NA_real_, nrow = length(predictions), ncol = length(grid))
  for (index in seq_along(predictions)) {
    output[index, ] <- survival_on_grid(predictions[[index]], grid)
  }
  output[, 1L] <- ifelse(grid[[1L]] == 0, 1, output[, 1L])
  output <- t(apply(output, 1L, cummin))
  write.table(
    output, file = arguments[[5L]], sep = ",", row.names = FALSE,
    col.names = FALSE, quote = FALSE
  )
} else {
  stop(sprintf("Unknown mode: %s", mode))
}
