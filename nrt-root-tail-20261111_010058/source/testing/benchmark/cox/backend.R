suppressPackageStartupMessages({
  library(survival)
  library(icenReg)
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

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) < 1L) stop("The first argument must be 'fit', 'predict', or 'risk'.")
mode <- arguments[[1L]]

if (mode == "fit") {
  if (length(arguments) != 4L) {
    stop("fit expects: input.csv model.rds censor_type")
  }
  data <- parse_numeric_columns(read.csv(arguments[[2L]], check.names = FALSE))
  feature_names <- grep("^x_", names(data), value = TRUE)
  if (length(feature_names) < 1L) stop("No x_ feature columns were supplied.")
  formula_rhs <- paste(feature_names, collapse = " + ")
  censor_type <- arguments[[4L]]

  if (censor_type %in% c("exact", "right")) {
    data$time <- data$L
    data$event <- as.integer(is.finite(data$R))
    formula <- as.formula(sprintf("Surv(time, event) ~ %s", formula_rhs))
    fit <- survival::coxph(formula, data = data, ties = "efron", x = TRUE)
    artifact <- list(kind = "coxph", fit = fit, feature_names = feature_names)
  } else if (censor_type %in% c("interval", "current")) {
    formula <- as.formula(sprintf(
      "Surv(L, R, type = 'interval2') ~ %s", formula_rhs
    ))
    fit <- icenReg::ic_sp(formula, data = data, model = "ph")
    artifact <- list(kind = "ic_sp", fit = fit, feature_names = feature_names)
  } else {
    stop(sprintf("Unsupported censor type: %s", censor_type))
  }
  saveRDS(artifact, arguments[[3L]])
} else if (mode %in% c("predict", "risk")) {
  expected <- if (mode == "predict") 5L else 4L
  if (length(arguments) != expected) {
    stop(sprintf("%s received the wrong number of arguments", mode))
  }
  artifact <- readRDS(arguments[[2L]])
  data <- parse_numeric_columns(read.csv(arguments[[3L]], check.names = FALSE))
  missing_features <- setdiff(artifact$feature_names, names(data))
  if (length(missing_features) > 0L) stop("Prediction data are missing fitted features.")
  newdata <- data[, artifact$feature_names, drop = FALSE]

  if (mode == "risk") {
    if (artifact$kind == "coxph") {
      output <- as.numeric(predict(
        artifact$fit, newdata = newdata, type = "lp", reference = "zero"
      ))
    } else {
      output <- as.numeric(predict(artifact$fit, newdata = newdata, type = "lp"))
    }
    write.table(
      matrix(output, ncol = 1L), file = arguments[[4L]], sep = ",",
      row.names = FALSE, col.names = FALSE, quote = FALSE
    )
  } else {
    grid <- scan(arguments[[4L]], what = numeric(), quiet = TRUE, sep = ",")
    if (artifact$kind == "coxph") {
      baseline <- survival::basehaz(artifact$fit, centered = FALSE)
      positions <- findInterval(grid, baseline$time)
      baseline_hazard <- numeric(length(grid))
      positive <- positions > 0L
      baseline_hazard[positive] <- baseline$hazard[positions[positive]]
      linear_predictor <- as.numeric(predict(
        artifact$fit, newdata = newdata, type = "lp", reference = "zero"
      ))
      output <- exp(-outer(exp(linear_predictor), baseline_hazard))
    } else {
      output <- matrix(1, nrow = nrow(newdata), ncol = length(grid))
      curves <- icenReg::getSCurves(artifact$fit)
      support_end <- tail(curves$Tbull_ints[, "upper"], 1L)
      terminal_survival <- tail(curves$S_curves$baseline, 1L)
      positive <- which(grid > 0)
      for (column in positive) {
        # icenReg's interpolation can round below zero at the last endpoint
        # and return NA. A PH curve with zero baseline survival is exactly
        # zero at and beyond that endpoint, for every finite risk score.
        if (terminal_survival == 0 && grid[[column]] >= support_end) {
          output[, column] <- 0
        } else {
          output[, column] <- icenReg::getFitEsts(
            artifact$fit, newdata = newdata, q = grid[[column]]
          )
        }
      }
    }
    output <- pmin(pmax(output, 0), 1)
    for (row in seq_len(nrow(output))) output[row, ] <- cummin(output[row, ])
    write.table(
      output, file = arguments[[5L]], sep = ",", row.names = FALSE,
      col.names = FALSE, quote = FALSE
    )
  }
} else {
  stop(sprintf("Unknown mode: %s", mode))
}
