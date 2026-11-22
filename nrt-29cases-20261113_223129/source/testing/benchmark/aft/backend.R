suppressPackageStartupMessages(library(survival))

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
if (length(arguments) < 1L) stop("The first argument must be 'fit' or 'predict'.")
mode <- arguments[[1L]]

if (mode == "fit") {
  if (length(arguments) != 5L) {
    stop("fit expects: input.csv model.rds distribution max_iter")
  }
  data <- parse_numeric_columns(read.csv(arguments[[2L]], check.names = FALSE))
  feature_names <- grep("^x_", names(data), value = TRUE)
  if (length(feature_names) < 1L) stop("No x_ feature columns were supplied.")
  distribution <- arguments[[4L]]
  if (!distribution %in% c("exponential", "weibull", "lognormal", "loglogistic")) {
    stop(sprintf("Unsupported AFT distribution: %s", distribution))
  }
  max_iter <- as.integer(arguments[[5L]])

  data$L_surv <- data$L
  data$R_surv <- data$R
  left_censored <- data$L == 0 & is.finite(data$R) & data$R > data$L
  data$L_surv[left_censored] <- NA_real_
  formula <- as.formula(sprintf(
    "Surv(L_surv, R_surv, type = 'interval2') ~ %s",
    paste(feature_names, collapse = " + ")
  ))
  fit <- survival::survreg(
    formula, data = data, dist = distribution,
    control = survival::survreg.control(maxiter = max_iter)
  )
  artifact <- list(
    fit = fit, distribution = distribution, feature_names = feature_names
  )
  saveRDS(artifact, arguments[[3L]])
} else if (mode == "predict") {
  if (length(arguments) != 5L) {
    stop("predict expects: model.rds input.csv grid.csv output.csv")
  }
  artifact <- readRDS(arguments[[2L]])
  data <- parse_numeric_columns(read.csv(arguments[[3L]], check.names = FALSE))
  grid <- scan(arguments[[4L]], what = numeric(), quiet = TRUE, sep = ",")
  missing_features <- setdiff(artifact$feature_names, names(data))
  if (length(missing_features) > 0L) stop("Prediction data are missing fitted features.")
  newdata <- data[, artifact$feature_names, drop = FALSE]

  linear_predictor <- as.numeric(predict(artifact$fit, newdata = newdata, type = "lp"))
  output <- matrix(1, nrow = nrow(newdata), ncol = length(grid))
  positive <- which(grid > 0)
  for (column in positive) {
    if (artifact$distribution == "exponential") {
      output[, column] <- pexp(
        grid[[column]], rate = exp(-linear_predictor), lower.tail = FALSE
      )
    } else if (artifact$distribution == "weibull") {
      output[, column] <- pweibull(
        grid[[column]], shape = 1 / artifact$fit$scale,
        scale = exp(linear_predictor), lower.tail = FALSE
      )
    } else if (artifact$distribution == "lognormal") {
      output[, column] <- plnorm(
        grid[[column]], meanlog = linear_predictor,
        sdlog = artifact$fit$scale, lower.tail = FALSE
      )
    } else if (artifact$distribution == "loglogistic") {
      output[, column] <- plogis(
        (log(grid[[column]]) - linear_predictor) / artifact$fit$scale,
        lower.tail = FALSE
      )
    } else {
      stop(sprintf("Unsupported AFT distribution: %s", artifact$distribution))
    }
  }
  output <- pmin(pmax(output, 0), 1)
  for (row in seq_len(nrow(output))) output[row, ] <- cummin(output[row, ])
  write.table(
    output, file = arguments[[5L]], sep = ",", row.names = FALSE,
    col.names = FALSE, quote = FALSE
  )
} else {
  stop(sprintf("Unknown mode: %s", mode))
}
