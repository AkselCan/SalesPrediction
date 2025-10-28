knitr::opts_chunk$set(echo = TRUE)

# Data Quality Report (Train/Test)

suppressPackageStartupMessages({
library(data.table)
library(dplyr)
library(purrr)
library(readr)
library(knitr)
})


if (!exists("train_raw")) {
train_raw <- data.table::fread("Train.csv", na.strings = c("", "NA", "null", "(not set)"))
}
if (!exists("test_raw")) {
test_raw  <- data.table::fread("Test.csv",  na.strings = c("", "NA", "null", "(not set)"))
}


suppressWarnings({
if ("date" %in% names(train_raw)) train_raw$date <- as.Date(train_raw$date)
if ("date" %in% names(test_raw))  test_raw$date  <- as.Date(test_raw$date)
})


infer_type <- function(x) {
cls <- class(x)
dplyr::case_when(
"Date"    %in% cls ~ "date",
"POSIXct" %in% cls ~ "datetime",
is.logical(x)      ~ "logical",
is.integer(x)      ~ "integer",
is.numeric(x)      ~ "numeric",
is.factor(x)       ~ "factor",
is.character(x)    ~ "character",
TRUE               ~ paste(cls, collapse = ",")
)
}

top_levels <- function(x, k = 5) {
if (is.factor(x)) x <- as.character(x)
if (!is.character(x)) return(NA_character_)
tb <- sort(table(x), decreasing = TRUE)
if (!length(tb)) return(NA_character_)
tb <- head(tb, k)
paste0(names(tb), " (", as.integer(tb), ")", collapse = "; ")
}

make_dqr <- function(df, dataset_name = "dataset") {
cols <- names(df)
tibble::tibble(
dataset     = dataset_name,
feature     = cols,
type        = map_chr(cols, ~ infer_type(df[[.x]])),
missing_n   = map_int(cols, ~ sum(is.na(df[[.x]]))),
missing_pct = round(map_dbl(cols, ~ mean(is.na(df[[.x]])) * 100), 2),
unique_n    = map_int(cols, ~ dplyr::n_distinct(df[[.x]], na.rm = TRUE)),
num_min     = map_dbl(cols, ~ if (infer_type(df[[.x]]) %in% c("numeric","integer")) suppressWarnings(min(df[[.x]], na.rm = TRUE)) else NA_real_),
num_max     = map_dbl(cols, ~ if (infer_type(df[[.x]]) %in% c("numeric","integer")) suppressWarnings(max(df[[.x]], na.rm = TRUE)) else NA_real_),
num_mean    = map_dbl(cols, ~ if (infer_type(df[[.x]]) %in% c("numeric","integer")) suppressWarnings(mean(df[[.x]], na.rm = TRUE)) else NA_real_),
num_sd      = map_dbl(cols, ~ if (infer_type(df[[.x]]) %in% c("numeric","integer")) suppressWarnings(sd(df[[.x]], na.rm = TRUE)) else NA_real_),
date_min    = map_chr(cols, ~ if (infer_type(df[[.x]]) == "date") as.character(suppressWarnings(min(df[[.x]], na.rm = TRUE))) else NA_character_),
date_max    = map_chr(cols, ~ if (infer_type(df[[.x]]) == "date") as.character(suppressWarnings(max(df[[.x]], na.rm = TRUE))) else NA_character_)
) |>
dplyr::mutate(
top_levels = map_chr(
cols,
~ if (infer_type(df[[.x]]) %in% c("character","factor")) top_levels(df[[.x]], 5) else NA_character_
)
) |>
dplyr::arrange(dplyr::desc(missing_pct), feature)
}


dqr_train <- make_dqr(train_raw, "train_raw")
dqr_test  <- make_dqr(test_raw,  "test_raw")


readr::write_csv(dqr_train, "DQR_train.csv")
readr::write_csv(dqr_test,  "DQR_test.csv")


knitr::kable(head(dqr_train, 20), caption = "Data Quality Report — Train (Top 20 by Missing %)")
knitr::kable(head(dqr_test,  20), caption = "Data Quality Report — Test (Top 20 by Missing %)")



suppressPackageStartupMessages({
library(dplyr)
library(ggplot2)
library(scales)
})

safe_entropy <- function(x) {
x <- x[!is.na(x)]
if (!length(x)) return(0)
DescTools::Entropy(prop.table(table(x))) |> as.numeric()
}

add_features_min <- function(df) {
df |>
dplyr::group_by(custId) |>
dplyr::summarise(
total_sessions   = dplyr::n(),
total_pageviews  = sum(pageviews, na.rm = TRUE),
bounce_rate      = mean(bounces, na.rm = TRUE),
new_visit_ratio  = mean(newVisits, na.rm = TRUE),
channel_entropy  = safe_entropy(channelGrouping)
) |>
dplyr::ungroup()
}


if (!exists("train_feats")) {
train_feats <- add_features_min(train_raw)
}


train_targets <- train_raw |>
dplyr::group_by(custId) |>
dplyr::summarise(targetRevenue = log(sum(revenue, na.rm = TRUE) + 1),
.groups = "drop")

# Visual 1: Distribution of targetRevenue

p1 <- ggplot(train_targets, aes(x = targetRevenue)) +
geom_histogram(bins = 40) +
labs(title = "Distribution of log(customer revenue + 1)",
x = "targetRevenue", y = "Count")
print(p1)
ggsave("viz_targetRevenue_hist.png", p1, width = 7, height = 4, dpi = 150)

# Visual 2: total_pageviews vs. targetRevenue (with smooth)

plot_df <- train_feats |>
dplyr::left_join(train_targets, by = "custId")

p2 <- ggplot(plot_df, aes(x = total_pageviews, y = targetRevenue)) +
geom_point(alpha = 0.25) +
geom_smooth(method = "loess", se = FALSE) +
scale_x_continuous(labels = scales::comma) +
labs(title = "Total Pageviews vs Log Revenue",
x = "Total Pageviews (per customer)", y = "targetRevenue")
print(p2)
ggsave("viz_pageviews_vs_target.png", p2, width = 7, height = 4, dpi = 150)



# ISE/DSA 5103 - HW7 
# Team Dataholics | Farzana Tasnim, Aksel Can Sozudogru
# Stacked Ensemble with Ridge + ENet + PLS + PCR + GAM + SVR

suppressPackageStartupMessages({
  library(data.table)
  library(tidyverse)
  library(caret)
  library(glmnet)
  library(pls)
  library(e1071)
  library(nnls)
  library(lubridate)
  library(DescTools)
  library(readr)
  library(mgcv)
})

set.seed(5103)


train_path <- "Train.csv"
test_path  <- "Test.csv"

train_raw <- fread(train_path, na.strings = c("", "NA", "null", "(not set)"))
test_raw  <- fread(test_path,  na.strings = c("", "NA", "null", "(not set)"))

train_raw$date <- as.Date(train_raw$date)
test_raw$date  <- as.Date(test_raw$date)


safe_entropy <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) return(0)
  as.numeric(Entropy(prop.table(table(x))))
}

add_features <- function(df) {
  df %>%
    group_by(custId) %>%
    summarise(
      total_sessions = n(),
      first_date = min(date, na.rm = TRUE),
      last_date  = max(date, na.rm = TRUE),
      visit_span_days = as.numeric(last_date - first_date) + 1,

      total_pageviews = sum(pageviews, na.rm = TRUE),
      avg_pageviews   = mean(pageviews, na.rm = TRUE),
      sd_pageviews    = sd(pageviews, na.rm = TRUE),
      cv_pageviews    = sd(pageviews, na.rm = TRUE) / (mean(pageviews, na.rm = TRUE) + 1e-6),
      bounce_rate     = mean(bounces, na.rm = TRUE),
      new_visit_ratio = mean(newVisits, na.rm = TRUE),

      ts_mean = mean(timeSinceLastVisit, na.rm = TRUE),
      ts_sd   = sd(timeSinceLastVisit, na.rm = TRUE),

      pct_organic = mean(channelGrouping == "Organic Search", na.rm = TRUE),
      pct_paid    = mean(channelGrouping == "Paid Search", na.rm = TRUE),
      pct_direct  = mean(channelGrouping == "Direct", na.rm = TRUE),
      pct_ref     = mean(channelGrouping == "Referral", na.rm = TRUE),
      pct_social  = mean(channelGrouping == "Social", na.rm = TRUE),

      channel_entropy = safe_entropy(channelGrouping),
      device_entropy  = safe_entropy(deviceCategory),
      geo_entropy     = safe_entropy(country),

      is_mobile_user  = as.numeric(any(isMobile == 1)),
      is_desktop_user = as.numeric(any(deviceCategory == "desktop")),
      browser_diversity = n_distinct(browser),
      os_diversity      = n_distinct(operatingSystem),
      country_diversity = n_distinct(country),
      is_us_user        = as.numeric(any(country == "United States")),

      sessions_7d  = sum(date >= (last_date - 7),  na.rm = TRUE),
      sessions_30d = sum(date >= (last_date - 30), na.rm = TRUE)
    ) %>%
    mutate(
      visit_span_days = ifelse(visit_span_days <= 0 | is.na(visit_span_days), 1, visit_span_days),
      sessions_per_day = total_sessions / visit_span_days,
      pageviews_per_session = ifelse(total_sessions > 0, total_pageviews / total_sessions, 0),
      engagement_score = total_pageviews * (1 - bounce_rate),
      cv_pageviews = ifelse(is.na(cv_pageviews) | is.infinite(cv_pageviews), 0, cv_pageviews)
    ) %>%
    select(-first_date, -last_date)
}

train_feats <- add_features(train_raw)
test_feats  <- add_features(test_raw)

target_tbl <- train_raw %>%
  group_by(custId) %>%
  summarise(total_revenue = sum(revenue, na.rm = TRUE)) %>%
  mutate(targetRevenue = log(total_revenue + 1),
         has_revenue = as.integer(total_revenue > 0))

train_feats <- train_feats %>% left_join(target_tbl, by = "custId")

train_feats <- train_feats %>% filter(total_sessions > 0)
test_feats  <- test_feats  %>% filter(total_sessions > 0)


numify <- function(df) {
  df %>%
    mutate(across(everything(), ~ if (is.logical(.x)) as.integer(.x) else .x)) %>%
    mutate(across(where(is.character), ~ as.factor(.))) %>%
    mutate(across(where(is.factor), ~ as.integer(.))) %>%
    mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
}

X_all <- train_feats %>%
  select(-custId, -total_revenue, -targetRevenue, -has_revenue) %>%
  numify() %>% as.data.frame()
y <- train_feats$targetRevenue
y_bin <- train_feats$has_revenue

X_test <- test_feats %>%
  select(any_of(colnames(X_all))) %>%
  numify() %>% as.data.frame()

missing_cols <- setdiff(colnames(X_all), colnames(X_test))
if (length(missing_cols)) X_test[, missing_cols] <- 0
X_test <- X_test[, colnames(X_all), drop = FALSE]

# remove constant/NZV cols
nzv <- caret::nearZeroVar(X_all)
if (length(nzv) > 0) {
  X_all <- X_all[, -nzv, drop = FALSE]
  X_test <- X_test[, colnames(X_all), drop = FALSE]
}

# PCA
pca_obj <- prcomp(X_all, center = TRUE, scale. = TRUE)
var_cum <- summary(pca_obj)$importance[3, ]
k_pcs <- min(which(var_cum >= 0.93)[1], 40)
X_pca <- as.data.frame(pca_obj$x[, 1:k_pcs, drop = FALSE])
X_test_pca <- as.data.frame(predict(pca_obj, newdata = X_test)[, 1:k_pcs, drop = FALSE])


sanitize_df <- function(df) {
  df <- as.data.frame(df)
  df[!is.finite(as.matrix(df))] <- 0
  df[is.na(df)] <- 0
  return(df)
}

# model stacking
K <- 5  
folds <- createFolds(y, k = K, returnTrain = FALSE)
ctrl_cv <- trainControl(method = "cv", number = 5)
rmse <- function(a,b) sqrt(mean((a-b)^2))

oof <- list(ridge=numeric(nrow(X_all)), enet=numeric(nrow(X_all)), pls=numeric(nrow(X_all)),
            pcr=numeric(nrow(X_all)), gam=numeric(nrow(X_all)), svr=numeric(nrow(X_all)),
            svr_pca=numeric(nrow(X_all)), twostage_svr=numeric(nrow(X_all)))
test_preds <- lapply(oof, function(x) rep(0, nrow(X_test)))

for (k in seq_along(folds)) {
  cat(sprintf("\nFold %d/%d\n", k, K))
  val_idx <- folds[[k]]
  tr_idx <- setdiff(seq_len(nrow(X_all)), val_idx)

  X_tr <- sanitize_df(X_all[tr_idx, , drop=FALSE])
  X_va <- sanitize_df(X_all[val_idx, , drop=FALSE])
  y_tr <- y[tr_idx]; y_va <- y[val_idx]
  X_tr_pca <- sanitize_df(X_pca[tr_idx, , drop=FALSE])
  X_va_pca <- sanitize_df(X_pca[val_idx, , drop=FALSE])
  X_test_pca <- sanitize_df(X_test_pca)

  # ----- Ridge -----
  ridge_fit <- train(
    x=X_tr, y=y_tr, method="glmnet",
    tuneGrid=expand.grid(alpha=0, lambda=10^seq(-3,2,length=30)),
    trControl=ctrl_cv, preProcess=c("center","scale")
  )
  oof$ridge[val_idx] <- predict(ridge_fit, X_va)
  test_preds$ridge <- test_preds$ridge + predict(ridge_fit, X_test) / K

  # ----- Elastic Net -----
  enet_fit <- train(
    x=X_tr, y=y_tr, method="glmnet",
    tuneGrid=expand.grid(alpha=seq(0.05,0.95,length=7), lambda=10^seq(-3.5,1,length=25)),
    trControl=ctrl_cv, preProcess=c("center","scale")
  )
  oof$enet[val_idx] <- predict(enet_fit, X_va)
  test_preds$enet <- test_preds$enet + predict(enet_fit, X_test) / K

  # ----- PLS -----
  pls_fit <- train(x=X_tr, y=y_tr, method="pls",
                   tuneGrid=expand.grid(ncomp=1:10),
                   trControl=ctrl_cv, preProcess=c("center","scale"))
  oof$pls[val_idx] <- predict(pls_fit, X_va)
  test_preds$pls <- test_preds$pls + predict(pls_fit, X_test) / K

  # ----- PCR -----
  pcr_fit <- pcr(y_tr ~ ., data=data.frame(y_tr, X_tr_pca),
                 ncomp=min(20, ncol(X_tr_pca)), validation="CV")
  n_use <- pcr_fit$ncomp
  pcr_va <- as.numeric(predict(pcr_fit, newdata=X_va_pca, ncomp=n_use))
  pcr_te <- as.numeric(predict(pcr_fit, newdata=X_test_pca, ncomp=n_use))
  oof$pcr[val_idx] <- pcr_va
  test_preds$pcr <- test_preds$pcr + pcr_te / K

  # ----- GAM on PCs -----
  pc_names <- paste0("PC", seq_len(ncol(X_tr_pca)))
  names(X_tr_pca) <- names(X_va_pca) <- names(X_test_pca) <- pc_names
  smooth_terms <- paste0("s(", pc_names, ", bs='cs')", collapse=" + ")
  gam_form <- as.formula(paste("y_tr ~", smooth_terms))
  gam_fit <- gam(gam_form, data=cbind(y_tr=y_tr, X_tr_pca), method="REML", select=TRUE)
  pred_va <- as.numeric(predict(gam_fit, newdata=X_va_pca))
  pred_te <- as.numeric(predict(gam_fit, newdata=X_test_pca))
  pred_va[!is.finite(pred_va)] <- mean(pred_va, na.rm=TRUE)
  pred_te[!is.finite(pred_te)] <- mean(pred_te, na.rm=TRUE)
  oof$gam[val_idx] <- pred_va
  test_preds$gam <- test_preds$gam + pred_te / K

  # ----- SVR (RBF) -----
  X_tr <- sanitize_df(X_tr); X_va <- sanitize_df(X_va)
  X_test <- sanitize_df(X_test)
  svr_fit <- svm(x=X_tr, y=y_tr, kernel="radial", cost=5, gamma=1/ncol(X_tr), epsilon=0.05)
  oof$svr[val_idx] <- predict(svr_fit, X_va)
  test_preds$svr <- test_preds$svr + predict(svr_fit, X_test) / K

  # ----- SVR on PCs -----
  svr_pca_fit <- svm(x=X_tr_pca, y=y_tr, kernel="radial", cost=5,
                     gamma=1/ncol(X_tr_pca), epsilon=0.05)
  oof$svr_pca[val_idx] <- predict(svr_pca_fit, X_va_pca)
  test_preds$svr_pca <- test_preds$svr_pca + predict(svr_pca_fit, X_test_pca) / K

  # ----- TWO-STAGE -----
  logit_fit <- glm(y_bin[tr_idx] ~ ., data=data.frame(X_tr), family=binomial())
  p_buy_va <- plogis(predict(logit_fit, newdata=X_va, type="link"))
  p_buy_te <- plogis(predict(logit_fit, newdata=X_test, type="link"))
  pos_idx <- which(y_bin[tr_idx] == 1)
  svr_pos <- svm(x=X_tr[pos_idx, , drop=FALSE], y=y_tr[pos_idx],
                 kernel="radial", cost=6, gamma=1/ncol(X_tr), epsilon=0.06)
  amount_va <- predict(svr_pos, X_va)
  amount_te <- predict(svr_pos, X_test)
  oof$twostage_svr[val_idx] <- p_buy_va * amount_va
  test_preds$twostage_svr <- test_preds$twostage_svr + (p_buy_te * amount_te) / K
}


oof_mat <- as.data.frame(oof)
rmse_each <- sapply(oof_mat, function(col) rmse(col, y))
cat("\n==== OOF RMSE by model ====\n"); print(round(rmse_each, 4))

Z_oof <- as.matrix(oof_mat)
w <- nnls(Z_oof, y)$x
w <- pmax(w, 0); w <- w / sum(w)
names(w) <- colnames(oof_mat)
blend_oof <- as.numeric(Z_oof %*% w)
rmse_blend <- rmse(blend_oof, y)

cat("\n==== Blend Weights (NNLS) ====\n"); print(round(w, 3))
cat(sprintf("Blended OOF RMSE: %.4f\n", rmse_blend))


Z_test <- do.call(cbind, test_preds)
Z_test <- as.matrix(as.data.frame(Z_test)[, names(w), drop=FALSE])
final_pred <- as.numeric(Z_test %*% w)
final_pred <- pmax(final_pred, 0)
final_pred <- pmin(final_pred, quantile(y, 0.995, na.rm=TRUE))
final_pred[!is.finite(final_pred)] <- median(final_pred, na.rm=TRUE)

submission <- data.frame(custId=test_feats$custId, predRevenue=final_pred)
write.csv(submission, "final_stack_nnls_GAM_submission.csv", row.names=FALSE)


print(round(rmse_each, 4))
cat(sprintf("OOF Blend RMSE: %.4f\n", rmse_blend))
print(round(w, 3))


suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(caret)
  library(glmnet)
  library(pls)
  library(MASS)
  library(kernlab)
  library(knitr)
  library(readr)
})

set.seed(5103)

train <- fread("Train.csv", na.strings = c("", "NA", "null", "(not set)"))
train$date <- as.Date(train$date)

agg <- train %>%
  group_by(custId) %>%
  summarise(
    total_sessions   = n(),
    total_pageviews  = sum(pageviews, na.rm = TRUE),
    avg_pageviews    = mean(pageviews, na.rm = TRUE),
    bounce_rate      = mean(bounces, na.rm = TRUE),
    new_visit_ratio  = mean(newVisits, na.rm = TRUE),
    pct_organic      = mean(channelGrouping == "Organic Search", na.rm = TRUE),
    pct_paid         = mean(channelGrouping == "Paid Search", na.rm = TRUE),
    pct_direct       = mean(channelGrouping == "Direct", na.rm = TRUE),
    pct_ref          = mean(channelGrouping == "Referral", na.rm = TRUE),
    pct_social       = mean(channelGrouping == "Social", na.rm = TRUE),
    .groups = "drop"
  )

target <- train %>%
  group_by(custId) %>%
  summarise(total_revenue = sum(revenue, na.rm = TRUE), .groups = "drop") %>%
  mutate(targetRevenue = log(total_revenue + 1))

df <- left_join(agg, target, by = "custId") %>%
  dplyr::select(-custId, -total_revenue)


for (col in names(df)) {
  if (is.numeric(df[[col]])) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
  }
}

ctrl <- trainControl(method = "cv", number = 5)
form <- as.formula("targetRevenue ~ .")

models <- list(
  OLS = train(form, data = df, method = "lm", trControl = ctrl),
  Ridge = train(form, data = df, method = "glmnet",
                tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-3,2,length=30)),
                preProcess = c("center","scale"), trControl = ctrl),
  LASSO = train(form, data = df, method = "glmnet",
                tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-3,2,length=30)),
                preProcess = c("center","scale"), trControl = ctrl),
  ElasticNet = train(form, data = df, method = "glmnet",
                     tuneGrid = expand.grid(alpha = seq(0.1,0.9,by=0.2),
                                            lambda = 10^seq(-3,1,length=20)),
                     preProcess = c("center","scale"), trControl = ctrl),
  PLS = train(form, data = df, method = "pls",
              tuneGrid = expand.grid(ncomp = 1:10),
              preProcess = c("center","scale"), trControl = ctrl),
  PCR = train(form, data = df, method = "pcr",
              tuneGrid = expand.grid(ncomp = 1:10),
              preProcess = c("center","scale"), trControl = ctrl),
  SVR = train(form, data = df, method = "svmRadial",
              preProcess = c("center","scale"), tuneLength = 6, trControl = ctrl),
  Robust = train(form, data = df, method = "rlm",
                 preProcess = c("center","scale"), trControl = ctrl)
)

extract_perf <- function(model) {
  best <- model$bestTune
  res <- model$results
  if (!is.null(best)) {
    res <- merge(res, best)
    res <- res[1, ]
  } else res <- res[1, ]
  data.frame(
    Model = model$method,
    RMSE = round(res$RMSE, 4),
    R2 = round(res$Rsquared, 4),
    Hyperparameters = paste(names(best), best, collapse=", ")
  )
}

perf <- bind_rows(lapply(models, extract_perf))

perf <- perf %>%
  mutate(Package = case_when(
    Model == "lm" ~ "stats",
    grepl("glmnet", Model) ~ "glmnet",
    Model %in% c("pls", "pcr") ~ "pls",
    grepl("svm", Model) ~ "kernlab",
    Model == "rlm" ~ "MASS",
    TRUE ~ "caret"
  )) %>%
  rename(Method = Model)

kable(perf, caption = "Table 1. Cross-validated Model Performance (5-Fold CV)")
write_csv(perf, "model_performance_table.csv")

