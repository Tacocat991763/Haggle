# Dataset analysis (train.csv)

4,500 projects × 73 columns. Model inputs exclude `project_id`, `hours_spent`, `log_hours`, `hours_bucket`.
There is no local `test.csv`; all findings below come from the training data only.

## Target

- `hours_spent`: 0.1 to 140.7, median 6.0, mean 12.1. Right-skewed. `log1p(hours)` has mean 2.04 and std 0.96.
- **The target is capped at 140.7 hours** (24 rows at the cap, about the 99.5th percentile). Predictions above 140.7 can only hurt, so inference clips to `[0.1, 140.7]`.
- **The labels look self-reported or rounded.** 37% are whole numbers and 45% are multiples of 0.5. The most common values are 1, 2, 3, 4, 5, 6, 8 and 10 hours. This gives an irreducible noise floor: the best models reach about 0.63–0.65 RMSLE.
- `log_hours` is **natural `log(hours)`**, not `log1p`, at higher precision than the 0.1-rounded `hours_spent`. It is a label, never a feature.
- `hours_bucket` is a coarse label (<10, 10-30, 30-100, 100+), with 5 rows bucketed inconsistently. Never used.

## Columns and missingness

- 45 numeric, 17 boolean (5 of them object-typed with a few NaNs), 5 nominal categoricals (`commit_cadence`, `primary_language` 22 levels, `framework` 18, `hosting_type` 17, `project_category` 6 + 5 NaN), and 2 ordinal string columns (`author_prior_projects`, `author_account_age_years`).
- Missing values follow fixed structures. None of it looks random:
  - `avg_file_length`, `comment_line_ratio`, `pct_primary_language` are NaN **exactly when** `num_source_files == 0` (194 rows, `primary_language == "none"`).
  - `pct_code_in_first_commit` is NaN for 186 rows, mostly `lines_added_total == 0`.
  - Web metrics (`page_weight_kb`, `load_time_ms` 1,349 NaN; `num_js_files_loaded`, `num_internal_pages`, `is_mobile_responsive` 1,423 NaN) are NaN for every row where `url_is_live == False`, and for non-web hosting types (github-repo, video, package-registry, mod-platform, chat-invite).
- Missingness is predictive. Rows without source files average 1.64 log-hours against 2.06 overall. Rows without web metrics average 2.33 against 1.91. Most of this signal is already carried by `num_source_files == 0` and `hosting_type`, so explicit indicators are close to redundant for tree models (see ablation).

## The data looks anonymized or synthetic

- `lines_added_total`, `lines_deleted_total`, `lines_of_code`, `max_lines_in_one_commit` and `load_time_ms` are **all multiples of 100**. They were coarsened on purpose, which wipes out the small-project resolution: 320 projects have `lines_of_code == 0`.
- `churn_ratio ≈ lines_deleted / lines_added` (median abs error 0.005). It was derived before rounding, so it carries finer information than the rounded line counts.
- `lines_of_code ≈ num_source_files × avg_file_length` (median ratio 1.00). `avg_file_length` is unrounded, so `num_source_files × avg_file_length` recovers a **de-rounded LOC estimate**.
- There are 363 rows where `project_span_days + 1 < active_commit_days`, which is impossible for real git history. This points to noise injection or independent synthesis of the two columns.
- There are 22 exact duplicate feature rows. Row order carries no information: an adversarial classifier separating the first 80% from the last 20% reaches AUC 0.48.
- Taken together, this is most likely real Hack Club project metadata with rounding and noise added, plus labels that are noisy self-reports. There is no clean closed-form generator to reverse-engineer.

## Relationships with log-hours (Spearman)

- Activity is strongest: `active_commit_days` 0.56, `lines_added_total` 0.53, `lines_of_code` 0.53, `lines_deleted_total` 0.53, `num_commits` 0.51, `num_source_files` 0.50, `project_span_days` 0.43.
- Negative: `pct_generic_commit_msgs` -0.38, `pct_code_in_first_commit` -0.36. Bulk-upload projects take less time per line.
- `pct_commits_weekend` has near-zero Spearman but high mutual information (0.19). Its discrete values (0, 1/n, ...) encode commit count, so the relationship is non-monotonic.
- Weak or no signal: `num_issues`, `num_test_files`, `readme_num_code_blocks`, `forks`, `num_3d_model_files` (mutual information < 0.02).
- Categories (mean log-hours):
  - Websites 1.64, games 2.03, apps 2.23, tools 2.32.
  - HTML/CSS 1.46/1.69 against C#/Rust/Java about 2.55–2.60.
  - gamemaker/slack-bot frameworks about 1.58 against unity/express about 2.4.
  - `has_database` 2.59 against 2.00; `is_multiplayer` 2.61 against 2.01; `mentions_ai_api` 2.40 against 2.01.
  - `commit_cadence`: daily 2.40, sporadic 2.20, frequent 1.84.
  - Author experience barely matters: `<1` year account age 1.78, otherwise about 2.15.
- Redundant pairs (|Spearman| > 0.85): `active_commit_days`/`project_span_days`, `lines_added_total`/`max_lines_in_one_commit`, `num_directories`/`max_dir_depth`.

## Simple baselines (5-fold CV on all 4,500 rows, log1p target)

| Model | RMSLE |
|---|---|
| Global mean | 0.963 |
| Project-category mean | 0.936 |
| Ridge on 6 log activity features | 0.791 |
| CatBoost, raw features (previous submission config) | 0.659 (dev CV) |
| Previous submission, public leaderboard | 0.7548 |

The previous CatBoost scored 0.672 on its own last-20% validation but **0.7548 on the leaderboard**. Row order shows no drift, so the gap most likely comes from a test-set distribution shift or a small, noisy leaderboard sample. This argues for robust, well-regularized ensembles over hyperparameter overfitting to CV.
