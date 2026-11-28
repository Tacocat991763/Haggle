"""Feature pipeline shared by training and inference.

`FeatureBuilder(families).fit(train_df).transform(df)` returns a DataFrame with
numeric columns (float, NaN allowed) plus string categorical columns listed in
`builder.cat_cols`. Everything learned (group medians, category levels) comes
from the frame passed to `fit` only.
"""
import numpy as np
import pandas as pd

CAT_COLS = ["commit_cadence", "primary_language", "framework", "hosting_type", "project_category"]
ORDINAL = {
    "author_prior_projects": {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
    "author_account_age_years": {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
}
CADENCE_ORD = {"sporadic": 0, "frequent": 1, "daily": 2}
BOOL_COLS = [
    "uses_game_engine", "has_tests", "has_ci_config", "has_dockerfile", "has_backend", "has_database",
    "has_custom_art", "readme_has_install_instructions", "has_license", "has_docs_folder", "url_is_live",
    "is_mobile_responsive", "has_custom_domain", "is_multiplayer", "mentions_ai_api", "is_clone_or_tutorial",
    "is_forked_from_template",
]
NUM_COLS = [
    "num_commits", "active_commit_days", "project_span_days", "pct_commits_weekend", "pct_commits_late_night",
    "lines_added_total", "lines_deleted_total", "churn_ratio", "median_lines_per_commit", "max_lines_in_one_commit",
    "pct_code_in_first_commit", "num_commit_authors", "avg_commit_msg_length", "pct_generic_commit_msgs",
    "num_branches", "num_merge_commits", "lines_of_code", "num_source_files", "num_directories", "max_dir_depth",
    "num_languages", "pct_primary_language", "num_dependencies", "num_test_files", "avg_file_length",
    "comment_line_ratio", "num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files",
    "asset_size_mb", "readme_word_count", "readme_num_headings", "readme_num_images", "readme_num_code_blocks",
    "page_weight_kb", "num_js_files_loaded", "num_internal_pages", "load_time_ms", "description_word_count",
    "stars", "forks", "num_issues", "num_pull_requests", "num_releases",
]
SKEWED = [
    "num_commits", "active_commit_days", "project_span_days", "lines_added_total", "lines_deleted_total",
    "median_lines_per_commit", "max_lines_in_one_commit", "num_commit_authors", "avg_commit_msg_length",
    "num_branches", "num_merge_commits", "lines_of_code", "num_source_files", "num_directories",
    "num_dependencies", "num_test_files", "avg_file_length", "num_image_files", "num_audio_files",
    "num_3d_model_files", "num_font_files", "asset_size_mb", "readme_word_count", "readme_num_headings",
    "readme_num_images", "readme_num_code_blocks", "page_weight_kb", "num_js_files_loaded", "load_time_ms",
    "description_word_count", "stars", "forks", "num_issues", "num_pull_requests", "num_releases",
]
MISSING_SRC = ["churn_ratio", "pct_code_in_first_commit", "avg_file_length", "page_weight_kb", "num_js_files_loaded"]
REL_GROUPS = ["project_category", "primary_language", "framework"]
REL_FEATS = ["l_lines_added_total", "l_active_commit_days", "l_num_commits", "l_project_span_days",
             "l_lines_of_code", "l_num_source_files"]
ALL_FAMILIES = ("log", "ratio", "activity", "missing", "rel", "interact")


def _to_bool(s):
    m = {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0, True: 1.0, False: 0.0, 1: 1.0, 0: 0.0}
    return s.map(lambda v: m.get(v, np.nan) if not (isinstance(v, float) and np.isnan(v)) else np.nan).astype(float)


def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r[~np.isfinite(r)] = np.nan
    return r


def _l(x):
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


class FeatureBuilder:
    def __init__(self, families=ALL_FAMILIES, keep_raw_skewed=True):
        self.families = set(families)
        self.keep_raw_skewed = keep_raw_skewed
        self.cat_cols = list(CAT_COLS)

    # ---- base -------------------------------------------------------------
    def _base(self, df):
        d = pd.DataFrame(index=df.index)
        for c in NUM_COLS:
            d[c] = pd.to_numeric(df[c], errors="coerce").astype(float) if c in df else np.nan
        for c in BOOL_COLS:
            d[c] = _to_bool(df[c]) if c in df else np.nan
        for c, m in ORDINAL.items():
            d[c] = df[c].astype(str).map(m).astype(float) if c in df else np.nan
        d["cadence_ord"] = df["commit_cadence"].astype(str).map(CADENCE_ORD).astype(float)
        for c in CAT_COLS:
            d[c] = df[c].astype("string").fillna("__NA__").astype(str) if c in df else "__NA__"
        return d

    def fit(self, df):
        d = self._base(df)
        self.levels = {c: sorted(d[c].unique().tolist()) for c in CAT_COLS}
        # group medians for relative features (feature-only statistics, fitted on train portion)
        self.group_med, self.global_med = {}, {}
        tmp = pd.DataFrame({f: _l(d[f[2:]]) for f in REL_FEATS})
        for g in REL_GROUPS:
            tmp[g] = d[g].values
            self.group_med[g] = {}
            counts = tmp[g].value_counts()
            ok = counts[counts >= 10].index
            for f in REL_FEATS:
                self.group_med[g][f] = tmp[tmp[g].isin(ok)].groupby(g)[f].median().to_dict()
        for f in REL_FEATS:
            self.global_med[f] = float(np.nanmedian(tmp[f]))
        return self

    def transform(self, df):
        d = self._base(df)
        F = self.families
        out = {}
        if "log" in F:
            for c in SKEWED:
                out["l_" + c] = _l(d[c])
        commits = d.num_commits.clip(lower=1)
        days = d.active_commit_days.clip(lower=1)
        span = d.project_span_days
        added, deleted = d.lines_added_total, d.lines_deleted_total
        churn = added + deleted
        authors = d.num_commit_authors.clip(lower=1)
        src_est = d.num_source_files * d.avg_file_length
        assets = d.num_image_files + d.num_audio_files + d.num_3d_model_files + d.num_font_files
        if "ratio" in F:
            out["commits_per_day"] = _div(d.num_commits, days)
            out["l_added_per_commit"] = _l(_div(added, commits))
            out["l_deleted_per_commit"] = _l(_div(deleted, commits))
            out["l_churn_per_day"] = _l(_div(churn, days))
            out["active_frac"] = _div(d.active_commit_days, span + 1)
            out["non_merge_commits"] = _l(d.num_commits - d.num_merge_commits)
            out["l_churn_per_author"] = _l(_div(churn, authors))
            out["days_per_author"] = _div(d.active_commit_days, authors)
            out["l_src_est"] = _l(src_est)
            out["l_comment_lines"] = _l(src_est * d.comment_line_ratio)
            out["deps_per_file"] = _div(d.num_dependencies, d.num_source_files + 1)
            out["tests_per_file"] = _div(d.num_test_files, d.num_source_files + 1)
            out["l_assets"] = _l(assets)
            out["assets_per_file"] = _div(assets, d.num_source_files + 1)
            out["branches_per_commit"] = _div(d.num_branches, commits)
            out["issues_pr_per_day"] = _div(d.num_issues + d.num_pull_requests, days)
            out["readme_per_heading"] = _div(d.readme_word_count, d.readme_num_headings + 1)
            out["loc_retention"] = _l(d.lines_of_code) - _l(added)
        if "activity" in F:
            first = added * d.pct_code_in_first_commit
            out["l_after_first"] = _l((added - first).clip(lower=0))
            out["l_after_max"] = _l((churn - d.max_lines_in_one_commit).clip(lower=0))
            out["max_share"] = _div(d.max_lines_in_one_commit, added.clip(lower=1))
            out["weekend_commits"] = _l(d.num_commits * d.pct_commits_weekend)
            out["late_commits"] = _l(d.num_commits * d.pct_commits_late_night)
            out["descriptive_commits"] = _l(d.num_commits * (1 - d.pct_generic_commit_msgs))
            out["irregularity"] = _l(_div(d.max_lines_in_one_commit, d.median_lines_per_commit + 1))
            out["typical_churn"] = _l(d.median_lines_per_commit * d.num_commits)
            out["msg_chars_total"] = _l(d.num_commits * d.avg_commit_msg_length)
            out["l_span_per_day"] = _l(_div(span + 1, days))
        if "missing" in F:
            for c in MISSING_SRC:
                out["na_" + c] = d[c].isna().astype(float).values
            out["n_missing"] = d.isna().sum(axis=1).astype(float).values
        if "rel" in F:
            for g in REL_GROUPS:
                for f in REL_FEATS:
                    med = d[g].map(self.group_med[g][f]).astype(float).fillna(self.global_med[f])
                    out[f"{f}_rel_{g}"] = _l(d[f[2:]]) - med.values
        if "interact" in F:
            lc, ld, la, ll = _l(d.num_commits), _l(d.active_commit_days), _l(added), _l(d.lines_of_code)
            out["x_days_commits"] = ld * lc
            out["x_days_added"] = ld * la
            out["x_commits_loc"] = lc * ll
            out["x_days_files"] = ld * _l(d.num_source_files)
        res = pd.concat([d, pd.DataFrame(out, index=d.index)], axis=1)
        if not self.keep_raw_skewed:
            res = res.drop(columns=[c for c in SKEWED if "l_" + c in res])
        num = [c for c in res.columns if c not in CAT_COLS]
        res[num] = res[num].astype(float).replace([np.inf, -np.inf], np.nan)
        return res

    def num_cols(self, X):
        return [c for c in X.columns if c not in self.cat_cols]
