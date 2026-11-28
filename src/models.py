"""Model wrappers with a common interface: Model(params, cat_cols).fit(X, y_log).predict(X) -> log1p hours.

All preprocessing specific to a model family (encoding, scaling, imputation,
target encoding) is fitted inside `fit`, so CV folds never see validation data.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- encoders
class TargetEncoder:
    """Smoothed mean target encoding. fit_transform uses inner out-of-fold means for training rows."""

    def __init__(self, cols, smoothing=20, n_inner=5, seed=0):
        self.cols, self.m, self.n_inner, self.seed = cols, smoothing, n_inner, seed

    def _stats(self, X, y):
        prior = float(np.mean(y))
        maps = {}
        for c in self.cols:
            g = pd.DataFrame({"k": X[c].values, "y": y}).groupby("k")["y"].agg(["sum", "count"])
            maps[c] = ((g["sum"] + self.m * prior) / (g["count"] + self.m)).to_dict()
        return prior, maps

    def fit(self, X, y):
        self.prior, self.maps = self._stats(X, y)
        return self

    def transform(self, X):
        return pd.DataFrame({f"te_{c}": X[c].map(self.maps[c]).astype(float).fillna(self.prior).values
                             for c in self.cols}, index=X.index)

    def fit_transform(self, X, y):
        self.fit(X, y)
        out = pd.DataFrame(index=X.index, columns=[f"te_{c}" for c in self.cols], dtype=float)
        rng = np.random.RandomState(self.seed)
        fold = rng.randint(0, self.n_inner, len(X))
        for k in range(self.n_inner):
            tr, va = fold != k, fold == k
            prior, maps = self._stats(X[tr], y[tr])
            for c in self.cols:
                out.loc[X.index[va], f"te_{c}"] = X[c][va].map(maps[c]).astype(float).fillna(prior).values
        return out


def _as_category(X, cat_cols, levels=None):
    X = X.copy()
    lv = {}
    for c in cat_cols:
        cats = levels[c] if levels else sorted(X[c].unique().tolist())
        lv[c] = cats
        X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)
    return X, lv


class DenseEncoder:
    """Numeric matrix for linear / NN / kNN models: median-impute, robust clip, standardize, one-hot cats, optional TE."""

    def __init__(self, cat_cols, min_freq=15, te=True, onehot=True):
        self.cat_cols, self.min_freq, self.te, self.onehot = cat_cols, min_freq, te, onehot

    def fit_transform(self, X, y):
        num = X.drop(columns=self.cat_cols)
        self.num_cols = num.columns
        self.med = num.median()
        num = num.fillna(self.med)
        self.lo, self.hi = num.quantile(0.005), num.quantile(0.995)
        num = num.clip(self.lo, self.hi, axis=1)
        self.mu, self.sd = num.mean(), num.std().replace(0, 1)
        parts = [((num - self.mu) / self.sd)]
        if self.onehot:
            self.ohe_levels = {c: [v for v, n in X[c].value_counts().items() if n >= self.min_freq] for c in self.cat_cols}
            parts.append(self._ohe(X))
        if self.te:
            self.enc = TargetEncoder(self.cat_cols)
            t = self.enc.fit_transform(X, y)
            self.te_mu, self.te_sd = t.mean(), t.std().replace(0, 1)
            parts.append((t - self.te_mu) / self.te_sd)
        return np.column_stack([p.values.astype(float) for p in parts])

    def _ohe(self, X):
        return pd.DataFrame({f"{c}={v}": (X[c] == v).astype(float) for c in self.cat_cols for v in self.ohe_levels[c]},
                            index=X.index)

    def transform(self, X):
        num = X[self.num_cols].fillna(self.med).clip(self.lo, self.hi, axis=1)
        parts = [((num - self.mu) / self.sd)]
        if self.onehot:
            parts.append(self._ohe(X))
        if self.te:
            parts.append((self.enc.transform(X) - self.te_mu) / self.te_sd)
        return np.column_stack([p.values.astype(float) for p in parts])


# ---------------------------------------------------------------- models
class Base:
    def __init__(self, params=None, cat_cols=(), seed=0):
        self.params = dict(params or {})
        self.cat_cols = list(cat_cols)
        self.seed = seed


class CatModel(Base):
    def fit(self, X, y):
        from catboost import CatBoostRegressor
        p = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3, loss_function="RMSE",
                 verbose=False, allow_writing_files=False, thread_count=8)
        p.update(self.params)
        p["random_seed"] = self.seed
        self.m = CatBoostRegressor(**p)
        self.m.fit(X, y, cat_features=self.cat_cols)
        return self

    def predict(self, X):
        return self.m.predict(X)


class LGBModel(Base):
    def fit(self, X, y):
        import lightgbm as lgb
        p = dict(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, verbose=-1, n_jobs=8,
                 cat_smooth=20, min_data_per_group=30)
        p.update(self.params)
        p["random_state"] = self.seed
        Xc, self.lv = _as_category(X, self.cat_cols)
        self.m = lgb.LGBMRegressor(**p).fit(Xc, y)
        return self

    def predict(self, X):
        return self.m.predict(_as_category(X, self.cat_cols, self.lv)[0])


class XGBModel(Base):
    def fit(self, X, y):
        import xgboost as xgb
        p = dict(n_estimators=800, learning_rate=0.02, max_depth=4, min_child_weight=5, subsample=0.8,
                 colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                 max_cat_to_onehot=1, n_jobs=8)
        p.update(self.params)
        p["random_state"] = self.seed
        Xc, self.lv = _as_category(X, self.cat_cols)
        self.m = xgb.XGBRegressor(**p).fit(Xc, y)
        return self

    def predict(self, X):
        return self.m.predict(_as_category(X, self.cat_cols, self.lv)[0])


class RidgeModel(Base):
    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        self.enc = DenseEncoder(self.cat_cols)
        A = self.enc.fit_transform(X, y)
        self.m = Ridge(alpha=self.params.get("alpha", 30.0)).fit(A, y)
        return self

    def predict(self, X):
        return self.m.predict(self.enc.transform(X))


class ETModel(Base):
    def fit(self, X, y):
        from sklearn.ensemble import ExtraTreesRegressor
        p = dict(n_estimators=500, min_samples_leaf=5, max_features=0.5, n_jobs=8)
        p.update(self.params)
        self.enc = DenseEncoder(self.cat_cols, onehot=False, te=True)
        A = self.enc.fit_transform(X, y)
        self.m = ExtraTreesRegressor(random_state=self.seed, **p).fit(A, y)
        return self

    def predict(self, X):
        return self.m.predict(self.enc.transform(X))


class MLPModel(Base):
    """Small torch MLP ensemble-friendly regressor on DenseEncoder features."""

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        p = dict(hidden=128, layers=2, dropout=0.2, lr=2e-3, wd=1e-3, epochs=120, batch=128)
        p.update(self.params)
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.enc = DenseEncoder(self.cat_cols)
        A = torch.tensor(self.enc.fit_transform(X, y), dtype=torch.float32)
        self.ymu = float(np.mean(y))
        t = torch.tensor(y - self.ymu, dtype=torch.float32)
        mods, d = [], A.shape[1]
        for _ in range(p["layers"]):
            mods += [nn.Linear(d, p["hidden"]), nn.SiLU(), nn.Dropout(p["dropout"])]
            d = p["hidden"]
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["wd"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=p["epochs"] * (len(A) // p["batch"] + 1))
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(p["epochs"]):
            self.net.train()
            perm = torch.randperm(len(A), generator=g)
            for i in range(0, len(A), p["batch"]):
                b = perm[i:i + p["batch"]]
                loss = ((self.net(A[b]).squeeze(1) - t[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            A = torch.tensor(self.enc.transform(X), dtype=torch.float32)
            return self.net(A).squeeze(1).numpy() + self.ymu


def _tfm_matrix(enc_holder, X, y=None):
    """Numeric matrix for tabular foundation models: median-imputed numerics + ordinal cat codes + target encodings."""
    if y is not None:
        enc_holder.enc = DenseEncoder(enc_holder.cat_cols, onehot=False, te=enc_holder.params.get("te", True))
        A = enc_holder.enc.fit_transform(X, y)
        enc_holder.codes = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))} for c in enc_holder.cat_cols}
    else:
        A = enc_holder.enc.transform(X)
    if enc_holder.params.get("codes", True):
        C = np.column_stack([X[c].map(enc_holder.codes[c]).fillna(-1).values for c in enc_holder.cat_cols])
        A = np.column_stack([A, C])
    return A.astype(np.float32)


class TabICLModel(Base):
    def fit(self, X, y):
        from tabicl import TabICLRegressor
        from pathlib import Path
        ck = Path(__file__).resolve().parent.parent / "models" / "tabicl" / "tabicl-regressor-v2-20260212.ckpt"
        kw = dict(model_path=str(ck), allow_auto_download=False) if ck.exists() else {}
        self.m = TabICLRegressor(n_estimators=self.params.get("n_estimators", 8), device="cpu",
                                 batch_size=self.params.get("batch_size", 8),
                                 random_state=self.seed, **kw)
        self.m.fit(_tfm_matrix(self, X, y), y)
        return self

    def predict(self, X):
        return self.m.predict(_tfm_matrix(self, X))


class TabDPTModel(Base):
    def fit(self, X, y):
        from tabdpt import TabDPTRegressor
        from pathlib import Path
        w = Path(__file__).resolve().parent.parent / "models" / "tabdpt" / "tabdpt.safetensors"
        self.m = TabDPTRegressor(device="cpu", compile=False, verbose=False,
                                 model_weight_path=str(w) if w.exists() else None,
                                 normalizer=self.params.get("normalizer", "standard"),
                                 missing_indicators=self.params.get("missing_indicators", False))
        self.m.fit(_tfm_matrix(self, X, y), y)
        return self

    def predict(self, X):
        return self.m.predict(_tfm_matrix(self, X), n_ensembles=self.params.get("n_ensembles", 8), seed=self.seed)


class TabPFNModel(Base):
    """TabPFN v2 regressor (Prior Labs License 1.1, Apache-2.0 + attribution). Built with TabPFN."""

    def fit(self, X, y):
        from tabpfn import TabPFNRegressor
        from pathlib import Path
        ck = Path(__file__).resolve().parent.parent / "models" / "tabpfn" / "tabpfn-v2-regressor.ckpt"
        self.m = TabPFNRegressor(n_estimators=self.params.get("n_estimators", 8), model_path=str(ck), device="cpu",
                                 random_state=self.seed, ignore_pretraining_limits=True)
        self.m.fit(_tfm_matrix(self, X, y), y)
        return self

    def predict(self, X):
        return self.m.predict(_tfm_matrix(self, X))


MODELS = {"tabpfn": TabPFNModel, "tabicl": TabICLModel,"tabdpt": TabDPTModel,"cat": CatModel, "lgb": LGBModel, "xgb": XGBModel, "ridge": RidgeModel, "et": ETModel, "mlp": MLPModel}
