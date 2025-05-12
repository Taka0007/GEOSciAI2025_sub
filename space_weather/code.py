## install
!pip -q install lightgbm optuna joblib

## import
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna, joblib, os, gc, warnings
warnings.filterwarnings("ignore")
## set seed
SEED = 777
np.random.seed(SEED)

ROOT = "/content"
train_path = os.path.join(ROOT, "geosciai2025_sw_train.csv")
test_path  = os.path.join(ROOT, "geosciai2025_sw_all_test_timeshift.csv")

def read_csv(fp):
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

train = read_csv(train_path)
test  = read_csv(test_path)

# ターゲットを 24h シフト
train["dst_t+24"] = train["dst (nT)"].shift(-24)
train = train.iloc[:-24]

def add_domain_features(df):
    # Southward IMF (Bs)
    df["Bs"] = df["bz (nT)"].clip(upper=0).abs()

    # IMF Clock 角と BT
    df["BT"] = np.sqrt(df["by (nT)"]**2 + df["bz (nT)"]**2)
    df["theta_c"] = np.degrees(np.arctan2(df["BT"], df["bx (nT)"]))

    # Akasofu ε [W]
    μ0 = 4*np.pi*1e-7
    df["epsilon"] = 1e3 * (df["vsw (km/s)"] * (df["BT"]**2) * (np.sin(np.radians(df["theta_c"])/2)**4)) / μ0

    # Newell coupling
    df["phi_newell"] = (df["vsw (km/s)"]**(4/3)) * (df["BT"]**(2/3)) * (np.sin(np.radians(df["theta_c"])/2)**(8/3))

    # Ey (mV/m) 再計算 (= -V * Bz * 1e-3)
    df["Ey_calc"] = -1e-3 * df["vsw (km/s)"] * df["bz (nT)"]

    # Burton Q (注入率) と Pdyn 補正項
    df["Q_burton"] = 0.20 * df["Ey_calc"] - 1.5
    df["Dst_star"] = df["dst (nT)"] - 7.26 * np.sqrt(df["pdyn (nPa)"]) + 11  # Dst* (Burton)

    # Plasma β・Alfven マッハ数そのまま使用
    return df

train = add_domain_features(train)
test  = add_domain_features(test)

def make_temporal_features(df, lags=24, windows=(3,6,12)):
    feature_cols = []
    for lag in range(1, lags+1):
        for col in ["dst (nT)", "bz (nT)", "Bs", "Ey_calc", "phi_newell", "Q_burton",
                    "vsw (km/s)", "pdyn (nPa)", "nsw (n/cc)"]:
            new_col = f"{col}_lag{lag}"
            df[new_col] = df[col].shift(lag)
            feature_cols.append(new_col)
    for w in windows:
        for col in ["Bs", "Ey_calc", "phi_newell"]:
            for stat, func in zip(["mean","std","min","max"],
                                  [np.mean,np.std,np.min,np.max]):
                new_col = f"{col}_{stat}{w}"
                df[new_col] = df[col].rolling(window=w, min_periods=1).apply(func)
                feature_cols.append(new_col)
    return df, feature_cols

train, F_cols  = make_temporal_features(train)
test,  _        = make_temporal_features(test)

# 欠損補完（先頭分は線形補間→残りは0）
train[F_cols] = train[F_cols].interpolate(limit_direction="both").fillna(0)
test[F_cols]  = test[F_cols].interpolate(limit_direction="both").fillna(0)

TARGET = "dst_t+24"
FEATS  = F_cols + ["Bs","BT","theta_c","epsilon","phi_newell",
                   "Ey_calc","Q_burton","Dst_star","beta","amach","mmach",
                   "sunspot","f10.7","lymana (W/m^2)","kp10","ap (nT)"]

# TimeSeriesSplit → 5 fold Walk-Forward (2018 以降を val に近くなるように)
tscv = TimeSeriesSplit(n_splits=5, test_size=24*365)  # 1 年 ≒ 8760 h
X, y = train[FEATS], train[TARGET]

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": SEED,
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 128, 1024),
        "max_depth": trial.suggest_int("max_depth", 6, 16),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.1, log=True),
        "lambda_l1": trial.suggest_float("l1", 0, 5),
        "lambda_l2": trial.suggest_float("l2", 0, 5),
        "min_data_in_leaf": trial.suggest_int("min_data", 50, 300),
    }
    rmses = []
    for train_idx, val_idx in tscv.split(X):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        dval   = lgb.Dataset(X.iloc[val_idx],  label=y.iloc[val_idx])
        model = lgb.train(params, dtrain, num_boost_round=5000,
                          valid_sets=[dval])
        pred = model.predict(X.iloc[val_idx])
        rmses.append(mean_squared_error(y.iloc[val_idx], pred))
    return np.mean(rmses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, timeout=60*20)
best_params = study.best_params
best_params.update({"objective":"regression","metric":"rmse","verbosity":-1,"seed":SEED})

print("Best CV RMSE:", study.best_value)
print("Best params :", best_params)

dall = lgb.Dataset(X, label=y)
best_iter = int(study.best_trial.user_attrs.get("best_iteration", 4000))
model = lgb.train(best_params, dall, num_boost_round=best_iter)

imp = pd.Series(model.feature_importance(), index=FEATS).sort_values(ascending=False).head(30)
print("\nTop-30 Importance\n", imp)

test_pred = model.predict(test[FEATS])

sub = pd.DataFrame({
    "datetime": test["datetime"],
    "dst_pred": test_pred
})
sub.to_csv("submission_dst.csv", index=False)
print("submission_dst.csv saved, head:\n", sub.head())

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import numpy as np, lightgbm as lgb, pandas as pd

def cv_rmse(data, feat_cols, target="dst_t+24", n_split=5, seed=42):
    tscv = TimeSeriesSplit(n_splits=n_split, test_size=24*365)
    X, y = data[feat_cols], data[target]
    rmses = []
    for tr_idx, va_idx in tscv.split(X):
        train_ds = lgb.Dataset(X.iloc[tr_idx], label=y.iloc[tr_idx])
        valid_ds = lgb.Dataset(X.iloc[va_idx], label=y.iloc[va_idx])
        model = lgb.train(
            {"objective":"regression","metric":"rmse","seed":seed},
            train_ds, num_boost_round=2000,
            valid_sets=[valid_ds],
        )
        pred = model.predict(X.iloc[va_idx])
        rmses.append(mean_squared_error(y.iloc[va_idx], pred))
    print(f"CV RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    return np.array(rmses)

_ = cv_rmse(train, FEATS)
