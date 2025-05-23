!pip -q install lightgbm optuna joblib

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna, joblib, os, gc, warnings
warnings.filterwarnings("ignore")

SEED = 777
np.random.seed(SEED)

ROOT = "/content"
train_path = os.path.join(ROOT, "geosciai2025_sw_train.csv")
test_path  = os.path.join(ROOT, "geosciai2025_sw_all_test_timeshift.csv")

def read_csv(fp):
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

train_raw = read_csv(train_path)
test      = read_csv(test_path)

## 訓練データのカットオフ(2019-12-31 0:00まで)
cutoff = pd.Timestamp("2019-12-31 00:00", tz="UTC")
train = train_raw.loc[train_raw["datetime"] <= cutoff].copy()

# 24 h 先をターゲットに
train["dst_t+24"] = train["dst (nT)"].shift(-24)
train = train.iloc[:-24]

train.head(3)

train.tail(3)

def add_domain_features(df):
    df["Bs"] = df["bz (nT)"].clip(upper=0).abs()
    df["BT"] = np.sqrt(df["by (nT)"]**2 + df["bz (nT)"]**2)
    df["theta_c"] = np.degrees(np.arctan2(df["BT"], df["bx (nT)"]))
    μ0 = 4*np.pi*1e-7
    df["epsilon"] = 1e3 * (df["vsw (km/s)"] * (df["BT"]**2) * (np.sin(np.radians(df["theta_c"])/2)**4)) / μ0
    df["phi_newell"] = (df["vsw (km/s)"]**(4/3)) * (df["BT"]**(2/3)) * (np.sin(np.radians(df["theta_c"])/2)**(8/3))
    df["Ey_calc"] = -1e-3 * df["vsw (km/s)"] * df["bz (nT)"]
    df["Q_burton"] = 0.20 * df["Ey_calc"] - 1.5
    df["Dst_star"] = df["dst (nT)"] - 7.26 * np.sqrt(df["pdyn (nPa)"]) + 11
    return df

def add_fourier_features(df: pd.DataFrame,
                         periods_hours=(24, 12, 6, 648, 8766),
                         order=1,
                         time_col="datetime") -> list:
    # 連続時間軸 (単位 = hour) を取得
    t = (df[time_col] - df[time_col].min()).dt.total_seconds() / 3600.0
    new_cols = []
    for P in periods_hours:
        for k in range(1, order + 1):
            w = 2 * np.pi * k / P
            sin_col = f"sin_{P}h_k{k}"
            cos_col = f"cos_{P}h_k{k}"
            df[sin_col] = np.sin(w * t)
            df[cos_col] = np.cos(w * t)
            new_cols.extend([sin_col, cos_col])
    return new_cols

train = add_domain_features(train)
test  = add_domain_features(test)

fourier_cols = add_fourier_features(train, order=1)
_            = add_fourier_features(test,  order=1)

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

# 欠損補完（先頭分は線形補間→残り 0）
train[F_cols] = train[F_cols].interpolate(limit_direction="both").fillna(0)
test[F_cols]  = test[F_cols].interpolate(limit_direction="both").fillna(0)

TARGET = "dst_t+24"
FEATS  = F_cols + ["Bs","BT","theta_c","epsilon","phi_newell",
                   "Ey_calc","Q_burton","Dst_star","beta","amach","mmach",
                   "sunspot","f10.7","lymana (W/m^2)","kp10","ap (nT)"] + fourier_cols

# TimeSeriesSplit → 5 fold Walk-Forward (2018 以降を val に近くなるように)
tscv = TimeSeriesSplit(n_splits=5, test_size=24*365)  # 1 年 ≒ 8760 h
X, y = train[FEATS], train[TARGET]

## 特徴量の総数を出力
print(f"Total feature count: {len(FEATS)}")

def objective(trial):
    params = {
        "objective":"regression",
        "metric":"rmse",
        "verbosity":-1,
        "boosting_type":"gbdt",
        "seed":SEED,
        "feature_fraction": trial.suggest_float("ff",0.6,1.0),
        "num_leaves":       trial.suggest_int("leaves",128,1024),
        "max_depth":        trial.suggest_int("depth",6,16),
        "learning_rate":    trial.suggest_float("lr",0.01,0.1,log=True),
        "lambda_l1":        trial.suggest_float("l1",0,5),
        "lambda_l2":        trial.suggest_float("l2",0,5),
        "min_data_in_leaf": trial.suggest_int("min_data",50,300),
    }
    rmses=[]
    for tr_idx,va_idx in tscv.split(X):
        m=lgb.train(params,lgb.Dataset(X.iloc[tr_idx],label=y.iloc[tr_idx]),
                    num_boost_round=3000,
                    valid_sets=[lgb.Dataset(X.iloc[va_idx],label=y.iloc[va_idx])])
        rmses.append(root_mean_squared_error(y.iloc[va_idx],m.predict(X.iloc[va_idx])))
    return np.mean(rmses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, timeout=60*30)
best_params = study.best_params
best_params.update({"objective":"regression",
                    "metric":"rmse",
                    "verbosity":-1,
                    "seed":SEED})

print("Best CV RMSE:",study.best_value)
print("Best params :", best_params)

dall = lgb.Dataset(X, label=y)
best_iter = int(study.best_trial.user_attrs.get("best_iteration", 4000))
model = lgb.train(best_params, dall, num_boost_round=best_iter)

imp = pd.Series(model.feature_importance(), index=FEATS).sort_values(ascending=False).head(20)
print("\nTop-20 Importance\n", imp)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(imp.index, imp.values)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 20 Feature Importance")
plt.show()

test_pred_all = model.predict(test[FEATS])

mask = (test["datetime"] >= "2024-05-02") & (test["datetime"] <= "2024-05-31 23:00")
y_true = test.loc[mask, "dst (nT)"].values
y_pred = test_pred_all[mask]

rmse_test = root_mean_squared_error(y_true, y_pred)
print(f"Test RMSE 2024-05-02–31: {rmse_test:.3f} nT")

plt.figure(figsize=(12,5))
plt.plot(test.loc[mask,"datetime"], y_true, label="True Dst", lw=1.5)
plt.plot(test.loc[mask,"datetime"], y_pred, label="Predicted Dst", lw=1.5)
plt.title("Dst 24 h Forecast: 2024-05-02–31")
plt.ylabel("Dst [nT]")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

sub = pd.DataFrame({"datetime": test["datetime"], "dst_pred": test_pred_all})
sub.to_csv("submission_dst2_1.csv", index=False)

sub_2 = pd.DataFrame(
    {
        "datetime": test["datetime"],
        "dst_pred": test_pred_all,
        "dst_true": test["dst (nT)"]
        })

sub_2.to_csv("submission_dst_Part2_2.csv", index=False)
