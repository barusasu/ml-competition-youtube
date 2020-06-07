import gc   # ガベージコレクタインターフェース
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
# import data_processor as dp

path = 'data/'

train = pd.read_csv(path + "train_data.csv")
test = pd.read_csv(path + "test_data.csv")

# trainとtestをくっつける
target = train['y']
del train['y']
data = pd.concat([train, test])

# 日付データの整形
# 動画の投稿日
# datetime64型に変換
data["publishedAt"] = pd.to_datetime(data["publishedAt"])
# dtアクセで列全体を一括処理
data["year"] = data["publishedAt"].dt.year
data["month"] = data["publishedAt"].dt.month
data["day"] = data["publishedAt"].dt.day
data["hour"] = data["publishedAt"].dt.hour
data["minute"] = data["publishedAt"].dt.minute

# データレコードの収集日
data["collection_date"] = "20" + data["collection_date"]
# datetime64型に変換
data["collection_date"] = pd.to_datetime(data["collection_date"], format="%Y.%d.%m")
data["c_year"] = data["collection_date"].dt.year
data["c_month"]  = data["collection_date"].dt.month
data["c_day"] = data["collection_date"].dt.day

# タグの数を特徴量に入れる
data["length_tags"] = data["tags"].astype(str).apply(lambda x: len(x.split("|")))

data = data.drop(["channelId",
                "video_id",
                "publishedAt",
                "thumbnail_link",
                "channelTitle",
                "collection_date",
                "id",
                "tags",
                "description",
                "title"],axis=1)


# 分割
train = data.iloc[:len(target), :]
test = data.iloc[len(target):, :]

# Kfold
cv = KFold(n_splits=5, shuffle=True, random_state=123)

# RMSLE用
score = 0

# testデータの予測用
pred = np.zeros(test.shape[0])

target = np.log(target)

params = {
    'boosting_type' : 'gbdt',
    'metric' : 'rmse',
    'objective' : 'regression',
    'seed' : 20,
    'learning_rate' : 0.01,
    'n_jobs' : -1,
    'verbose' : -1
}

for tr_idx, val_idx in cv.split(train):
    x_train, x_val = train.iloc[tr_idx], train.iloc[val_idx]
    y_train, y_val = target[tr_idx], target[val_idx]

    # Datasetに入れて学習させる
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val, reference=train_set)

    # Training
    model = lgb.train(params, train_set, num_boost_round=8000, early_stopping_rounds=100,
                    valid_sets=[train_set, val_set], verbose_eval=500)
    
    # 予測したらexpで元に戻す
    test_pred = np.exp(model.predict(test))
    # 0より小さな値があるとエラーになるので補正
    test_pred = np.where(test_pred < 0, 0, test_pred)
    pred += test_pred / 5 # 5Fold回すので

    oof = np.exp(model.predict(x_val))
    oof = np.where(oof < 0, 0, oof)
    rmsle = np.sqrt(mean_squared_log_error(np.exp(y_val), oof))
    print(f"RMSLE : {rmsle}")
    score += rmsle / 5

lgb.plot_importance(model, importance_type="gain", max_num_features=20)

print(f"Mean RMSLE SCORE :{score}")

submit_df = pd.DataFrame({"y": pred})
submit_df.index.name = "id"
submit_df.to_csv("submit.csv")
