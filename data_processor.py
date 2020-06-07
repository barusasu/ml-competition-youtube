import pandas as pd
import numpy as np

path = './data'

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
data["c_day"] = data["collection_data"].dt.day

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



