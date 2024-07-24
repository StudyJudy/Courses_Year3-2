import pandas as pd
import os
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

########################################################################
#                下面的路径参数可以改，其他的参数不要动
########################################################################
# 数据路径
kernel_log_data_path = "memory_sample_kernel_log_round1_a_train.csv"
address_log_data_path = "memory_sample_address_log_round1_a_train.csv"
mce_log_data_path = "memory_sample_mce_log_round1_a_train.csv"
failure_tag_data_path = "memory_sample_failure_tag_round1_a_train.csv"
PARENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 模型路径
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.dat")
########################################################################


# 计算每个agg_time区间的和
def etl(path, agg_time):
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data["collect_time"] = pd.to_datetime(data["collect_time"]).dt.ceil(agg_time)
    group_data = data.groupby(["serial_number", "collect_time"], as_index=False).agg(
        "sum"
    )
    return group_data


def train():
    # 设置聚合时间粒度
    AGG_VALUE = 7
    AGG_UNIT = "min"
    AGG_TIME = str(AGG_VALUE) + AGG_UNIT

    # 为数据打标
    failure_tag = pd.read_csv(os.path.join(PARENT_FOLDER, failure_tag_data_path))
    failure_tag["failure_time"] = pd.to_datetime(failure_tag["failure_time"])

    # kernel 数据
    group_min_kernel = etl(kernel_log_data_path, AGG_TIME)
    merged_data = pd.merge(
        group_min_kernel,
        failure_tag[["serial_number", "failure_time"]],
        how="left",
        on=["serial_number"],
    )
    merged_data.describe()

    merged_data["failure_tag"] = (merged_data["failure_time"].notnull()) & (
        (merged_data["failure_time"] - merged_data["collect_time"]).dt.seconds
        <= AGG_VALUE * 60
    )
    merged_data["failure_tag"] = merged_data["failure_tag"] + 0
    feature_data = merged_data.drop(
        ["serial_number", "collect_time", "manufacturer", "vendor", "failure_time"],
        axis=1,
    )

    # 负样本下采样
    sample_0 = feature_data[feature_data["failure_tag"] == 0].sample(frac=0.1)
    sample = sample_0._append(feature_data[feature_data["failure_tag"] == 1])
    sample.describe()

    # 分割数据集
    X = sample.iloc[:, :-1]
    Y = sample["failure_tag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=7
    )

    # 训练模型
    model = XGBClassifier(base_score=0.66)
    model.fit(X_train, y_train)

    # 保存模型
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
