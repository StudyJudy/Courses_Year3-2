import os
import pickle
import pandas as pd


########################################################################
#                下面的路径参数可以改，其他的参数不要动
########################################################################
# 数据路径
PARENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
kernel_log_test_data_path = "memory_sample_kernel_log_round1_a_test.csv"
address_log_test_data_path = "memory_sample_address_log_round1_a_test.csv"

# 模型路径
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.dat")
########################################################################


AGG_VALUE = 7
AGG_UNIT = "min"
AGG_TIME = str(AGG_VALUE) + AGG_UNIT

batch = 400


def etl(path, agg_time):
    data = pd.read_csv(os.path.join(PARENT_FOLDER, path))
    data["collect_time"] = pd.to_datetime(data["collect_time"]).dt.ceil(agg_time)
    group_data = data.groupby(["serial_number", "collect_time"], as_index=False).agg(
        "sum"
    )
    return group_data


def dram_get_result(str):
    """
    参数说明：
        str:这个代表了目前从开机到现在的分钟数,也就是说str=0代表开机的时候,str=1代表过了1分钟
        因为我们是打算一分钟一次嘛，所以就是需要传入这个值，然后你处理一下就行了，如果说你
        在调用的时候没有这个值,那麻烦你自己处理一下,同时str如果不是一分钟的话,就是后期改了的话
        你自己设置一个标准就好了
    返回值：
        err_res:这个是一个[]类型的数组，其内结构类似于['server_5074', 'server_50781', 'server_5097']
                这样,当然它有可能是空的,其内装的值就是在输入的batch(这个变量看上面)个server的报告日志中
                预测到的会发生故障的server名称
    模拟说明：
    因为每个str采集一次,也就是需要计算一次,模拟的过程是每个str中读入batch个server的报告日志(来源于test集)
    然后假装我们这一分钟内有这些个服务器报告了日志，然后来预测
    """
    err_res = []

    kernel_log_test = etl(kernel_log_test_data_path, AGG_TIME)
    address_log_test = etl(address_log_test_data_path, AGG_TIME)
    group_data_test = pd.merge(
        address_log_test,
        kernel_log_test,
        how="inner",
        on=["serial_number", "collect_time"],
    )

    # 按照"serial_number"列进行分组
    grouped = group_data_test.groupby("serial_number")

    # 获取所有不同的"serial_number"
    unique_serial_numbers = group_data_test["serial_number"].unique()

    # 每次读入batch个不同的"serial_number"的所有行，作为模拟一次采集的输入
    group_data_test = grouped.get_group(unique_serial_numbers[0])

    serial_numbers_batch = unique_serial_numbers[
        str % 11 * batch : (str % 11 + 1) * batch
    ]
    for serial_number in serial_numbers_batch:
        group = grouped.get_group(serial_number)
        group_data_test = pd.concat([group_data_test, group], axis=0)

    group_min_sn_test = pd.DataFrame(group_data_test[["serial_number", "collect_time"]])
    group_min_test = group_data_test.drop(
        [
            "serial_number",
            "collect_time",
            "memory",
            "rankid",
            "bankid",
            "row",
            "col",
            "manufacturer_y",
            "vendor_y",
            "manufacturer_x",
            "vendor_x",
        ],
        axis=1,
    )

    # 模型预测
    model = pickle.load(open(model_path, "rb"))
    res = model.predict(group_min_test)
    group_min_sn_test["predict"] = res

    # 处理并保存结果
    group_min_sn_test = group_min_sn_test[group_min_sn_test["predict"] == 1]
    group_min_sn_res = group_min_sn_test.drop("predict", axis=1)
    group_min_sn_res = group_min_sn_res.drop("collect_time", axis=1)
    group_min_sn_res = group_min_sn_res.drop_duplicates(
        subset="serial_number", keep="first"
    )

    for i in range(0, len(group_min_sn_res)):
        err_res.append(group_min_sn_res.iloc[i, 0])

    return err_res
