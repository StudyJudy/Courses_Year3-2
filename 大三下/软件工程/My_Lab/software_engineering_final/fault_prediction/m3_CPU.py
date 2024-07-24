from __future__ import print_function
import pandas as pd
import pmdarima as pm
import os


########################################################################
#                下面的路径参数可以改，其他的参数不要动
########################################################################
# 数据路径
train_data_path = "train.csv"
test_data_path = "test.csv"
PARENT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
########################################################################


def process_csv(file_path):
    # 读取csv文件
    df = pd.read_csv(file_path, header=None, names=["datetime", "cpu"])

    # 提取分钟数
    df["minutes"] = df["datetime"].apply(lambda x: int(x[-2:]))

    # 取出分钟数和对应的cpu值
    minutes = df["minutes"].tolist()
    cpu_values = df["cpu"].tolist()

    return minutes, cpu_values


def cpu_get_result(str, r_x, r_y):
    """
    参数说明：
    str:这个代表了目前从开机到现在的分钟数,也就是说str=0代表开机的时候,str=1代表过了1分钟
        因为我们是打算一分钟一次嘛，所以就是需要传入这个值，然后你处理一下就行了，如果说你
        在调用的时候没有这个值,那麻烦你自己处理一下,同时str如果不是一分钟的话,就是后期改了的话
        你自己设置一个标准就好了
    r_x:传回的上一个时刻的res_x,如果str==0的话,直接传一个[]就行,因为str==0不会操作
    r_y:传回的上一个时刻的res_y,如果str==0的话,直接传一个[]就行,因为str==0不会操作
    返回值：
    res_x:这个是一个长度为70的list,形式为[0,1,2,3,4,5,6,7,8,9,.........]
          然后这个里面,前60个代表了历史的cpu利用率对应的时间,后面的10个代表了未来的时间
          这里需要说明一下,我直接截取了分钟数,比如说16:01,我只取了01这个分钟数,所以里面的数是01
          然后呢,后面的10个时间是接着前面的时间往后推的,比如说前面的时间是59,那后面的时间就是00,01,02,03,04,05,06,07,08,09
    res_y:这个是一个长度为70的list,形式为[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,.........]
          这个里面,前60个代表了历史的cpu利用率,后面的10个代表了未来的cpu利用率
    error:这个是一个bool值,代表了这个时刻是否故障,如果说故障了,那就是True,否则就是False
    """
    res_x = []
    res_y = []
    error = False

    train_time, train_cpu = process_csv(os.path.join(PARENT_FOLDER, train_data_path))
    test_time, test_cpu = process_csv(os.path.join(PARENT_FOLDER, train_data_path))

    # 故障检测，如果预测的cpu利用率和当前时间点的cpu利用率相差小于0.1，则认为是故障
    if str >= 1:
        error = r_y[60] - test_cpu[str - 1] < 0.1

    for i in range(min(str, 70)):
        train_time.append(test_time[i])
        train_cpu.append(test_cpu[i])

    for i in range(60):
        res_x.append(train_time[-60 + i])
        res_y.append(train_cpu[-60 + i])

    model = pm.auto_arima(
        train_cpu,
        start_p=1,
        start_q=1,
        max_p=8,
        max_q=8,
        m=1,
        start_P=0,
        seasonal=False,
        max_d=3,
        trace=True,
        information_criterion="aic",
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    forecast = model.predict(10)

    last_time = train_time[-1]
    for value in forecast:
        last_time = (last_time + 1) % 60
        res_x.append(last_time)
        res_y.append(value)

    return res_x, res_y, error
