########################################################################
#                     这是调用功能模块3的对外接口
#                             有问题找我
########################################################################
#下面是导入其他文件的函数
#你根据你的IDE自己改就行了
from .m3_DRAM_model import *
from .m3_DRAM_predict import *
from .m3_CPU import *

import time

# 模型路径
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.dat")

def m3init():
    import os
    if os.path.exists(model_path) == False:
        train()


#下面的函数只是示例和说明，具体你自己改
def main():
    #这里需要说一下，每个时间刻度下，目前商量的好像是1分钟，就是1分钟采集一次更新一次
    #每个刻度都需要运行一次该函数
    #下面的初始化运行一次就够了，主要是需要训练好一个模型
    #提示：如果你修改了路径，需要4个文件全改，保持一致
    m3init()
    #下面的这个str就是第几次，表示的就是第几分钟（我记得是咱们说过是1分钟采集一次更新一次来着），就是第几次这个意思
    #你可以换别的变量，但是就是这么个意思和用法
    str=0
    #下面是预测CPU利用率的
    #参数的说明详见cpu_get_result这个函数
    #如果可以的话，我希望能够在最后的ui界面上开一个框，上面画这个xy的曲线图
    #我觉得比较好看，然后xy就是下面的xy，然后预测的值换成红色，原来的是蓝色，这种，比较好看
    cpu_res_x=[]
    cpu_res_y=[]
    cpu_err=False
    cpu_res_x,cpu_res_y,cpu_err=cpu_get_result(str,cpu_res_x,cpu_res_y)
    #下面是预测服务器内存DRAM故障与否的
    #参数的说明详见dram_get_result这个函数
    #然后返回值的说明都在函数说明里了，你直接输出这个字符串就行
    #我返回的是服务器的名称，至于怎么排版和显示，你自己决定
    server_error_name=[]
    server_error_name=dram_get_result(str)

if __name__ == '__main__':
    main()