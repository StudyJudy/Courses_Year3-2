import random
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from errortools import Evaluate,maxError,minError,meanSqErr,meanError

# 该值是用于确定算法启动模式，为True时会读取/Dataset文件夹下的数据集，为False时使用随机数据
# 真实数据的好处是来自现实世界，是实际有效的数据，更有意义
# 随机数据的好处是分布更加均匀
USING_INPUT_DATA = True

# B:原始数据
# Q:查询集
# T:迭代次数
# eps:隐私预算
# repetitions:乘法权重算法的重复次数
def MWEM(B, Q, T, eps, repetitions):
    # 生成真实的直方图，其逻辑是，找到数据集中有效属性的最大值和最小值
    # 然后以其差值生成一个拥有差值+1个桶的直方图，以列表形式存储
    minVal = min(B)
    length = max(B) - minVal + 1
    histogram = [0]*(length)
    for val in range(len(B)):
        histogram[B[val] - minVal] += 1

    # 初始化数据合成流程
    nAtt = 1 # 合成算法作用的属性数量。设计上，本算法支持更多维度的运算，但考虑到实现复杂度和运行时间，此处设定为1
    A = 0
    n = 0
    # 生成一个分布平均的数据分布作为初始分布
    n = sum(histogram)
    A = [n/len(histogram) for i in range(len(histogram))]
    measurements = { } #mesurements是一个dict类型，其初始化时应该以{ }初始化
    # 迭代优化循环体
    for i in range(T):
        print("ITERATION ROUND#" + str(i))
        # 选定本轮要优化的查询，同时，如果出现与此前一致的查询则再运行一次算法，直到选中从未优化过的查询
        # mesurements:已观测的查询，将查询收入此以避免重复优化
        qi = ExpM(histogram,A,Q,(eps /(2*T)))
        j = 1
        print("选中了"+str(Q[qi]))
        while(qi in measurements):
            qi = ExpM(histogram, A, Q, eps / (2*T))
            print("选定的值已被观测过，启动重选，重选第"+str(j)+"次")
            j+=1
            print("选中了"+str(Q[qi]))

        # 对查询值进行观测并加入已观测的列表中
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap

        # 乘法权重更新开始
        MultiplicativeWeights(A, Q, measurements, repetitions)
        # print(A)
    return A, histogram

# B:原始数据
# A:合成数据
# Q:查询集
# T:迭代次数
# eps:隐私预算
# ExpM-使用指数机制来选定一个查询
# 本部分算法原理：首先测量对原始数据集和合成数据运行同一个查询时其之间的误差作为打分函数，
# 根据各查询得分的多少计算出每个查询被选中的概率，之后基于这些概率随机抽取一个查询。
def ExpM(B, A, Q, eps):
    # 初始化一个Q长度的打分列表。其他类似初始化方式不再赘述。
    scores = [0] * len(Q)
    for i in range(len(scores)):
        # 打分函数。此处我们使用的打分原始来源是原数据和估计数据在响应同一个查询时差异的绝对值。
        # 但是由于绝对误差太大，会造成指数计算越界，所以此处需要除以100000来缩小错误的规模
        # 使用该方法对得分的相对差距是不会有影响的
        scores[i] = abs(Evaluate(Q[i], B) - Evaluate(Q[i], A))/100000
    # 基于分数计算每个查询被选中的概率
    Pr = [np.exp(eps * score / 2 ) for score in scores]
    # 规范化概率，使其和为1
    Pr = Pr / np.linalg.norm(Pr, ord=1)
    # 基于生成的概率表抽取查询
    return np.random.choice(len(Q), 1, p=Pr)[0]

# 利用numpy的随机数模块来生成拉普拉斯随机数作为噪音
def Laplace(sigma):
    return np.random.laplace(loc=0,scale=sigma)

# A:合成数据
# Q:查询集
# measurements:已观测过的查询
# repetitions:重复次数
# 本函数实现了乘法权重更新
def MultiplicativeWeights(A, Q, measurements, repetitions):
    total = sum(A)
    # 这个地方，会有一个内联的循环体，用于重复地更新多次乘法权重
    for iteration in range(repetitions):
        # 对所有观测值进行更新。根据原始Julia，此处先对measurements生成一个随机的更新顺序
        update_order = [qi for qi in measurements]
        # 利用洗牌算法打乱更新顺序
        random.shuffle(update_order)
        # 根据新生成的访问顺序来对乘法权重执行更新
        for qi in update_order:
            # 观测值和合成数据间的误差计算。此处注意，不能再使用绝对值
            # 这是因为，MW流程需要根据误差的正负来升/降已有观测的权重
            error = measurements[qi] - Evaluate(Q[qi], A)
            # 乘法权重更新
            # 首先将查询转换为二进制形式，用于将所有在属于该查询区间内的分布值更新
            query = queryToBinary(Q[qi], len(A))
            for i in range(len(A)): 
                # 此处公式参考原始Julia。注意，原始实现这里不需要多除一个total，但是此处需要一个值来降低实际error以防越界                
                A[i] = A[i] * math.exp(query[i] * error/(2.0*total))      
            # 重新归一化，保证更新后的权重在有效区间内
            count = sum(A)
            for k in range(len(A)):
                A[k] *= total/count


# Histo:matplotlib.plot方法所需的格式
# B:输入的数据集
# 转换查询为生成直方图所用的Plot
# 运行逻辑：确定输入数据集B的最小值后，从最小值开始，先对输入的值进行四舍五入，然后生成样本点，其个数等于四舍五入后的值
# 四舍五入的原因是加噪后的数据集内可能包含小数
def transformForPlotting(Histo, B):
    start = min(B)
    end = max(B)
    newHisto = []
    for i in range(len(Histo)):
        val = int(round(Histo[i],0))  
        for j in range(val):
            newHisto.append(start)
        start = start + 1
    return newHisto


# qi:输入的查询
# length:数据集属性长度
# 转换查询为二进制形式，用于MW机制使用
# 运行逻辑：MW机制是对符合区间的桶进行乘法权重更新，这里使用一个binary序列来代表数据集的桶是否属于区间。
# 首先，将binary全部置0，然后将输入的查询区间内的桶对应的binary值变为1。这样的话，根据这个binary值，
# MW机制就能只更新所选查询影响到的桶了。
def queryToBinary(qi,length):
    binary = [0]*length
    for i in range(length):
        if((i>=qi[0]) and i<=qi[1]):
            binary[i] = 1
    return binary


def histoDraw(real,syn,B):
    # 启动直方图绘制
    plt.figure()
    # 定桶数量
    bins = np.linspace(min(B), max(B)+1, max(B)+2)
    # Plot化数据
    HistA = transformForPlotting(real, B)
    HistB = transformForPlotting(syn, B)    
    # 制图和保存
    plt.hist([HistA,HistB],bins)
    plt.savefig("./Results/result_histo.png")
    plt.show()
    
    # 绘制折线图
    plt.plot(real)
    plt.plot(syn, color = 'red')
    plt.savefig("./Results/result_normal.png")


# 随机生成Q_size个查询，并对查询集中的已有查询规避
# 运行逻辑：首先生成一个空列表，然后根据数据集的最大值和最小值，生成两个随机数作为查询上下界。
# 下界down介于0到最大值减去最小值之间，而上界upper介于下界和最大值减去最小值之间。
# 之后，再判断生成的上下界对是否已经生成过，若生成则重新生成，直到不再重复。
def randomQueries(Q_size, maxVal, minVal):
    Q = []
    count_regen = 0 
    for i in range (Q_size):
        down = random.randint(0,maxVal-minVal)
        upper = random.randint(down,maxVal-minVal)
        while (down,upper) in Q:
             down = random.randint(0,maxVal-minVal)
             upper = random.randint(down,maxVal-minVal)
             count_regen+=1
        Q.append((down,upper))
    print("查询集随机生成完毕，生成了"+str(Q_size)+"个查询，共进行规避生成"+str(count_regen)+"次")
    return Q


def main():
    B = []  # 一维数据，使用list即可
    Q_size = 60 # 查询集个数
    T = 30 # T是迭代运行次数。要注意的是，该处迭代次数不应大于上一步所生成的查询总数。
    eps = 0.1 # eps即epsilon，隐私预算。
    repetitions = 20 # MW机制的重复次数，一般不需更改

    # 使用外部数据时，该部分即为读取指定路径的测试数据。
    if USING_INPUT_DATA == True:
        with open('Datasets/childMentalHealth_1M.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                    try:
                        B.append(int(row[0]))
                    except ValueError as e:
                        continue
                    except IndexError as e:
                        continue
    # 保留了一个测试模式：当不指定输入数据集时，允许随机生成500个记录来进行测试
    else:
        B = [random.randint(0,50) for i in range(500)] #Dataset

    maxVal = max(B)
    minVal = min(B)
    # 随机生成Q_size个查询，并对查询集中的已有查询规避
    Q = randomQueries(Q_size, maxVal, minVal)
   
    syntheticData, RealHisto = MWEM(B, Q, T, eps, repetitions)
    # 格式化地输出合成数据到屏幕
    formattedList = ['%.3f' % elem for elem in syntheticData]
    print()
    print("Real data histogram: " + str(RealHisto))
    print()
    print("Synthetic Data: " + str(formattedList))
    print()
    print("Q: " + str(Q))
    print()
    print("Metrics:")
    print("    - MaxError: " + str(maxError(RealHisto,syntheticData, Q)))
    print("    - MinError: " + str(minError(RealHisto,syntheticData, Q)))
    print("    - MeanSquaredError: " + str(meanSqErr(RealHisto, syntheticData, Q)))
    print("    - MeanError: " + str(meanError(RealHisto,syntheticData, Q)))
    histoDraw(RealHisto, syntheticData, B)


main()
