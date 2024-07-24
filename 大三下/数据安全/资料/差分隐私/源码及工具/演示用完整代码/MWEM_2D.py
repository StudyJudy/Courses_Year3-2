import random
import math
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 该值是用于确定算法启动模式，为True时会读取/Dataset文件夹下的数据集，为False时使用随机数据
# 真实数据的好处是来自现实世界，是实际有效的数据，更有意义
# 随机数据的好处是分布更加均匀
USING_INPUT_DATA = True

# B:输入数据
# rows：拆分出的行信息
# colums: 拆分出的列信息
# minRow:行最小值
# minCol:列最小值
# 从原始的MWEM文件我们可知，1D数据可以看做是一维的数据，所以只需要一个列表
# 而2D数据是二维的，所以我们需要创造一个矩阵。
# 实际上，这个函数就是将行、列的分桶情况组合并创造出一个二维的列表（其他语言称之为数组）
# 其格式类似于:histogram[row][column]，而histogram[x][y]相当于是对B执行了一次row=x，column=y的count查询的解
# 对于我们给定的两列数据，这个矩阵规模会是14行4列，因为第一列14种取值，第二列4种
def matrixCreation(B, rows, columns, minRow, minCol):
    # 先初始化一个全0的、规模为row*column的矩阵
    histogram = [[0]*(columns) for i in range(rows)]
    # 启动遍历，将B中数据填进矩阵中 
    for val in range(len(B[0])):
        histogram[B[0][val] - minRow][B[1][val] - minCol] += 1 
    return histogram


# B:原始数据
# Q:查询集
# T:迭代次数
# eps:隐私预算
# repetitions:乘法权重算法的重复次数
def MWEM(B, Q, T, eps, repetitions):
    # 初始化真实数据集对应的直方图
    # 基本逻辑与1D相同，而与1D不同的是，这里要同时考虑行列的最大最小值以确定桶
    minRow = min(B[0])
    minCol = min(B[1])
    rows = max(B[0]) - min(B[0]) + 1 #age
    columns = max(B[1]) - min(B[1]) + 1 #satisfaction
    # 这里的直方图发生了变化，从一个列表变成了矩阵，具体可以参考matrixCreation函数
    histogram = matrixCreation(B,rows,columns,minRow,minCol)

    # 初始化数据合成流程
    nAtt = 2 # 合成算法作用的属性数量
    A = []
    n = 0
    # 初始化一个分布平均的查询作为初始的分布
    # 先算所有行的每行加和，再将这些sum值再次sum，得到总矩阵的和
    # 然后平均分配给矩阵各个单元格
    m = [sum(histogram[i]) for i in range(len(histogram))]
    n = sum(m)
    value = n/(rows*columns)
    A = [[0]*(columns) for i in range(rows)]
    for i in range(len(histogram)):
        for j in range(len(histogram[i])):
            A[i][j] += value

    measurements = {} # esurements是一个dict类型，其初始化时应该以{ }初始化
    # 迭代优化循环体
    for i in range(T):
        print("ITERATION #" + str(i))

        # 选定本轮要优化的查询，同时，如果出现与此前一致的查询则再运行一次算法，直到选中从未优化过的查询
        # mesurements:已观测的查询，将查询收入此以避免重复优化
        qi = ExpM(histogram, A, Q, (eps /(2*T)))

        while(qi in measurements):
            qi = ExpM(histogram, A, Q, eps / (2*T))

        # 对查询值进行观测并加入已观测的列表中
        evaluate = Evaluate(Q[qi],histogram)
        lap = Laplace((2*T)/(eps*nAtt))
        measurements[qi] = evaluate + lap

        # 乘法权重更新开始
        MultiplicativeWeights(A, Q, measurements, repetitions)

    return A, histogram


# B:原始数据
# A:合成数据
# Q:查询集
# T:迭代次数
# eps:隐私预算
# ExpM-使用指数机制来选定一个查询
# 本部分算法原理：首先测量对原始数据集和合成数据运行同一个查询时其之间的误差作为打分函数，
# 根据各查询得分的多少计算出每个查询被选中的概率，之后基于这些概率随机抽取一个查询。
# 【本机制与一维时无变化！】
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
    # 依然是先计算出整个矩阵值的加和
    m = [sum(A[i]) for i in range(len(A))]
    total = sum(m)
    
    # 循环体
    for iteration in range(repetitions): #repetitions = 5, testing
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
            # 这次这个二进制转换器是同时工作在两个维度的了
            query = queryToBinary(Q[qi], len(A[0]), len(A))
            for i in range(len(A)):
                for j in range(len(A[i])):
                    # 其实和1D的区别就在于，1D是一个一维的线性遍历，这里要改成二维的矩阵遍历
                    # 此处公式参考原始Julia。注意，原始实现这里不需要多除一个total，但是此处需要一个值来降低实际error以防越界 
                    A[i][j] = A[i][j] * math.exp(query[i][j] * error/(2.0*total)) #histogram[i][j]

            # 重新归一化，保证更新后的权重在有效区间内
            # 原理和1D类似，只不过同时对矩阵的两个维度一起工作
            m = [sum(A[i]) for i in range(len(A))]
            count = sum(m)
            for k in range(len(A)):
                for l in range(len(A[k])):
                    A[k][l] *= total/count

# 由于2D的算法是提高用内容，我们不再将以下函数独立成一个python文件
# 但是其实下面这几个函数就是errortools.py内函数的二维版

# query:指定的查询
# data:运行查询的数据
# Evaluate函数用于执行查询。其基本逻辑为：
# 对于传入的查询{(x,y):(x,y)}，其累加传入的数据中，值在x和y区间(含x和y)的查询。
def Evaluate(query, data):
    # 查询集Q本身是一个以dict为项目的list，也就是说，每个查询都是一个dict类型数据
    # 这里，我们先将dict类型做list化来正确获取dict的左值(即在行上的查询)，绕开dict无序的限制
    # 然后我们就可以再用获取到的左值来获得dict的右值了，即可将dict还原回两个list的组合
    q_x = list(query)[0]
    q_y = query[q_x]
    counts = 0
    for i in range(q_x[0],q_x[1]+1):
        for j in range(q_y[0],q_y[1]+1):
            counts += data[i][j]
    return counts

# qi:输入的查询
# cols:行长度
# rows：列长度
# 转换查询为二进制形式，用于MW机制使用
# 运行逻辑：MW机制是对符合区间的桶进行乘法权重更新，这里使用一个binary序列来代表数据集的桶是否属于区间。
# 首先，将binary全部置0，然后将输入的查询区间内的桶对应的binary值变为1。这样的话，根据这个binary值，
# MW机制就能只更新所选查询影响到的桶了。
def queryToBinary(qi, cols, rows):
    binary = [[0]*cols for i in range(rows)] 
    # 这个地方操作原理和Evaluate以及1D情况下的ToBinary类似，不再赘述
    q_x = list(qi)[0]
    q_y = qi[q_x]
    for i in range(rows):
        if (i >= q_x[0]) and (i <= q_x[1]):
            for j in range(cols):
                if (j >= q_y[0]) and (j <= q_y[1]):
                    binary[i][j] = 1
    return binary

# real:真实数据
# synthetic:合成数据
# Q:查询集合
# maxError函数用于检测传入的两数据集间的最大差异。
# 该函数会计算传入的两数据集对于查询集Q的响应，并找出最大的差异值（以绝对值计算）
def maxError(real, synthetic, Q):
    maxVal = 0
    diff = 0
    for i in range(len(Q)):
        diff = abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff > maxVal:
            maxVal = diff
    return maxVal

# real:真实数据
# synthetic:合成数据
# Q:查询集合
# meansSqError函数用于检测传入的两数据集间的均方差。
# 该函数会计算传入的两数据集对于查询集Q的响应，计算其均方差。（以绝对值计算）
def meanSqErr(real, synthetic, Q):
    errors = [(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return (np.linalg.norm((errors))**2)/len(errors)

# real:真实数据
# synthetic:合成数据
# Q:查询集合
# minError函数用于检测传入的两数据集间的最小差异。
# 该函数会计算传入的两数据集对于查询集Q的响应，并找出最小的差异值（以绝对值计算）
def minError(real, synthetic, Q):
    minVal = 100000000000
    diff = 0
    for i in range(len(Q)):
        diff = abs(Evaluate(Q[i], real) - Evaluate(Q[i], synthetic))
        if diff < minVal:
            minVal = diff
    return minVal

# real:真实数据
# synthetic:合成数据
# Q:查询集合
# meanError函数用于检测传入的两数据集间的平均差异。
# 该函数会计算传入的两数据集对于查询集Q的响应，并计算差异平均值（以绝对值计算）
def meanError(real, synthetic, Q):
    errors = [abs(Evaluate(Q[i], synthetic) - Evaluate(Q[i], real)) for i in range(len(Q))]
    return sum(errors)/len(errors)


# 随机生成Q_size个查询，并对查询集中的已有查询规避
# 运行逻辑和1D时基本类似，也是先生成两个维度上的查询上下界，然后判断这个查询是否已经出现在已有查询之中了
# 如果在的话则重新生成一个
# 当两个维度值域较小的时候，重新生成的次数可能会非常多
def randomQueries(Q_size, maxVal1, minVal1,maxVal2,minVal2):
    # 此时生成的是一个列表嵌套字典的格式
    Q=[]
    count_regen = 0
    for i in range (Q_size):
        down_x = random.randint(0, maxVal1-minVal1)
        upper_x = random.randint(down_x, maxVal1-minVal1)
        down_y = random.randint(0, maxVal2-minVal2)
        upper_y = random.randint(down_y, maxVal2-minVal2)
        while({(down_x,upper_x):(down_y,upper_y)} in Q):
            down_x = random.randint(0, maxVal1-minVal1)
            upper_x = random.randint(down_x, maxVal1-minVal1)
            down_y = random.randint(0, maxVal2-minVal2)
            upper_y = random.randint(down_y, maxVal2-minVal2)
            count_regen+=1
        Q.append({(down_x,upper_x):(down_y,upper_y)})
    print("查询集随机生成完毕，生成了"+str(Q_size)+"个查询，共进行规避生成"+str(count_regen)+"次")
    return Q

def main():

    B = []  # 二维数据也可以是个多维的list
    Q_size = 400 # 查询集查询个数
    T = 200 # T是迭代运行次数。要注意的是，该处迭代次数不应大于上一步所生成的查询总数。
    eps = 0.1 # eps即epsilon，隐私预算。
    repetitions = 20 # MW机制的重复次数，一般不需更改

    B.append([])
    B.append([])
    # 使用外部数据时，该部分即为读取指定路径的测试数据。
    if USING_INPUT_DATA == True: 
        with open('Datasets/childMentalHealth_1M.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                    try:
                        B[0].append(int(row[0]))
                        B[1].append(int(row[1]))
                    except ValueError as e:
                        continue
                    except IndexError as e:
                        continue
    # 保留了一个测试模式：当不指定输入数据集时，允许随机生成一个6行2000列的记录来进行测试
    else:
        B = [[random.randint(0,100) for i in range(6)], [random.randint(0,100) for i in range(2000)]] #Dataset


   # 限制上下界
    maxVal1 = max(B[0])
    maxVal2 = max(B[1])
    minVal1 = min(B[0])
    minVal2 = min(B[1])

    # 随机生成Q_size个查询，并对查询集中的已有查询规避
    Q = {}
    Q = randomQueries(Q_size, maxVal1, minVal1,maxVal2,minVal2)

    print()
        
    #启动MWEM
    SyntheticData, RealHisto = MWEM(B, Q, T, eps, repetitions)
        
    #获得分析数据
    maxErr = maxError(RealHisto, SyntheticData, Q)
    minErr = minError(RealHisto, SyntheticData, Q)
    mse = meanSqErr(RealHisto, SyntheticData, Q)
        
    print()
    print("Real data histogram: " + str(RealHisto))
    print()
    # 格式化地输出合成数据到屏幕
    print("Synthetic Data: " + str(SyntheticData))
    print()
    print("Metrics:")
    print("  - Max Error: " + str(maxErr))
    print("  - Min Error: " + str(minErr))
    print("  - Mean Squared Error: " + str(mse))
    print("  - Mean Error: " + str(meanError(RealHisto, SyntheticData, Q)))
    print()
    
    
    #Plot化数据和生成直方图
    print("************ REAL DATA *******************")    
    
    H = np.array(RealHisto)

    fig = plt.figure(figsize=(4, 4.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical',ax=cax)
    plt.savefig("./Results/result_2D_TRUE.png")
    plt.show()
    
    print()
    print("************ Synthetic DATA **************")
    
    H2 = np.array(SyntheticData)

    fig = plt.figure(figsize=(4, 4.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H2)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical',ax=cax)
    plt.savefig("./Results/result_2D_SYN.png")
    plt.show()
    
main()
