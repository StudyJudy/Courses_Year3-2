#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
import numpy as np
import random
import math
from sklearn.metrics import mean_squared_error
import time
from collections import defaultdict
from typing import List, Tuple
import os
import sys
from operator import itemgetter
import csv
from statistics import mode, median
import matplotlib.pyplot as plt
from threading import Thread
from time import sleep


# In[2]:


# 设置字体为 SimHei黑体
plt.rcParams['font.family'] = 'SimHei'


# In[3]:


train_data_path = './Data-RecommandationSystem/train.txt'
test_data_path='./Data-RecommandationSystem/test.txt'


# ### 对数据集进行分析

# In[4]:


def data_analysis(dataPath, isTest):
    user_rating_counts = {}  # 存储每个用户的评分次数
    item_rating_counts = {}  # 存储每个物品被评分的次数
    score_counts = {}  # 存储不同评分的次数
    users = set()  # 存储所有用户ID
    items = set()  # 存储所有物品ID
    ratings = 0  # 总评分数
    min_user_id = float('inf')  # 最小用户ID，初始为正无穷大
    max_user_id = 0  # 最大用户ID，初始为0
    minItemId = float('inf')  # 最小物品ID，初始为正无穷大
    maxItemId = 0  # 最大物品ID，初始为0
    dataInfo = {}  # 存储最终的统计信息

    with open(dataPath, 'r') as readStream:  
        for line in readStream:  
            if not line.strip(): 
                continue
            header = line.strip()  # 读取并去除行首尾空白字符
            sepPos = header.find("|")  # 找到分隔符的位置
            userId = int(header[:sepPos])  # 提取用户ID
            rateNum = int(header[sepPos + 1:])  # 提取评分数量
            users.add(userId)  # 将用户ID添加到集合中
            ratings += rateNum  # 增加总评分数
            min_user_id = min(userId, min_user_id)  # 更新最小用户ID
            max_user_id = max(userId, max_user_id)  # 更新最大用户ID
            
            # 统计用户的评分次数
            if userId in user_rating_counts:
                user_rating_counts[userId] += rateNum
            else:
                user_rating_counts[userId] = rateNum

            # 统计每个物品的评分次数和不同评分的次数
            for _ in range(rateNum):  
                if not isTest:  # 如果不是测试集
                    itemId, score = map(int, readStream.readline().strip().split())  # 读取物品ID和评分
                    if itemId in item_rating_counts:  # 如果物品ID已经存在于字典中
                        item_rating_counts[itemId] += 1  # 增加评分次数
                    else:
                        item_rating_counts[itemId] = 1  # 否则初始化评分次数为1
                    
                    if score in score_counts:# 如果当前分数已存在于字典中
                        score_counts[score] += 1# 增加分数出现次数
                    else:
                        score_counts[score] = 1 # 初始化当前分数出现次数为1
                    
                else:  # 如果是测试集
                    itemId = int(readStream.readline().strip())  # 只读取物品ID
                items.add(itemId)  # 将物品ID添加到集合中
                minItemId = min(itemId, minItemId)  # 更新最小物品ID
                maxItemId = max(itemId, maxItemId)  # 更新最大物品ID
    
    if not isTest:
        # 计算平均每个用户的评分次数和平均每个物品的评分次数
        avg_user_ratings = sum(user_rating_counts.values()) / len(user_rating_counts)
        avg_item_ratings = sum(item_rating_counts.values()) / len(item_rating_counts)
    
        # 找出评分次数最多和最少的物品
        max_rated_item = max(item_rating_counts, key=item_rating_counts.get)
        min_rated_item = min(item_rating_counts, key=item_rating_counts.get)
    
        # 找出最常见的评分和次数
        most_common_score = max(score_counts, key=score_counts.get)
        most_common_score_count = score_counts[most_common_score]
    
        # 找出最不常见的评分和次数
        least_common_score = min(score_counts, key=score_counts.get)
        least_common_score_count = score_counts[least_common_score]
        
        # 计算评分的中位数
        scores = [score for score, count in score_counts.items() for _ in range(count)]
        median_score = median(scores)
    
    # 填充统计信息
    dataInfo["userNum"] = len(users)  # 用户数
    dataInfo["itemNum"] = len(items)  # 物品数
    dataInfo["rateNum"] = ratings  # 总评分数
    dataInfo["min_user_id"] = min_user_id  # 最小用户ID
    dataInfo["max_user_id"] = max_user_id  # 最大用户ID
    dataInfo["minItemId"] = minItemId  # 最小物品ID
    dataInfo["maxItemId"] = maxItemId  # 最大物品ID
    
    if not isTest:
        dataInfo["avg_user_ratings"] = avg_user_ratings # 平均每个用户的评分次数
        dataInfo["avg_item_ratings"] = avg_item_ratings # 平均每个物品的评分次数
        dataInfo["max_rated_item"] = max_rated_item # 评分次数最多的物品
        dataInfo["min_rated_item"] = min_rated_item # 评分次数最少的物品
        dataInfo["most_common_score"] = most_common_score # 最常见的评分
        dataInfo["most_common_score_count"] = most_common_score_count # 最常见评分出现的次数
        dataInfo["least_common_score"] = least_common_score # 最不常见的评分
        dataInfo["least_common_score_count"] = least_common_score_count # 最不常见评分出现的次数
        dataInfo["median_score"] = median_score # 评分的中位数

    if not isTest:  # 如果不是测试集
        with open("./Data-RecommandationSystem/Long_Tail_Data.txt", 'w') as longTailData:  # 打开文件写入长尾数据
            allItemRatedTimes = [(i, item_rating_counts.get(i, 0)) for i in range(minItemId, maxItemId + 1)]  # 获取所有物品的评分次数
            allItemRatedTimes.sort(key=lambda x: x[1], reverse=True)  # 按评分次数从高到低排序
            for item in allItemRatedTimes:  # 写入每个物品的评分次数
                longTailData.write(f"{item[0]} {item[1]}\n")

    return dataInfo  # 返回统计信息


# In[5]:


# 统计训练集的信息
train_result = data_analysis(train_data_path, False)
for key, value in train_result.items():
    print(key, ":", value)


# In[6]:


# 统计测试集的信息
test_result = data_analysis(test_data_path, True)
for key, value in test_result.items():
    print(key, ":", value)


# In[ ]:





# In[7]:


longtail_data_path = './Data-RecommandationSystem/Long_Tail_Data.txt'


# In[8]:


# 长尾效应分析
def analyze_longtail(data_path, threshold):
    item_rating_counts = {}  # 存储每个物品被评分的次数

    # 读取数据并统计每个物品的评分次数
    with open(data_path, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            item_id, rating_count = map(int, line.strip().split())
            item_rating_counts[item_id] = rating_count

    # 按评分次数从高到低排序
    sorted_items = sorted(item_rating_counts.items(), key=lambda x: x[1], reverse=True)

    # 计算长尾部分的数据量和总数据量
    total_items = len(sorted_items)
    tail_items = [item for item in sorted_items if item[1] <= threshold]
    tail_items_count = len(tail_items)

    # 计算长尾部分的占比
    tail_ratio = tail_items_count / total_items

    # 绘制长尾效应图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, total_items + 1), [count for _, count in sorted_items], color='blue', label='Rating Counts')
    plt.axvline(x=total_items - tail_items_count, color='red', linestyle='--', label='Tail Threshold')
    plt.title('Long Tail Effect Analysis')
    plt.xlabel('Items')
    plt.ylabel('Rating Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    return tail_ratio


# In[9]:


threshold = 15  # 评分次数小于等于15次的物品属于长尾部分
tail_ratio = analyze_longtail(longtail_data_path, threshold)
print(f"长尾部分的占比：{tail_ratio:.4%}")


# In[ ]:





# In[10]:


# 绘图完成用户活跃度分析，分析用户的评分次数分布
def plot_user_rating_distribution(dataPath, isTest):
    user_rating_counts = {}  # 存储每个用户的评分次数
    item_rating_counts = {}  # 存储每个物品被评分的次数
    score_counts = {}  # 存储不同评分的次数
    users = set()  # 存储所有用户ID
    items = set()  # 存储所有物品ID
    ratings = 0  # 总评分数
    min_user_id = float('inf')  # 最小用户ID，初始为正无穷大
    max_user_id = 0  # 最大用户ID，初始为0
    minItemId = float('inf')  # 最小物品ID，初始为正无穷大
    maxItemId = 0  # 最大物品ID，初始为0
    dataInfo = {}  # 存储最终的统计信息

    with open(dataPath, 'r') as readStream:  
        for line in readStream:  
            if not line.strip(): 
                continue
            header = line.strip()  # 读取并去除行首尾空白字符
            sepPos = header.find("|")  # 找到分隔符的位置
            userId = int(header[:sepPos])  # 提取用户ID
            rateNum = int(header[sepPos + 1:])  # 提取评分数量
            users.add(userId)  # 将用户ID添加到集合中
            ratings += rateNum  # 增加总评分数
            min_user_id = min(userId, min_user_id)  # 更新最小用户ID
            max_user_id = max(userId, max_user_id)  # 更新最大用户ID
            
            # 统计用户的评分次数
            if userId in user_rating_counts:
                user_rating_counts[userId] += rateNum
            else:
                user_rating_counts[userId] = rateNum

            # 统计每个物品的评分次数和不同评分的次数
            for _ in range(rateNum):  
                if not isTest:  # 如果不是测试集
                    itemId, score = map(int, readStream.readline().strip().split())  # 读取物品ID和评分
                    if itemId in item_rating_counts:  # 如果物品ID已经存在于字典中
                        item_rating_counts[itemId] += 1  # 增加评分次数
                    else:
                        item_rating_counts[itemId] = 1  # 否则初始化评分次数为1
                    
                    if score in score_counts:# 如果当前分数已存在于字典中
                        score_counts[score] += 1# 增加分数出现次数
                    else:
                        score_counts[score] = 1 # 初始化当前分数出现次数为1
                    
                else:  # 如果是测试集
                    itemId = int(readStream.readline().strip())  # 只读取物品ID
                items.add(itemId)  # 将物品ID添加到集合中
                minItemId = min(itemId, minItemId)  # 更新最小物品ID
                maxItemId = max(itemId, maxItemId)  # 更新最大物品ID
    
    if not isTest:
        plt.figure(figsize=(10, 6))
        plt.hist(user_rating_counts.values(), bins=150, color='cyan', edgecolor='black', alpha=0.7)
        plt.title('User Rating Counts Distribution')
        plt.xlabel('Number of Ratings 评分次数')
        plt.ylabel('Number of Users 具有特定评分次数的用户数量')
        plt.grid(True)
        plt.show()


# In[11]:


plot_user_rating_distribution(train_data_path, False)


# In[ ]:





# In[12]:


# 绘图完成物品受欢迎程度分析，分析每个物品的评分次数分布
def plot_item_rating_distribution(dataPath, isTest):
    user_rating_counts = {}  # 存储每个用户的评分次数
    item_rating_counts = {}  # 存储每个物品被评分的次数
    score_counts = {}  # 存储不同评分的次数
    users = set()  # 存储所有用户ID
    items = set()  # 存储所有物品ID
    ratings = 0  # 总评分数
    min_user_id = float('inf')  # 最小用户ID，初始为正无穷大
    max_user_id = 0  # 最大用户ID，初始为0
    minItemId = float('inf')  # 最小物品ID，初始为正无穷大
    maxItemId = 0  # 最大物品ID，初始为0
    dataInfo = {}  # 存储最终的统计信息

    with open(dataPath, 'r') as readStream:  
        for line in readStream:  
            if not line.strip(): 
                continue
            header = line.strip()  # 读取并去除行首尾空白字符
            sepPos = header.find("|")  # 找到分隔符的位置
            userId = int(header[:sepPos])  # 提取用户ID
            rateNum = int(header[sepPos + 1:])  # 提取评分数量
            users.add(userId)  # 将用户ID添加到集合中
            ratings += rateNum  # 增加总评分数
            min_user_id = min(userId, min_user_id)  # 更新最小用户ID
            max_user_id = max(userId, max_user_id)  # 更新最大用户ID
            
            # 统计用户的评分次数
            if userId in user_rating_counts:
                user_rating_counts[userId] += rateNum
            else:
                user_rating_counts[userId] = rateNum

            # 统计每个物品的评分次数和不同评分的次数
            for _ in range(rateNum):  
                if not isTest:  # 如果不是测试集
                    itemId, score = map(int, readStream.readline().strip().split())  # 读取物品ID和评分
                    if itemId in item_rating_counts:  # 如果物品ID已经存在于字典中
                        item_rating_counts[itemId] += 1  # 增加评分次数
                    else:
                        item_rating_counts[itemId] = 1  # 否则初始化评分次数为1
                    
                    if score in score_counts:# 如果当前分数已存在于字典中
                        score_counts[score] += 1# 增加分数出现次数
                    else:
                        score_counts[score] = 1 # 初始化当前分数出现次数为1
                    
                else:  # 如果是测试集
                    itemId = int(readStream.readline().strip())  # 只读取物品ID
                items.add(itemId)  # 将物品ID添加到集合中
                minItemId = min(itemId, minItemId)  # 更新最小物品ID
                maxItemId = max(itemId, maxItemId)  # 更新最大物品ID
    
    if not isTest:
        plt.figure(figsize=(10, 6))
        plt.hist(item_rating_counts.values(), bins=150, color='orange', edgecolor='black', alpha=0.7)
        plt.title('Item Rating Counts Distribution')
        plt.xlabel('Number of Ratings 评分次数')
        plt.ylabel('Number of Items 具有特定评分次数的项目数量')
        plt.grid(True)
        plt.show()


# In[13]:


plot_item_rating_distribution(train_data_path, False)


# In[ ]:





# ### ItemCF

# In[14]:


class ItemCF:
    def __init__(self, test_path="./Data-RecommandationSystem/test.txt", train_path="./Data-RecommandationSystem/train.txt", attribute_path="./Data-RecommandationSystem/itemAttribute.txt"):      
        self.test_path = test_path # 测试集路径   
        self.train_path = train_path # 训练集路径 
        self.attribute_path = attribute_path # 属性文件路径
        self.user_num = 0 # 用户数量
        self.num_items = 0
        self.global_avg = 0 # 全局平均评分

    def process_header(self, header):
        sep_pos = header.find("|")  # 找到分隔符的位置
        user_id = int(header[:sep_pos])  # 提取用户ID
        rate_num = int(header[sep_pos + 1:])  # 提取评分数量
        return user_id, rate_num
    
    # 获取user_num和num_items,并进行初始化
    def initialize_variables(self):
        maxUserId = 0
        minUserId = float('inf')
        maxItemId = 0
        minItemId = float('inf')
        
        with open(self.train_path, 'r') as read_stream:
            for line in read_stream:
                if not line.strip(): 
                    continue
                parts = line.strip().split()
                userId, rateNum = self.process_header(parts[0])
                minUserId = min(userId, minUserId)
                maxUserId = max(userId, maxUserId)
                
                # 遍历每个用户的评分信息
                for _ in range(rateNum):
                    itemId, score = map(int, read_stream.readline().strip().split())  # 读取物品ID和评分
                    minItemId = min(itemId, minItemId)
                    maxItemId = max(itemId, maxItemId)
        
        # 计算用户数量和物品数量
        self.user_num = maxUserId - minUserId + 1
        self.num_items = maxItemId - minItemId + 1
        
        # 初始化
        self.user_avg = np.zeros(self.user_num) # 用户平均评分
        self.item_avg = np.zeros(self.num_items) # 物品平均评分
        self.rating_count_user = np.zeros(self.user_num, dtype=int) # 用户评分数量
        self.rating_count_item = np.zeros(self.num_items, dtype=int) # 每个物品被评分的数量
        self.attr_list = [None] * self.num_items # 物品属性列表
        self.data = [[] for _ in range(self.user_num)]# 用户评分数据
        self.similarity_map = [[] for _ in range(self.num_items)]# 物品之间相似性的映射关系
        self.training_data = [[] for _ in range(self.num_items)]# 存储训练集中的物品评分数据
        print("initialize_variables done ...")
    
    def process_traindata(self):
        total_sum = 0
        total_count = 0
        
        with open(self.train_path, 'r') as file:
            for line in file:
                if not line.strip(): 
                    continue
                # 每个用户的第一行
                parts = line.strip().split()
                user_id, rate_num = self.process_header(parts[0])# 提取用户ID和评分数量
                self.rating_count_user[user_id] = rate_num
                total_count += rate_num # 累加总评分数量
                
                # 遍历每个用户的评分信息
                for _ in range(rate_num):
                    item_id, score = map(int, file.readline().strip().split()) # 读取物品ID和评分
                    self.training_data[user_id].append((item_id, score))
                    self.user_avg[user_id] += score # 累加用户的评分总和
                    self.item_avg[item_id] += score # 累加用户的评分总和
                    total_sum += score # 累加总评分的和
                
                self.user_avg[user_id] /= self.rating_count_user[user_id]  # 计算用户平均评分
        
        self.global_avg = total_sum / total_count # 计算全局平均评分
        
        # 计算每个物品的平均评分
        for i in range(self.num_items):
            if self.rating_count_item[i] != 0:
                self.item_avg[i] /= self.rating_count_item[i]
            else: # 如果某个物品没有被评分，则将其添加一个虚拟评分，以避免除以零错误
                self.training_data[i].append((None, 0))
                self.rating_count_item[i] += 1
        
        print("Read train.txt done...")

    def load_item_attributes(self):
        with open(self.attribute_path, 'r') as file:
            # lines = file.readlines()
            for line in file:
                if line.strip() == "":
                    break
                parts = line.strip().split('|')
                item_id = int(parts[0]) # 获取物品 ID
                attr1 = int(parts[1]) if parts[1] != "None" else 0 # 获取第一个属性
                attr2 = int(parts[2]) if parts[2] != "None" else 0 # 获取第二个属性
                norm = np.sqrt(attr1 * attr1 + attr2 * attr2)# 计算属性的欧几里得范数
                self.attr_list[item_id] = {'attr1': attr1, 'attr2': attr2, 'norm': norm}# 将物品属性信息存储到 self.attr_list 中的字典中
        print("Read itemAttribute.txt done ...")

    # 评估两个物品之间的属性相似度
    def calculate_attribute_sim(self, item1, item2):
        # 如果item1或item2的属性列表为None，则相似度0
        if self.attr_list[item1] is None or self.attr_list[item2] is None:
            return 0
        # 如果item1或item2的属性向量的范数为0，则相似度0
        if self.attr_list[item1]['norm'] == 0 or self.attr_list[item2]['norm'] == 0:
            return 0
        # 计算两个物品属性向量的点积
        a = self.attr_list[item1]['attr1'] * self.attr_list[item2]['attr1'] + self.attr_list[item1]['attr2'] * self.attr_list[item2]['attr2']
        # 计算两个物品属性向量范数的乘积
        b = self.attr_list[item1]['norm'] * self.attr_list[item2]['norm']
        # 计算属性相似度，即点积除以范数的乘积
        attribute_similarity = a / b
        # 返回属性相似度
        return attribute_similarity
    
    # 评估两个物品在用户评分上的相似程度
    def calculate_item_rating_pearson_sim(self, item1, item2):
        # 确定评分数量较少和较多的物品
        item_less = item1 if self.rating_count_item[item1] <= self.rating_count_item[item2] else item2
        item_more = item2 if self.rating_count_item[item1] <= self.rating_count_item[item2] else item1
        
        # 获取评分数量的最小值
        num = min(self.rating_count_item[item1], self.rating_count_item[item2])
        
        # 初始化皮尔逊相关系数的分子和各自物品的评分差值的平方和
        numerator = 0
        sum_sq1 = 0
        sum_sq2 = 0
        
        # 如果两个物品中任意一个没有评分数据，则相似度为 0
        if self.training_data[item_less][0][0] is None or self.training_data[item_more][0][0] is None:
            return 0
        
        # 使用集合存储每个物品的用户评分数据
        item_less_ratings = {user_id: rating for user_id, rating in self.training_data[item_less]}
        item_more_ratings = {user_id: rating for user_id, rating in self.training_data[item_more]}
    
        # 找出共同评分用户
        common_users = set(item_less_ratings.keys()).intersection(set(item_more_ratings.keys()))
        
        for user_id in common_users:
            rating_less = item_less_ratings[user_id]
            rating_more = item_more_ratings[user_id]
        
            # 计算各自物品的评分与平均评分之差的乘积之和
            numerator += (rating_less - self.item_avg[item_less]) * (rating_more - self.item_avg[item_more])
        
            # 计算各自物品评分与平均评分之差的平方和
            sum_sq1 += (rating_less - self.item_avg[item_less]) ** 2
            sum_sq2 += (rating_more - self.item_avg[item_more]) ** 2
        
        # 计算用户1和用户2评分差的平方和的平方根
        denominator = np.sqrt(sum_sq1) * np.sqrt(sum_sq2)
        
        # 如果任何一个平方根为 0，则相似度为 0
        if denominator == 0:
            return 0
        
        # 皮尔逊相似度
        similarity = numerator / denominator
        return similarity

    # 综合计算两种相似度
    def calculate_overall_sim(self, item1, item2):
        # 确定两个物品的较小和较大的编号
        min_item = min(item1, item2)
        max_item = max(item1, item2)
        
        # 检查是否已经计算过这两个物品的相似度，如果计算过则直接返回缓存中的值
        for sim_item in self.similarity_map[min_item]:
            if sim_item['id'] == max_item:
                return sim_item['score']
        
        # 如果缓存中没有，则计算两个物品的皮尔逊相似度和属性相似度
        pearson_sim = self.calculate_item_rating_pearson_sim(item1, item2)
        attribute_sim = self.calculate_attribute_sim(item1, item2)
        
        # 综合计算得到最终的相似度得分
        sim_score = (pearson_sim + attribute_sim) / 2
        
        # 将计算结果存入相似度缓存
        self.similarity_map[min_item].append({'id': max_item, 'score': sim_score})
        # 返回计算得到的相似度得分
        return sim_score

    def predict(self, user_id, item_id):
        sum_sim = 0 # 初始化相似度总和
        sum_sim_item = 0 # 初始化相似度加权评分总和
    
        # 遍历所有物品
        for i in range(self.num_items):
            if i == item_id:# 跳过当前物品
                continue
            
            # 遍历物品 i 的所有用户评分数据，如果寻找到了用户 user_id 对物品 i 的评分
            if any(item[0] == user_id for item in self.training_data[i]):
                # 计算物品 item_id 和物品 i 的总体相似度
                sim = self.calculate_overall_sim(item_id, i)
                    
                if sim > 0:# 只考虑正相似度                    
                    # 获取用户 user_id 对物品 i 的评分
                    for item in self.training_data[i]:
                        if item[0] == user_id:
                            rating_i = item[1]
                            break
                            
                    # 计算基线预测值，包括全局平均、用户偏差和物品偏差的加权和
                    baseline = self.global_avg + (self.user_avg[user_id] - self.global_avg) + (self.item_avg[i] - self.global_avg)
                    # 更新相似度总和和加权评分总和
                    sum_sim += sim # 累加相似度
                    sum_sim_item += sim * (rating_i - baseline)# 累加加权评分差
        
        # 如果相似度总和和加权评分差总和都不为零
        if sum_sim != 0 and sum_sim_item != 0:
            # 重新计算基线预测值，包括全局平均、用户偏差和物品偏差的加权和
            baseline = self.global_avg + (self.user_avg[user_id] - self.global_avg) + (self.item_avg[item_id] - self.global_avg)
            # 计算最终预测评分
            predicted_score = baseline + sum_sim_item / sum_sim
             # 返回预测评分
            return min(100, max(0, predicted_score))
        
        else:# 如果没有有效的相似度或加权评分，则返回用户的平均评分
            return self.user_avg[user_id]

    def write_predictions(self):
        print("Predicting ...")
        with open("./Data-RecommandationSystem/Res_UserCF.txt", 'w') as file:
            with open(self.test_path, 'r') as test_file:
                user_count = 0 
                for line in test_file:
                    if not line.strip():
                        continue
                    parts = line.strip().split('|')
                    user_id = int(parts[0])
                    file.write(f"{user_id}|{len(parts[1:])}\n")
                    ratings = parts[1:]
                    
                    for _ in ratings:
                        item_id = int(test_file.readline().strip())
                        predict_rating = int(self.predict(user_id, item_id))
                        file.write(f"{item_id} {predict_rating}\n")
                        
                    user_count += 1
                    # 每预测300个用户输出一次提示信息
                    if user_count % 300 == 0:
                        print(f"Predicted {user_count} users...")
                        
        print("Prediction done ...")

    def run(self):
        start_time = time.time()
        self.initialize_variables()
        self.load_item_attributes()
        self.process_traindata()
        self.write_predictions()
        end_time = time.time()
        print(f"overall running time: {end_time - start_time}")


# In[ ]:


item_cf = ItemCF()
item_cf.run()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




