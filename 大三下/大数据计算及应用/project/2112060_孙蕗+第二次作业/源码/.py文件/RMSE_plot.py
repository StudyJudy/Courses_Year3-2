#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# 假设以下是训练过程中得到的 RMSE 值
rmse_values = [28.189479245705037, 26.012377917420725, 24.196028878575582, 22.546292669042224, 21.06244045845831, 19.79026591874451, 18.73611774785506, 17.87743062343224, 17.181927070815373, 16.618009741991507, 16.15843344121727, 15.781145579244157, 15.368825133313587, 15.108037575562344, 14.78839566987336, 14.60186378175991, 14.442202827744706, 14.30451245854927, 14.184908300391892, 14.080337659163852, 13.988339416427604, 13.806925218061578, 13.734482022648925, 13.669689181814854, 13.611450585800052, 13.558854972296818, 13.511144012447286, 13.467681942064266, 13.427934382577844, 13.391451298350922, 13.357847130300365 , 13.32679004239488, 13.297997833321514,13.2712184161282 ,13.246239997841528,13.222876355237239 , 13.20096501975838, 13.180360312650092,13.160944859547466, 13.142611473662165,13.125257394050972 , 13.108801000777002, 13.093166461075474, 13.078285336842762,13.064092783325187,13.05053940848336,13.037576044609115 ,13.025155878534039,13.013238141633464,13.001790692729472    ]
epochs = range(1, len(rmse_values) + 1)

# SVD
def plot_rmse_curve(epochs, rmse_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmse_values, marker='o', label='RMSE')

    # 标记最高点和最低点
    max_rmse = max(rmse_values)
    min_rmse = min(rmse_values)
    max_epoch = rmse_values.index(max_rmse) + 1
    min_epoch = rmse_values.index(min_rmse) + 1
    
    plt.scatter(max_epoch, max_rmse, color='red')
    plt.scatter(min_epoch, min_rmse, color='green')
    plt.text(max_epoch, max_rmse, f'Max RMSE: {max_rmse:.4f}', ha='right')
    plt.text(min_epoch, min_rmse, f'Min RMSE: {min_rmse:.4f}', ha='right')

    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用函数绘制图形
plot_rmse_curve(epochs, rmse_values)


# In[ ]:





# In[6]:


# SVD++

# 假设以下是训练过程中得到的 RMSE 值
rmse_values = [65.8658, 64.6596, 63.3882, 61.8409, 61.9198, 62.1358, 63.1784, 63.2056, 63.6864, 63.7258]
epochs = range(1, len(rmse_values) + 1)

def plot_rmse_curve(epochs, rmse_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmse_values, marker='o', label='RMSE')

    # 标记最高点和最低点
    max_rmse = max(rmse_values)
    min_rmse = min(rmse_values)
    max_epoch = rmse_values.index(max_rmse) + 1
    min_epoch = rmse_values.index(min_rmse) + 1
    
    plt.scatter(max_epoch, max_rmse, color='red')
    plt.scatter(min_epoch, min_rmse, color='green')
    plt.text(max_epoch, max_rmse, f'Max RMSE: {max_rmse:.4f}', ha='right')
    plt.text(min_epoch, min_rmse, f'Min RMSE: {min_rmse:.4f}', ha='right')

    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用函数绘制图形
plot_rmse_curve(epochs, rmse_values)


# In[ ]:





# In[3]:


# SVD_coopy

# 假设以下是训练过程中得到的 RMSE 值
rmse_values = [31.497486442178236, 31.519171191727086, 31.510833293244293, 31.526367319522212, 31.50389087108396, 31.50530319948567, 31.514515461442567, 31.520370667219254, 31.486837253275997, 31.509715942120824, 31.503224698644534, 31.493519903667085, 31.50262307460901, 31.512817672587367, 31.50642302440784, 31.501357212323374, 31.516060753726038, 31.50179489563415, 31.51922560093666]
epochs = range(1, len(rmse_values) + 1)

def plot_rmse_curve(epochs, rmse_values):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmse_values, marker='o', label='RMSE')

    # 标记最高点和最低点
    max_rmse = max(rmse_values)
    min_rmse = min(rmse_values)
    max_epoch = rmse_values.index(max_rmse) + 1
    min_epoch = rmse_values.index(min_rmse) + 1
    
    plt.scatter(max_epoch, max_rmse, color='red')
    plt.scatter(min_epoch, min_rmse, color='green')
    plt.text(max_epoch, max_rmse, f'Max RMSE: {max_rmse:.4f}', ha='right')
    plt.text(min_epoch, min_rmse, f'Min RMSE: {min_rmse:.4f}', ha='right')

    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# 调用函数绘制图形
plot_rmse_curve(epochs, rmse_values)


# In[ ]:




