from phe import paillier #开源库
import time #做性能测试
from Crypto.Cipher import DES3,AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import hashlib
import random
import os

# 在客户端保存对称密钥k，
# 在服务器端存储m个用对称密钥k加密的密文，
# 通过隐私信息获取方法得到指定密文后能解密得到对应的明文。

def generate_sym_key(length):
    return os.urandom(length)

# 随机生成对称密钥
sym_key = generate_sym_key(24)
print("sym_key: ",sym_key.hex())

################### 设置参数
# 服务器端保存的数值
message_list =[100,200,300,400,500,600,700,800,900,1000]
length =len(message_list)

# 服务器端进行加密并保存
enc_message_list = []
# DES3.MODE_ECB：DES3 加密算法的电子密码本模式（Electronic Codebook）。
# 在 ECB 模式下，每个分组都独立地进行加密，相同的明文分组将得到相同的密文分组
cipher = DES3.new(sym_key,DES3.MODE_ECB)
for i in range(length):
    ciphertext=cipher.encrypt(message_list[i].to_bytes(length=24,byteorder='big',signed=False))
    enc_message_list.append(ciphertext)

# 客户端生成公私钥，选择读取的位置
public_key,private_key = paillier.generate_paillier_keypair()
# 随机选择一个位置读取
pos = random.randint(0,length-1)
print("本次要读取的数值位置是: ",pos)

##########客户端生成密文选择向量
select_list=[]
enc_list=[]
for i in range(length):
    select_list.append(i == pos)
    enc_list.append(public_key.encrypt(select_list[i]))

############ 服务器端进行运算
c = 0
for i in range(length):
    trans_message_list=int().from_bytes(enc_message_list[i],byteorder='big',signed=False)
    c = c + trans_message_list * enc_list[i]
# print("产生密文: ",c.ciphertext())

########### 客户端进行解密
m = private_key.decrypt(c)
print("未解密的密文为: ",m)
m = cipher.decrypt(m.to_bytes(24, 'big', signed=True))
print("解密后的明文为: ",int().from_bytes(m,byteorder='big',signed=True))
