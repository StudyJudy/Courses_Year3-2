from phe import paillier # 开源库
import random # 选择随机数

################### 设置参数
# 服务器端保存的数值
message_list =[100,200,300,400,500,600,700,800,900,1000]
length =len(message_list)

# 客户端生成公私钥，选择读取的位置
public_key,private_key = paillier.generate_paillier_keypair()
pos = random.randint(0,length-1)

##########客户端生成密文选择向量
select_list=[]
enc_list=[]
for i in range(length):
    select_list.append(i == pos)
    enc_list.append(public_key.encrypt(select_list[i]))

############ 服务器端进行运算
c = 0
for i in range(length):
    c = c + message_list[i] * enc_list[i]
print("产生密文: ",c.ciphertext())


########### 客户端进行解密
m = private_key.decrypt(c)
print("得到数值: ",m)
