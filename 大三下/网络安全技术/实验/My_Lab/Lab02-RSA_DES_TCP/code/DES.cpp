#include <iostream>
#include <bitset>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <netinet/in.h>
// #include "commoncode.cpp"
#include "DES.h"

using namespace std;

//初始置换表(IP置换)
const static unsigned char IP_table[64] =
{
	58 , 50 , 42 , 34 , 26 , 18 , 10 , 2 ,
	60 , 52 , 44 , 36 , 28 , 20 , 12 , 4 ,
	62 , 54 , 46 , 38 , 30 , 22 , 14 , 6 ,
	64 , 56 , 48 , 40 , 32 , 24 , 16 , 8 ,
	57 , 49 , 41 , 33 , 25 , 17 , 9 , 1 ,
	59 , 51 , 43 , 35 , 27 , 19 , 11 , 3 ,
	61 , 53 , 45 , 37 , 29 , 21 , 13 , 5 ,
	63 , 55 , 47 , 39 , 31 , 23 , 15 , 7
};

//逆初始置换
const static unsigned char IP_reverse_table[64] =
{
	40 , 8 , 48 , 16 , 56 , 24 , 64 , 32 ,
	39 , 7 , 47 , 15 , 55 , 23 , 63 , 31 ,
	38 , 6 , 46 , 14 , 54 , 22 , 62 , 30 ,
	37 , 5 , 45 , 13 , 53 , 21 , 61 , 29 ,
	36 , 4 , 44 , 12 , 52 , 20 , 60 , 28 ,
	35 , 3 , 43 , 11 , 51 , 19 , 59 , 27 ,
	34 , 2 , 42 , 10 , 50 , 18 , 58 , 26 ,
	33 , 1 , 41 , 9 , 49 , 17 , 57 , 25
};

//选择扩展置换变换表（E盒）
static const unsigned char Extend_table[48] =
{
	32 , 1 , 2 , 3 , 4 , 5 ,
	4 , 5 , 6 , 7 , 8 , 9 ,
	8 , 9 , 10 , 11 ,12 , 13 ,
	12 , 13 , 14 , 15 , 16 , 17 ,
	16 , 17 , 18 , 19 , 20 , 21 ,
	20 , 21 , 22 , 23 , 24 , 25 ,
	24 , 25 , 26 , 27 , 28 , 29 ,
	28 , 29 , 30 , 31 , 32 , 1
};

//选择压缩运算 S盒
const static unsigned char S_Box[8][64] =
{
	{//S1
		14 , 4 , 13 , 1 , 2 , 15 , 11 , 8 , 3 , 10 , 6 , 12 , 5 , 9 , 0 , 7 ,
		0 , 15 , 7 , 4 , 14 , 2 , 13 , 1 , 10 , 6 , 12 , 11 , 9 , 5 , 3 , 8 ,
		4 , 1 , 14 , 8 , 13 , 6 , 2 , 11 , 15 , 12 , 9 , 7 , 3 , 10 , 5 , 0 ,
		15 , 12 , 8 , 2 , 4 , 9 , 1 , 7 , 5 , 11 , 3 , 14 , 10 , 0 , 6 , 13
	} ,

	{//S2
		15 , 1 , 8 , 14 , 6 , 11 , 3 , 4 , 9 , 7 , 2 , 13 , 12 , 0 , 5 , 10 ,
		3 , 13 , 4 , 7 , 15 , 2 , 8 , 14 , 12 , 0 , 1 , 10 , 6 , 9 , 11 , 5 ,
		0 , 14 , 7 , 11 , 10 , 4 , 13 , 1 , 5 , 8 , 12 , 6 , 9 , 3 , 2 , 15 ,
		13 , 8 , 10 , 1 , 3 , 15 , 4 , 2 , 11 , 6 , 7 , 12 , 0 , 5 , 14 , 9
	} ,

	{//S3
		10 , 0 , 9 , 14 , 6 , 3 , 15 , 5 , 1 , 13 , 12 , 7 , 11 , 4 , 2 , 8 ,
		13 , 7 , 0 , 9 , 3 , 4 , 6 , 10 , 2 , 8 , 5 , 14 , 12 , 11 , 15 , 1 ,
		13 , 6 , 4 , 9 , 8 , 15 , 3 , 0 , 11 , 1 , 2 , 12 , 5 , 10 , 14 , 7 ,
		1 , 10 , 13 , 0 , 6 , 9 , 8 , 7 , 4 , 15 , 14 , 3 , 11 , 5 , 2 , 12
	} ,

	{//S4
		7 , 13 , 14 , 3 , 0 , 6 , 9 , 10 , 1 , 2 , 8 , 5 , 11 , 12 , 4 , 15 ,
		13 , 8 , 11 , 5 , 6 , 15 , 0 , 3 , 4 , 7 , 2 , 12 , 1 , 10 , 14 , 9 ,
		10 , 6 , 9 , 0 , 12 , 11 , 7 , 13 , 15 , 1 , 3 , 14 , 5 , 2 , 8 , 4 ,
		3 , 15 , 0 , 6 , 10 , 1 , 13 , 8 , 9 , 4 , 5 , 11 , 12 , 7 , 2 , 14
	} ,

	{//S5
		2 , 12 , 4 , 1 , 7 , 10 , 11 , 6 , 8 , 5 , 3 , 15 , 13 , 0 , 14 , 9 ,
		14 , 11 , 2 , 12 , 4 , 7 , 13 , 1 , 5 , 0 , 15 , 10 , 3 , 9 , 8 , 6 ,
		4 , 2 , 1 , 11 , 10 , 13 , 7 , 8 , 15 , 9 , 12 , 5 , 6 , 3 , 0 , 14 ,
		11 , 8 , 12 , 7 , 1 , 14 , 2 , 13 , 6 , 15 , 0 , 9 , 10 , 4 , 5 , 3
	} ,

	{//S6
		12 , 1 , 10 , 15 , 9 , 2 , 6 , 8 , 0 , 13 , 3 , 4 , 14 , 7 , 5 , 11 ,
		10 , 15 , 4 , 2 , 7 , 12 , 9 , 5 , 6 , 1 , 13 , 14 , 0 , 11 , 3 , 8 ,
		9 , 14 , 15 , 5 , 2 , 8 , 12 , 3 , 7 , 0 , 4 , 10 , 1 , 13 , 11 , 6 ,
		4 , 3 , 2 , 12 , 9 , 5 , 15 , 10 , 11 , 14 , 1 , 7 , 6 , 0 , 8 , 13
	} ,

	{//S7
		4 , 11 , 2 , 14 , 15 , 0 , 8 , 13 , 3 , 12 , 9 , 7 , 5 , 10 , 6 , 1 ,
		13 , 0 , 11 , 7 , 4 , 9 , 1 , 10 , 14 , 3 , 5 , 12 , 2 , 15 , 8 , 6 ,
		1 , 4 , 11 , 13 , 12 , 3 , 7 , 14 , 10 , 15 , 6 , 8 , 0 , 5 , 9 , 2 ,
		6 , 11 , 13 , 8 , 1 , 4 , 10 , 7 , 9 , 5 , 0 , 15 , 14 , 2 , 3 , 12
	} ,

	{//S8
		13 , 2 , 8 , 4 , 6 , 15 , 11 , 1 , 10 , 9 , 3 , 14 , 5 , 0 , 12 , 7 ,
		1 , 15 , 13 , 8 , 10 , 3 , 7 , 4 , 12 , 5 , 6 , 11 , 0 , 14 , 9 , 2 ,
		7 , 11 , 4 , 1 , 9 , 12 , 14 , 2 , 0 , 6 , 10 , 13 , 15 , 3 , 5 , 8 ,
		2 , 1 , 14 , 7 , 4 , 10 , 8 , 13 , 15 , 12 , 9 , 0 , 3 , 5 , 6 , 11
	}
};

//置换运算 P盒
const static unsigned char P_table[32] =
{
	16 , 7 , 20 , 21 , 29 , 12 , 28 , 17 ,
	1 , 15 , 23 , 26 , 5 , 18 , 31 , 10 ,
	2 , 8 , 24 , 14 , 32 , 27 , 3 , 9 ,
	19 , 13 , 30 , 6 , 22 , 11 , 4 , 25
};

// 将 64 bit 的明文重新排列，而后分成左右两块，每块 32bit，用 left 和 right 表示。
// IP置换表中的数据指的是位置，例如IP_table[0]=58指将明文第58位放置第1位
// 观察IP置换表，发现相邻两列元素位置相差8位，前32位均为原奇数位号码，后32个均为原偶数位号码。
// 各列经过偶采样和奇采样置换后，再对其进行逆序排列，将阵中元素按行读出以便构成置换的输出。
// 右半部分会直接作为下一轮次的左半部分的输入

int IP(const Block& plain_block, HBlock& left_block, HBlock& right_block)
{
	// 第一个循环遍历right数组的每个元素，根据IP_table中的值从block中取出相应的元素赋值给right。
	for (int i = 0; i < right_block.size(); ++i)
	{
		right_block[i] = plain_block[IP_table[i] - 1];// 置换后的right部分
	}
	for (int i = 0; i < left_block.size(); ++i)
	{
		left_block[i] = plain_block[IP_table[i + left_block.size()] - 1];// 置换后的left部分
	}
	return 0;
}

// 16轮迭代
// 加解密运算
int Turn(HBlock& left_block, HBlock& right_block, const Code& subkey)
{
	Code Extend_code;//48位数据块
	HBlock P_code;//32位数据块

	// 下面右半部分进行F函数运算	
	//将右半部分通过E盒扩展变换为48位
	for (int i = 0; i < Extend_code.size(); ++i)
	{
		Extend_code[i] = right_block[Extend_table[i] - 1];//扩展置换
	}
	// 扩展完成后进行密钥加运算，
	// 将选择扩展运算输出的 48bit 作为输入，与 48bit 的子密钥进行异或运算，异或结果作为选择压缩运算（S 盒）的输入。

	Extend_code ^= subkey;//与子密钥异或

	//选择压缩运算（S盒）

	bitset<4> col;//S盒的列
	bitset<2> row;//S盒的行

	// 8个S盒，每个S盒有4行16列，
	// 每个S表输入为6bit，输出4bit。
	// 在查表前，需要将密钥加运算得到的48bit输出分为8组，每组6 bit，分别进入8个S盒表参与运算作为输入，得到的8个4bit的输出，将其串在一起的32bit输出作为置换运算（P盒）的输入。

	// S盒运算如下，S盒输入的6bit b1b2b3b4b5b6，取第一位b1和最后一位b6组成二进制数r = b1b6，其余位b2b3b4b5组成二进制数c = b2b3b4b5。
	// 查找对应的S盒表，找到第r + 1行c + 1列的数，转为二进制后即为该S盒运算要输出的结果。

	for (int i = 0; i < 8; ++i)
	{
		row[0] = Extend_code[6 * i];//行标
		row[1] = Extend_code[6 * i + 5];

		col[0] = Extend_code[6 * i + 1];//列标
		col[1] = Extend_code[6 * i + 2];
		col[2] = Extend_code[6 * i + 3];
		col[4] = Extend_code[6 * i + 4];

		// to_ulong()将row和col 两个bitset对象转换为unsigned long
		// S盒有4行16列，通过计算得到对应的转换值在S盒的哪个位置
		bitset<4> temp(S_Box[i][row.to_ulong() * 16 + col.to_ulong()]);

		// 将8组S盒计算得到的结果拼接成一个32 bit 字符串
		// P盒替换需要将S盒替换的32位输出作为输入
		for (int j = 0; j < temp.size(); ++j)
		{
			Extend_code[4 * i + j] = temp[j];
		}
	}
	// 将S盒计算得到的32位输出作为P盒的输入，经过替换表替换之后即位F轮函数的结果
	for (int i = 0; i < P_code.size(); ++i)
	{
		P_code[i] = Extend_code[P_table[i] - 1];//P盒置换
	}
	// P盒计算得到的结果与上一轮左边的 32bit 进行异或作为下一轮的右半部分
	left_block ^= P_code;
	return 0;
}

//交换左右两个部分
int exchange(HBlock& left_block, HBlock& right_block)
{
	HBlock temp_block;
	for (int i = 0; i < temp_block.size(); ++i)
	{
		temp_block[i] = left_block[i];
	}
	for (int i = 0; i < left_block.size(); ++i)
	{
		left_block[i] = right_block[i];
	}
	for (int i = 0; i < right_block.size(); ++i)
	{
		right_block[i] = temp_block[i];
	}
	return 0;
}

//将左右两部分数据进行逆初始置换并拼成一个数据块
int Reverse_IP(const HBlock& left_block, const HBlock& right_block, Block& block)
{
	for (int i = 0; i < block.size(); ++i)
	{
		if (IP_reverse_table[i] <= 32)
		{
			block[i] = right_block[IP_reverse_table[i] - 1];//从right部分获取数据
		}
		else
		{
			block[i] = left_block[IP_reverse_table[i] - 32 - 1];//从left部分获取数据
		}
	}
	return 0;
}

//密钥置换选择表PC-1，将64bit密钥置换为56bit
const static unsigned char PC1_table[56] =
{
	57 , 49 , 41 , 33 , 25 , 17 , 9 ,
	1 ,58 , 50 , 42 , 34 , 26 , 18 ,
	10 , 2 ,59 , 51 , 43 , 35 , 27 ,
	19 , 11 , 3 ,60 , 52 , 44 , 36 ,

	63 , 55 , 47 , 39 ,31 , 23 , 15 ,
	7 , 62 , 54 , 46 , 38 ,30 , 22 ,
	14 , 6 , 61 , 53 , 45 , 37 ,29 ,
	21 , 13 , 5 , 28 , 20 , 12 , 4
};

//循环左移每轮移动的位数
// 当i=1，2，9，16循环左移动1位, 否则循环左移动2位
const static unsigned char bit_leftshift[16] =
{
	1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 ,
	1 , 2 , 2 , 2 , 2 , 2 , 2 , 1
};

//密钥置换选择表PC-2，将56 bit中的第9、18、22、25、38、43、54位删除，将56bit密钥置换成48bit
const static unsigned char PC2_table[48] =
{
	14 , 17 , 11 , 24 , 1 , 5 , 3 , 28 ,
	15 , 6 , 21 , 10 , 23 , 19 , 12 , 4 ,
	26 , 8 , 16 , 7 , 27 , 20 , 13 , 2 ,
	41 , 52 , 31 , 37 , 47 , 55 , 30 , 40 ,
	51 , 45 , 33 , 48 , 44 , 49 , 39 , 56 ,
	34 , 53 , 46 , 42 , 50 , 36 , 29 , 32
};

//获取bkey产生的第n轮子密钥
Code getkey(const unsigned int N, const Block& bkey)
{
	//n在区间[0,15]之间取值，bkey为64位密钥

	Code subkey_pc2;//返回值,48位子密钥
	Key subkey_pc1;//56位密钥

	unsigned int pc1_len = subkey_pc1.size();// 56
	unsigned int pc2_len = subkey_pc2.size();// 48

	//PC-1置换，得到56位密钥
	for (int i = 0; i < 56; ++i)
	{
		subkey_pc1[i] = bkey[PC1_table[i] - 1];
	}

	// 输出的56bit将被分成两组，每组28bit，分别进入C寄存器（pc1_table的上半部分）和D寄存器（pc1_table的下半部分），准备进行循环左移

	for (int i = 0; i <= N; ++i)
	{
		//第N轮密钥产生需要循环左移的位数为bit_leftshift的第N个元素的值

		int shift = bit_leftshift[N]; // 获取当前轮次的左移位数

		// 对两个28位部分分别进行循环左移
		// 0为左半部分，1为右半部分
		for (int part = 0; part < 2; ++part)
		{
			unsigned char temp[28]; // 临时存储左移后的结果
			for (int i = 0; i < 28; ++i)
			{
				int new_index = (i + shift) % 28; // 计算左移后的新位置
				temp[new_index] = subkey_pc1[part * 28 + i]; // 执行左移
			}
			// 将临时数组中的结果复制回原数组
			for (int i = 0; i < 28; ++i)
			{
				subkey_pc1[part * 28 + i] = temp[i];
			}
		}
	}
	// PC-2置换
	// 将56 bit中的第9、18、22、25、38、43、54位删除，其余位置按照PC2_table置换位置，将56bit密钥置换成48bit
	for (int i = 0; i < subkey_pc2.size(); ++i)
	{
		subkey_pc2[i] = subkey_pc1[PC2_table[i] - 1];
	}
	return subkey_pc2;
}

//加解密运算
int des(Block& block, Block& bkey, const Method method)
{
	//block为数据块，bkey为64位密钥
	HBlock left_block, right_block;//左右部分
	IP(block, left_block, right_block);//初始置换

	switch (method)
	{
	case e://加密
		for (char i = 0; i < 16; ++i)
		{
			Code key = getkey(i, bkey);
			Turn(left_block, right_block, key);
			if (i != 15)
			{
				exchange(left_block, right_block);
			}
		}
		break;
	case d://解密
		for (char i = 15; i >= 0; --i)
		{
			Code key = getkey(i, bkey);
			Turn(left_block, right_block, key);
			if (i != 0)
			{
				exchange(left_block, right_block);
			}
		}
		break;
	default:
		break;
	}
	Reverse_IP(left_block, right_block, block);//末置换
	return 0;
}

