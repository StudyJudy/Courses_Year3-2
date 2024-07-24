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

//��ʼ�û���(IP�û�)
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

//���ʼ�û�
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

//ѡ����չ�û��任��E�У�
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

//ѡ��ѹ������ S��
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

//�û����� P��
const static unsigned char P_table[32] =
{
	16 , 7 , 20 , 21 , 29 , 12 , 28 , 17 ,
	1 , 15 , 23 , 26 , 5 , 18 , 31 , 10 ,
	2 , 8 , 24 , 14 , 32 , 27 , 3 , 9 ,
	19 , 13 , 30 , 6 , 22 , 11 , 4 , 25
};

// �� 64 bit �������������У�����ֳ��������飬ÿ�� 32bit���� left �� right ��ʾ��
// IP�û����е�����ָ����λ�ã�����IP_table[0]=58ָ�����ĵ�58λ���õ�1λ
// �۲�IP�û���������������Ԫ��λ�����8λ��ǰ32λ��Ϊԭ����λ���룬��32����Ϊԭż��λ���롣
// ���о���ż������������û����ٶ�������������У�������Ԫ�ذ��ж����Ա㹹���û��������
// �Ұ벿�ֻ�ֱ����Ϊ��һ�ִε���벿�ֵ�����

int IP(const Block& plain_block, HBlock& left_block, HBlock& right_block)
{
	// ��һ��ѭ������right�����ÿ��Ԫ�أ�����IP_table�е�ֵ��block��ȡ����Ӧ��Ԫ�ظ�ֵ��right��
	for (int i = 0; i < right_block.size(); ++i)
	{
		right_block[i] = plain_block[IP_table[i] - 1];// �û����right����
	}
	for (int i = 0; i < left_block.size(); ++i)
	{
		left_block[i] = plain_block[IP_table[i + left_block.size()] - 1];// �û����left����
	}
	return 0;
}

// 16�ֵ���
// �ӽ�������
int Turn(HBlock& left_block, HBlock& right_block, const Code& subkey)
{
	Code Extend_code;//48λ���ݿ�
	HBlock P_code;//32λ���ݿ�

	// �����Ұ벿�ֽ���F��������	
	//���Ұ벿��ͨ��E����չ�任Ϊ48λ
	for (int i = 0; i < Extend_code.size(); ++i)
	{
		Extend_code[i] = right_block[Extend_table[i] - 1];//��չ�û�
	}
	// ��չ��ɺ������Կ�����㣬
	// ��ѡ����չ��������� 48bit ��Ϊ���룬�� 48bit ������Կ����������㣬�������Ϊѡ��ѹ�����㣨S �У������롣

	Extend_code ^= subkey;//������Կ���

	//ѡ��ѹ�����㣨S�У�

	bitset<4> col;//S�е���
	bitset<2> row;//S�е���

	// 8��S�У�ÿ��S����4��16�У�
	// ÿ��S������Ϊ6bit�����4bit��
	// �ڲ��ǰ����Ҫ����Կ������õ���48bit�����Ϊ8�飬ÿ��6 bit���ֱ����8��S�б����������Ϊ���룬�õ���8��4bit����������䴮��һ���32bit�����Ϊ�û����㣨P�У������롣

	// S���������£�S�������6bit b1b2b3b4b5b6��ȡ��һλb1�����һλb6��ɶ�������r = b1b6������λb2b3b4b5��ɶ�������c = b2b3b4b5��
	// ���Ҷ�Ӧ��S�б��ҵ���r + 1��c + 1�е�����תΪ�����ƺ�Ϊ��S������Ҫ����Ľ����

	for (int i = 0; i < 8; ++i)
	{
		row[0] = Extend_code[6 * i];//�б�
		row[1] = Extend_code[6 * i + 5];

		col[0] = Extend_code[6 * i + 1];//�б�
		col[1] = Extend_code[6 * i + 2];
		col[2] = Extend_code[6 * i + 3];
		col[4] = Extend_code[6 * i + 4];

		// to_ulong()��row��col ����bitset����ת��Ϊunsigned long
		// S����4��16�У�ͨ������õ���Ӧ��ת��ֵ��S�е��ĸ�λ��
		bitset<4> temp(S_Box[i][row.to_ulong() * 16 + col.to_ulong()]);

		// ��8��S�м���õ��Ľ��ƴ�ӳ�һ��32 bit �ַ���
		// P���滻��Ҫ��S���滻��32λ�����Ϊ����
		for (int j = 0; j < temp.size(); ++j)
		{
			Extend_code[4 * i + j] = temp[j];
		}
	}
	// ��S�м���õ���32λ�����ΪP�е����룬�����滻���滻֮��λF�ֺ����Ľ��
	for (int i = 0; i < P_code.size(); ++i)
	{
		P_code[i] = Extend_code[P_table[i] - 1];//P���û�
	}
	// P�м���õ��Ľ������һ����ߵ� 32bit ���������Ϊ��һ�ֵ��Ұ벿��
	left_block ^= P_code;
	return 0;
}

//����������������
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

//���������������ݽ������ʼ�û���ƴ��һ�����ݿ�
int Reverse_IP(const HBlock& left_block, const HBlock& right_block, Block& block)
{
	for (int i = 0; i < block.size(); ++i)
	{
		if (IP_reverse_table[i] <= 32)
		{
			block[i] = right_block[IP_reverse_table[i] - 1];//��right���ֻ�ȡ����
		}
		else
		{
			block[i] = left_block[IP_reverse_table[i] - 32 - 1];//��left���ֻ�ȡ����
		}
	}
	return 0;
}

//��Կ�û�ѡ���PC-1����64bit��Կ�û�Ϊ56bit
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

//ѭ������ÿ���ƶ���λ��
// ��i=1��2��9��16ѭ�����ƶ�1λ, ����ѭ�����ƶ�2λ
const static unsigned char bit_leftshift[16] =
{
	1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 ,
	1 , 2 , 2 , 2 , 2 , 2 , 2 , 1
};

//��Կ�û�ѡ���PC-2����56 bit�еĵ�9��18��22��25��38��43��54λɾ������56bit��Կ�û���48bit
const static unsigned char PC2_table[48] =
{
	14 , 17 , 11 , 24 , 1 , 5 , 3 , 28 ,
	15 , 6 , 21 , 10 , 23 , 19 , 12 , 4 ,
	26 , 8 , 16 , 7 , 27 , 20 , 13 , 2 ,
	41 , 52 , 31 , 37 , 47 , 55 , 30 , 40 ,
	51 , 45 , 33 , 48 , 44 , 49 , 39 , 56 ,
	34 , 53 , 46 , 42 , 50 , 36 , 29 , 32
};

//��ȡbkey�����ĵ�n������Կ
Code getkey(const unsigned int N, const Block& bkey)
{
	//n������[0,15]֮��ȡֵ��bkeyΪ64λ��Կ

	Code subkey_pc2;//����ֵ,48λ����Կ
	Key subkey_pc1;//56λ��Կ

	unsigned int pc1_len = subkey_pc1.size();// 56
	unsigned int pc2_len = subkey_pc2.size();// 48

	//PC-1�û����õ�56λ��Կ
	for (int i = 0; i < 56; ++i)
	{
		subkey_pc1[i] = bkey[PC1_table[i] - 1];
	}

	// �����56bit�����ֳ����飬ÿ��28bit���ֱ����C�Ĵ�����pc1_table���ϰ벿�֣���D�Ĵ�����pc1_table���°벿�֣���׼������ѭ������

	for (int i = 0; i <= N; ++i)
	{
		//��N����Կ������Ҫѭ�����Ƶ�λ��Ϊbit_leftshift�ĵ�N��Ԫ�ص�ֵ

		int shift = bit_leftshift[N]; // ��ȡ��ǰ�ִε�����λ��

		// ������28λ���ֱַ����ѭ������
		// 0Ϊ��벿�֣�1Ϊ�Ұ벿��
		for (int part = 0; part < 2; ++part)
		{
			unsigned char temp[28]; // ��ʱ�洢���ƺ�Ľ��
			for (int i = 0; i < 28; ++i)
			{
				int new_index = (i + shift) % 28; // �������ƺ����λ��
				temp[new_index] = subkey_pc1[part * 28 + i]; // ִ������
			}
			// ����ʱ�����еĽ�����ƻ�ԭ����
			for (int i = 0; i < 28; ++i)
			{
				subkey_pc1[part * 28 + i] = temp[i];
			}
		}
	}
	// PC-2�û�
	// ��56 bit�еĵ�9��18��22��25��38��43��54λɾ��������λ�ð���PC2_table�û�λ�ã���56bit��Կ�û���48bit
	for (int i = 0; i < subkey_pc2.size(); ++i)
	{
		subkey_pc2[i] = subkey_pc1[PC2_table[i] - 1];
	}
	return subkey_pc2;
}

//�ӽ�������
int des(Block& block, Block& bkey, const Method method)
{
	//blockΪ���ݿ飬bkeyΪ64λ��Կ
	HBlock left_block, right_block;//���Ҳ���
	IP(block, left_block, right_block);//��ʼ�û�

	switch (method)
	{
	case e://����
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
	case d://����
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
	Reverse_IP(left_block, right_block, block);//ĩ�û�
	return 0;
}

