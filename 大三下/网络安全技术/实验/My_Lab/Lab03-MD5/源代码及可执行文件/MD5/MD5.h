#include<iostream>
#include<cstring>
#include<string>
#include<iostream>
#include<cmath>

using namespace std;

#define F(x, y, z) (((x) & (y)) | ((~x) & (z))) //F 函数
#define G(x, y, z) (((x) & (z)) | ((y) & (~z))) //G 函数
#define H(x, y, z) ((x) ^ (y) ^ (z)) //H 函数
#define I(x, y, z) ((y) ^ ((x) | (~z))) //I 函数
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))//循环左移

// 定义四轮运算中的FF、GG、HH、II函数
// a、b、c、d表示计算向量，x表示一个32位的子块，s表示循环左移的位数，ac表示弧度
#define FF(a, b, c, d, x, s, ac) { (a) += F ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s));(a) += (b); }
#define GG(a, b, c, d, x, s, ac) { (a) += G ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }
#define HH(a, b, c, d, x, s, ac) { (a) += H ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }
#define II(a, b, c, d, x, s, ac) { (a) += I ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }

//压缩函数每轮每步中A分块循环左移的位数
const unsigned S[64] =
{
	7, 12, 17, 22,  7, 12, 17, 22,  
	7, 12, 17, 22,  7, 12, 17, 22,

	5,  9, 14, 20,  5,  9, 14, 20,  
	5,  9, 14, 20,  5,  9, 14, 20,

	4, 11, 16, 23,  4, 11, 16, 23,  
	4, 11, 16, 23,  4, 11, 16, 23,

	6, 10, 15, 21,  6, 10, 15, 21, 
	6, 10, 15, 21,  6, 10, 15, 21
};

//常数表T
const unsigned T[64] =
{
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

class MD5
{
public:
	uint32_t state[4]; //用于表示 4 个初始向量
	uint32_t count[2]; //用于计数，count[0]表示低位，count[1]表示高位
	uint8_t buffer_block[64]; //用于保存计算过程中按块划分后剩下的比特流
	uint8_t digest[16]; //用于保存 128 比特长度的摘要
	bool is_finished; //用于标志摘要计算过程是否结束

	static const uint8_t padding[64]; //用于保存消息后面填充的数据块
	static const char hex[16]; //用于保存 16 进制的字符

	MD5() { Reset(); }

    MD5(const string& str) { Reset(); }

	MD5(ifstream& in) { Reset(); }

	//对给定长度的字节流进行 MD5 运算
	void Update(const uint8_t* input, size_t length);

	//对给定长度的输入流进行 MD5 运算
	void Update(const void* input, size_t length);

	//对给定长度的字符串进行 MD5 运算
	void Update(const string& str);

	void Update(ifstream& in); //对文件中的内容进行 MD5 运算

	const uint8_t* GetDigest(); //将 MD5 摘要以字节流的形式输出

	string Tostring(); //将 MD5 摘要以字符串形式输出

	void Reset(); //重置初始变量

	void Stop(); //用于终止摘要计算过程，输出摘要

	void Transform(const uint8_t block[64]); //对消息分组进行 MD5 运算

	//将双字流转换为字节流
	void Encode(const uint32_t* input, uint8_t* output, size_t length);

	//将字节流转换为双字流
	void Decode(const uint8_t* input, uint32_t* output, size_t length);

	//将字节流按照十六进制字符串形式输出
	string BytesToHexString(const uint8_t* input, size_t length);
	
};
