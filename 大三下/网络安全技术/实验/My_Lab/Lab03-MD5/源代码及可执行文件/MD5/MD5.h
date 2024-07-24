#include<iostream>
#include<cstring>
#include<string>
#include<iostream>
#include<cmath>

using namespace std;

#define F(x, y, z) (((x) & (y)) | ((~x) & (z))) //F ����
#define G(x, y, z) (((x) & (z)) | ((y) & (~z))) //G ����
#define H(x, y, z) ((x) ^ (y) ^ (z)) //H ����
#define I(x, y, z) ((y) ^ ((x) | (~z))) //I ����
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))//ѭ������

// �������������е�FF��GG��HH��II����
// a��b��c��d��ʾ����������x��ʾһ��32λ���ӿ飬s��ʾѭ�����Ƶ�λ����ac��ʾ����
#define FF(a, b, c, d, x, s, ac) { (a) += F ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s));(a) += (b); }
#define GG(a, b, c, d, x, s, ac) { (a) += G ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }
#define HH(a, b, c, d, x, s, ac) { (a) += H ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }
#define II(a, b, c, d, x, s, ac) { (a) += I ((b), (c), (d)) + (x) + ac; (a) = ROTATE_LEFT ((a), (s)); (a) += (b); }

//ѹ������ÿ��ÿ����A�ֿ�ѭ�����Ƶ�λ��
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

//������T
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
	uint32_t state[4]; //���ڱ�ʾ 4 ����ʼ����
	uint32_t count[2]; //���ڼ�����count[0]��ʾ��λ��count[1]��ʾ��λ
	uint8_t buffer_block[64]; //���ڱ����������а��黮�ֺ�ʣ�µı�����
	uint8_t digest[16]; //���ڱ��� 128 ���س��ȵ�ժҪ
	bool is_finished; //���ڱ�־ժҪ��������Ƿ����

	static const uint8_t padding[64]; //���ڱ�����Ϣ�����������ݿ�
	static const char hex[16]; //���ڱ��� 16 ���Ƶ��ַ�

	MD5() { Reset(); }

    MD5(const string& str) { Reset(); }

	MD5(ifstream& in) { Reset(); }

	//�Ը������ȵ��ֽ������� MD5 ����
	void Update(const uint8_t* input, size_t length);

	//�Ը������ȵ����������� MD5 ����
	void Update(const void* input, size_t length);

	//�Ը������ȵ��ַ������� MD5 ����
	void Update(const string& str);

	void Update(ifstream& in); //���ļ��е����ݽ��� MD5 ����

	const uint8_t* GetDigest(); //�� MD5 ժҪ���ֽ�������ʽ���

	string Tostring(); //�� MD5 ժҪ���ַ�����ʽ���

	void Reset(); //���ó�ʼ����

	void Stop(); //������ֹժҪ������̣����ժҪ

	void Transform(const uint8_t block[64]); //����Ϣ������� MD5 ����

	//��˫����ת��Ϊ�ֽ���
	void Encode(const uint32_t* input, uint8_t* output, size_t length);

	//���ֽ���ת��Ϊ˫����
	void Decode(const uint8_t* input, uint32_t* output, size_t length);

	//���ֽ�������ʮ�������ַ�����ʽ���
	string BytesToHexString(const uint8_t* input, size_t length);
	
};
