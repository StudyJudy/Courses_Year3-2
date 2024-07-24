#include<iostream>
#include<fstream>
#include<cstring>
#include<string>
#include<sstream>
#include<iomanip>
#include "MD5.h"

using namespace std;

const uint8_t MD5::padding[64] = 
{
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

//��˫����ת��Ϊ�ֽ���
void MD5::Encode(const uint32_t* input, uint8_t* output, size_t length)
{
    for (size_t i = 0, j = 0; j < length; ++i, j += 4) 
    {
        output[j] = input[i] & 0xff;
        output[j + 1] = (input[i] >> 8) & 0xff;
        output[j + 2] = (input[i] >> 16) & 0xff;
        output[j + 3] = (input[i] >> 24) & 0xff;
    }
}

//���ֽ���ת��Ϊ˫����
void MD5::Decode(const uint8_t* input, uint32_t* output, size_t length)
{
    for (size_t i = 0, j = 0; j < length; ++i, j += 4)
    {
        output[i] = (uint32_t)input[j] | 
            ((uint32_t)input[j + 1] << 8) |
            ((uint32_t)input[j + 2] << 16) | 
            ((uint32_t)input[j + 3] << 24);
    }
}

void MD5::Reset()
{
    count[0] = count[1] = 0;
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;
    memset(buffer_block, 0, 64);
    memset(digest, 0, 16);
    is_finished = false;
}

void MD5::Stop() 
{
    is_finished = true;
}

//�Ը������ȵ����������� MD5 ����
void MD5::Update(const uint8_t* input, size_t length)
{
    uint32_t i, index, partLen;
    //����ֹͣ��ʶ
    is_finished = false;
    //���� buffer �Ѿ���ŵ��ֽ���
    index = (uint32_t)((count[0] >> 3) & 0x3f);
    //���¼����� count�������������ĳ��ȼ��ϼ�����ԭ�е�ֵ
    if ((count[0] += ((uint32_t)length << 3)) < ((uint32_t)length << 3)) //�ж��Ƿ��λ
    {
        count[1]++;
    }
    count[1] += ((uint32_t)length >> 29);
    //��� buffer ��ʣ��ĳ���
    partLen = 64 - index;
    //�����ݿ������� MD5 ����
    if (length >= partLen)
    {
        memcpy(&buffer_block[index], input, partLen);
        Transform(buffer_block);

        for (i = partLen; i + 63 < length; i += 64)
        {
            Transform(&input[i]);
        }
        index = 0;
    }
    else 
    {
        i = 0;
    }
    //������ 64 �ֽڵ����ݸ��Ƶ� buffer_block ��
    memcpy(&buffer_block[index], &input[i], length - i);
}

//�Ը������ȵ����������� MD5 ����
void MD5::Update(const void* input, size_t length)
{
    // Reset();
    const uint8_t* input_bytes = static_cast<const uint8_t*>(input);
    Update(input_bytes, length);
}

//�Ը������ȵ��ַ������� MD5 ����
void MD5::Update(const string& str)
{
    // Reset();
    Update(reinterpret_cast<const uint8_t*>(str.c_str()), str.length());
}

//���ļ��е����ݽ��� MD5 ����
void MD5::Update(ifstream& in) 
{
    // Reset();
    char buffer[1024]; // ��������СΪ 64 �ֽ�
    while (!in.eof()) // ѭ����ȡ�ļ�ֱ���ļ�ĩβ
    {
        in.read(buffer, sizeof(buffer)); // ���ļ��ж�ȡ���ݵ�������
        size_t bytes_read = in.gcount(); // ��ȡʵ�ʶ�ȡ���ֽ���
        Update(reinterpret_cast<const uint8_t*>(buffer), bytes_read); // ���������е����ݸ��µ� MD5 �㷨��
    }
}

const uint8_t* MD5::GetDigest()
{
    if (!is_finished)
    {
        uint8_t bits[8];
        Encode(count, bits, 8);
        size_t index = count[0] / 8 % 64;
        size_t padLen = (index < 56) ? (56 - index) : (120 - index);
        Update(padding, padLen);
        Update(bits, 8);
        Encode(state, digest, 16);
        is_finished = true;
    }
    return digest;
}

//���ֽ�������ʮ�������ַ�����ʽ���
string MD5::BytesToHexString(const uint8_t* input, size_t length)
{
    stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < length; ++i) 
    {
        ss << std::setw(2) << static_cast<unsigned>(input[i]);
    }
    return ss.str();
}

string MD5::Tostring()
{
    const uint8_t* digest = GetDigest();
    return BytesToHexString(digest, 16);
}

void MD5::Transform(const uint8_t block[64])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    Decode(block, x, 64);

    // ��һ��
    FF(a, b, c, d, x[0], S[0], T[0]);
    FF(d, a, b, c, x[1], S[1], T[1]);
    FF(c, d, a, b, x[2], S[2], T[2]);
    FF(b, c, d, a, x[3], S[3], T[3]);
    FF(a, b, c, d, x[4], S[4], T[4]);
    FF(d, a, b, c, x[5], S[5], T[5]);
    FF(c, d, a, b, x[6], S[6], T[6]);
    FF(b, c, d, a, x[7], S[7], T[7]);
    FF(a, b, c, d, x[8], S[8], T[8]);
    FF(d, a, b, c, x[9], S[9], T[9]);
    FF(c, d, a, b, x[10], S[10], T[10]);
    FF(b, c, d, a, x[11], S[11], T[11]);
    FF(a, b, c, d, x[12], S[12], T[12]);
    FF(d, a, b, c, x[13], S[13], T[13]);
    FF(c, d, a, b, x[14], S[14], T[14]);
    FF(b, c, d, a, x[15], S[15], T[15]);

    // ��2��
    GG(a, b, c, d, x[1], S[16], T[16]);
    GG(d, a, b, c, x[6], S[17], T[17]);
    GG(c, d, a, b, x[11], S[18], T[18]);
    GG(b, c, d, a, x[0], S[19], T[19]);
    GG(a, b, c, d, x[5], S[20], T[20]);
    GG(d, a, b, c, x[10], S[21], T[21]);
    GG(c, d, a, b, x[15], S[22], T[22]);
    GG(b, c, d, a, x[4], S[23], T[23]);
    GG(a, b, c, d, x[9], S[24], T[24]);
    GG(d, a, b, c, x[14], S[25], T[25]);
    GG(c, d, a, b, x[3], S[26], T[26]);
    GG(b, c, d, a, x[8], S[27], T[27]);
    GG(a, b, c, d, x[13], S[28], T[28]);
    GG(d, a, b, c, x[2], S[29], T[29]);
    GG(c, d, a, b, x[7], S[30], T[30]);
    GG(b, c, d, a, x[12], S[31], T[31]);

    // ��3��
    HH(a, b, c, d, x[5], S[32], T[32]);
    HH(d, a, b, c, x[8], S[33], T[33]);
    HH(c, d, a, b, x[11], S[34], T[34]);
    HH(b, c, d, a, x[14], S[35], T[35]);
    HH(a, b, c, d, x[1], S[36], T[36]);
    HH(d, a, b, c, x[4], S[37], T[37]);
    HH(c, d, a, b, x[7], S[38], T[38]);
    HH(b, c, d, a, x[10], S[39], T[39]);
    HH(a, b, c, d, x[13], S[40], T[40]);
    HH(d, a, b, c, x[0], S[41], T[41]);
    HH(c, d, a, b, x[3], S[42], T[42]);
    HH(b, c, d, a, x[6], S[43], T[43]);
    HH(a, b, c, d, x[9], S[44], T[44]);
    HH(d, a, b, c, x[12], S[45], T[45]);
    HH(c, d, a, b, x[15], S[46], T[46]);
    HH(b, c, d, a, x[2], S[47], T[47]);

    // ��4��
    II(a, b, c, d, x[0], S[48], T[48]);
    II(d, a, b, c, x[7], S[49], T[49]);
    II(c, d, a, b, x[14], S[50], T[50]);
    II(b, c, d, a, x[5], S[51], T[51]);
    II(a, b, c, d, x[12], S[52], T[52]);
    II(d, a, b, c, x[3], S[53], T[53]);
    II(c, d, a, b, x[10], S[54], T[54]);
    II(b, c, d, a, x[1], S[55], T[55]);
    II(a, b, c, d, x[8], S[56], T[56]);
    II(d, a, b, c, x[15], S[57], T[57]);
    II(c, d, a, b, x[6], S[58], T[58]);
    II(b, c, d, a, x[13], S[59], T[59]);
    II(a, b, c, d, x[4], S[60], T[60]);
    II(d, a, b, c, x[11], S[61], T[61]);
    II(c, d, a, b, x[2], S[62], T[62]);
    II(b, c, d, a, x[9], S[63], T[63]);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;

    memset(x, 0, sizeof(x));
}

