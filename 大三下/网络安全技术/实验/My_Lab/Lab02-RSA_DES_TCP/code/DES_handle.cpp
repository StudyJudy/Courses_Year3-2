#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include<cmath>
#include <limits>
#include <ios>
// #include <WinSock2.h>
// #include <ws2tcpip.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include<netinet/in.h>
#include <sys/types.h>
#include <stdio.h>
#include "DES.h"

#ifndef DES_handle_h
#define DES_handle_h

void StrFromBlock(char* str, Block& block)
{
    memset(str, 0, 8); // ��8���ֽ�ȫ����0
    for (int i = 0; i < block.size(); ++i)
    {
        // ��iλΪ1
        if (block[i] == 1)
        {
            // *((unsigned char*)(str)+i / 8) |= (1 << (7 - i % 8));
            size_t byteIndex = i / 8; // ���㵱ǰλ���ڵ��ֽ�����
            int pos = 7 - i % 8; // ������Ŀ���ֽڵ�λ��
            unsigned char mask = 1 << pos; // ����һ�����룬����������Ҫ���õ�λ��Ϊ1������λ��Ϊ0
            str[byteIndex] |= mask; // ʹ������������Ӧ��λ
        }
    }
}

void BlockFromStr(Block& block, char* str)
{
    for (int i = 0; i < block.size(); ++i)
    {
        // if (0 != (*((unsigned char*)(str)+i / 8) & (1 << (7 - i % 8))))
        // ���㵱ǰλ���ڵ��ֽ�����
        int byteIndex = i / 8;
        // ����һ�����룬����������Ҫ����λ��Ϊ1������λ��Ϊ0
        unsigned char mask = 1 << (7 - i % 8);
        // ʹ��������str����Ӧ�ֽڵ�λ�Ƿ�Ϊ1
        if ((str[byteIndex] & mask) != 0)
        {
            block[i] = 1;
        }
        else
        {
            block[i] = 0;
        }
    }
}

Method method_en = e;
Method method_de = d;

const int BUFFER_SIZE = 2048;

void DecryptMsg(char* msg, Block bkey)
{
    cout << "DES_handle: Message decrypted!!!  " << endl;
    int decrypt_len = BUFFER_SIZE;

    // ����Ϣ�ֳ�8�ֽڵĿ���д���
    for (int i = 0; i < decrypt_len / 8 + 1; i++)
    {
        // �鵱ǰ���Ƿ�ȫΪ0��
        // ����ǣ�˵����Ϣ�Ѿ��������������Ĳ��֣���������ӡһ�����з������أ���������
        if (strncmp(msg + i * 8, "\0\0\0\0\0\0\0\0", 8) == 0)
        {
            // cout << endl;
            return;
        }
        Block block;
        // ���ַ���תΪ���ݿ�
        BlockFromStr(block, msg + i * 8);
        // ���ܵ�ǰ��
        des(block, bkey, method_de);  //decrypt
        char temp[8];
        StrFromBlock(temp, block);
        // �����ܺ�����ݿ�blockת�����ַ���������ת������ַ����洢��msg������
        StrFromBlock(msg + i * 8, block);

    }
}

void EncryptMsg(char* msg, Block bkey)
{
    cout << "DES_handle: Message encrypted!!!" << endl;
    int encrypt_len = strlen(msg);

    // ����Ϣ�ֳ�8�ֽڵĿ���д���
    for (int i = 0; i < encrypt_len / 8 + 1; i++)
    {
        Block block;
        // ���ַ���תΪ���ݿ�
        BlockFromStr(block, msg + i * 8);
        // ���ܵ�ǰ��
        des(block, bkey, method_en);  //encrypt
        // �����ܺ�����ݿ�blockת�����ַ���������ת������ַ����洢��msg������
        StrFromBlock(msg + i * 8, block);
    }
}

string DES_key_gen()
{
    string des_key;

    // �������DES��Կ
    srand((unsigned)time(NULL));
    for (int i = 0; i < 8; i++)
    {
        char temp = 65 + rand() % 26;
        des_key += temp;
    }
    return des_key;
    // cout << "Client: The plaintext of the DES key: " << des_key << endl;
}

#endif

