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
    memset(str, 0, 8); // 将8个字节全部置0
    for (int i = 0; i < block.size(); ++i)
    {
        // 第i位为1
        if (block[i] == 1)
        {
            // *((unsigned char*)(str)+i / 8) |= (1 << (7 - i % 8));
            size_t byteIndex = i / 8; // 计算当前位所在的字节索引
            int pos = 7 - i % 8; // 计算在目标字节的位置
            unsigned char mask = 1 << pos; // 创建一个掩码，该掩码在需要设置的位上为1，其他位上为0
            str[byteIndex] |= mask; // 使用掩码设置相应的位
        }
    }
}

void BlockFromStr(Block& block, char* str)
{
    for (int i = 0; i < block.size(); ++i)
    {
        // if (0 != (*((unsigned char*)(str)+i / 8) & (1 << (7 - i % 8))))
        // 计算当前位所在的字节索引
        int byteIndex = i / 8;
        // 创建一个掩码，该掩码在需要检查的位上为1，其他位上为0
        unsigned char mask = 1 << (7 - i % 8);
        // 使用掩码检查str中相应字节的位是否为1
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

    // 将消息分成8字节的块进行处理
    for (int i = 0; i < decrypt_len / 8 + 1; i++)
    {
        // 查当前块是否全为0。
        // 如果是，说明消息已经结束或者是填充的部分，函数将打印一个换行符并返回，结束处理
        if (strncmp(msg + i * 8, "\0\0\0\0\0\0\0\0", 8) == 0)
        {
            // cout << endl;
            return;
        }
        Block block;
        // 将字符串转为数据块
        BlockFromStr(block, msg + i * 8);
        // 解密当前块
        des(block, bkey, method_de);  //decrypt
        char temp[8];
        StrFromBlock(temp, block);
        // 将加密后的数据块block转换成字符串，并将转换后的字符串存储到msg数组中
        StrFromBlock(msg + i * 8, block);

    }
}

void EncryptMsg(char* msg, Block bkey)
{
    cout << "DES_handle: Message encrypted!!!" << endl;
    int encrypt_len = strlen(msg);

    // 将消息分成8字节的块进行处理
    for (int i = 0; i < encrypt_len / 8 + 1; i++)
    {
        Block block;
        // 将字符串转为数据块
        BlockFromStr(block, msg + i * 8);
        // 加密当前块
        des(block, bkey, method_en);  //encrypt
        // 将解密后的数据块block转换成字符串，并将转换后的字符串存储到msg数组中
        StrFromBlock(msg + i * 8, block);
    }
}

string DES_key_gen()
{
    string des_key;

    // 随机生成DES密钥
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

