#pragma once
#ifndef COMMONLIBRARY_H
#define COMMONLIBRARY_H

// #include <WinSock2.h>
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <netinet/in.h>
// #include "DES.cpp"
using namespace std;

// extern int initializeWinSock();//用于初始化WinSock（Windows套接字）库
// extern void cleanupWinSock();//用于清理WinSock库
extern int createServerSocket(int port);//用于创建服务器套接字，传入的参数是端口号
extern int acceptClient(int serverSocket, sockaddr_in* clientAddr, socklen_t* clientAddrLen);//用于接受客户端连接请求，并返回客户端的套接字信息。
extern int sendMessage(int socket, const char* message);//用于发送聊天消息
extern int receiveMessage(int socket, char* message);//用于接收聊天消息
extern void closeSocket(int socket);//用于关闭套接字
extern int createClientSocket(const char* serverIP, int serverPort);//创建客户端套接字
extern string getTime();//用于获取当前时间

#endif // COMMONLIBRARY_H