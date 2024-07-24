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

// extern int initializeWinSock();//���ڳ�ʼ��WinSock��Windows�׽��֣���
// extern void cleanupWinSock();//��������WinSock��
extern int createServerSocket(int port);//���ڴ����������׽��֣�����Ĳ����Ƕ˿ں�
extern int acceptClient(int serverSocket, sockaddr_in* clientAddr, socklen_t* clientAddrLen);//���ڽ��ܿͻ����������󣬲����ؿͻ��˵��׽�����Ϣ��
extern int sendMessage(int socket, const char* message);//���ڷ���������Ϣ
extern int receiveMessage(int socket, char* message);//���ڽ���������Ϣ
extern void closeSocket(int socket);//���ڹر��׽���
extern int createClientSocket(const char* serverIP, int serverPort);//�����ͻ����׽���
extern string getTime();//���ڻ�ȡ��ǰʱ��

#endif // COMMONLIBRARY_H