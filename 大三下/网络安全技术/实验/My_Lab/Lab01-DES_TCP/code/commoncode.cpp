#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <mutex>
// #include <WinSock2.h>
// #include <ws2tcpip.h>

#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <netinet/in.h>
// #include "DES.cpp"
// #include"commoncode.h"

// #pragma comment(lib, "ws2_32.lib")

using namespace std;

// 初始化WinSock库
/* int initializeWinSock()
{
	cout << "commoncode: Initializing WinSock..." << endl;
	WSADATA wsaData; //wsaData用来存储系统传回的关于WINSOCK的资料.
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)//MAKEWORD(2, 2)表示使用WINSOCK2版本.
	{
		cout << "commoncode: WSAStartup failed." << endl;
		return -1;
	}
	else
	{
		cout << "commoncode: WSAStartup is successful." << endl;
		return 0;
	}
}*/

// 清理WinSock库资源
/* void cleanupWinSock()
{
	cout << "commoncode: Cleaning up WinSock..." << endl;
	WSACleanup();
}*/

// 关闭套接字
void closeSocket(int socket)
{
	cout << "commoncode: Closing socket..." << endl;
	close(socket);
}

// 创建服务器套接字
int createServerSocket(int port)
{
	cout << "commoncode: Creating server socket..." << endl;
	// 创建一个套接字，使用IPv4地址族(AF_INET)，流式套接字类型(SOCK_STREAM)，和TCP协议(IPPROTO_TCP)
	int serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (serverSocket < 0)
	{
		cerr << "commoncode: Create socket failed. Error: " << strerror(errno) << endl;
		return -1;// 返回-1表示创建套接字失败
	}

	// 配置服务器地址结构
	sockaddr_in serverAddr;//通用套接字地址
	serverAddr.sin_family = AF_INET;// 地址族，IPv4
	serverAddr.sin_port = htons(port);//端口号，需要使用htons函数将端口号从主机字节序转换为网络字节序
	serverAddr.sin_addr.s_addr = INADDR_ANY;//ip地址，INADDR_ANY 表示套接字可以接受来自本机上的所有网络接口（即所有可用的 IP 地址）的连接请求。

	// 设置SO_REUSEADDR选项，用于任意类型、任意状态套接口的设置选项值，以允许套接字地址被重用
	/*int reuse = 1;
	if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(int)) == -1)
	{
		cerr << "commoncode: Setsockopt failed. Error: " << strerror(errno) << endl;
		closeSocket(serverSocket);
		return -4;// 返回-4表示设置选项失败
	}
	else
	{
		cout << "commoncode: Setsockopt." << endl;
	}*/

	//将套接字绑定到指定的端口和IP地址
	if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
	{
		cerr << "commoncode: Bind failed. Error: " << strerror(errno) << endl;
		closeSocket(serverSocket);
		return -2;
	}
	else
	{
		cout << "commoncode: Bind." << endl;
	}

	// 开始监听连接请求，最多允许10个等待连接
	if (listen(serverSocket, 10) < 0)
	{
		cerr << "commoncode: Listen failed. Error: " << strerror(errno) << endl;
		closeSocket(serverSocket);
		return -3;// 返回-3表示监听失败
	}
	else
	{
		cout << "commoncode: Listen." << endl;
	}

	return serverSocket;// 返回创建的服务器套接字
}

// 创建客户端套接字
int createClientSocket(const char* serverIP, int serverPort)
{
	int clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (clientSocket == -1)
	{
		cerr << "commoncode: Failed to create a client socket. Error: " << strerror(errno) << endl;
		// cleanupWinSock();
		return -2;
	}

	// 连接服务器
	sockaddr_in serverAddr{};
	bzero(&serverAddr, sizeof(serverAddr));
	serverAddr.sin_family = AF_INET;
	inet_pton(AF_INET, serverIP, &serverAddr.sin_addr);
	serverAddr.sin_port = htons(serverPort);

	// 尝试连接到服务器
	if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
	{
		cerr << "commoncode: Failed to connect to the server. Error: " << strerror(errno) << endl;
		closeSocket(clientSocket);
		// cleanupWinSock();
		return -4;
	}
	else
	{
		cout << "commoncode: Connect to the server." << endl;
	}

	return clientSocket;
}

// 接受客户端连接
int acceptClient(int serverSocket, sockaddr_in* clientAddr, socklen_t* clientAddrLen)
{
	cout << "commoncode: Accepting client connection..." << endl;
	// 使用 accept 函数接受客户端的连接请求，返回一个新的套接字用于与客户端通信
	int clientSocket = accept(serverSocket, (struct sockaddr*)clientAddr, clientAddrLen);
	// 检查是否接受连接请求失败
	if (clientSocket < 0)
	{
		cerr << "commoncode: Accept failed. Error: " << strerror(errno) << endl;
		return -1;// 返回-1表示接受连接失败
	}
	cout << "commoncode: Client connected." << endl;
	return clientSocket;// 返回与客户端通信的套接字
}

// 发送消息到套接字
int sendMessage(int socket, const char* message)
{
	cout << "commoncode: Sending message..." << endl;
	int bytesSent = send(socket, message, sizeof(message), 0);
	if (bytesSent == -1)
	{
		cerr << "commoncode: Error sending message. Error code: " << strerror(errno) << endl;
	}
	return bytesSent;
}

// 从套接字接收消息
int receiveMessage(int socket, char* message)
{
	cout << "commoncode: Receiving message..." << endl;
	int bytesRead = recv(socket, message, sizeof(message), 0);
	if (bytesRead == -1)
	{
		cerr << "commoncode: Error receiving message. Error code: " << strerror(errno) << endl;
	}
	return bytesRead;
}

string getTime()
{
	// 获取系统时间
	time_t rawtime;
	// struct tm timeInfo;
	char time_buffer[64];
	time(&rawtime);
	// localtime_s(&timeInfo, &rawtime);
	strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", localtime(&rawtime));

	return string(time_buffer);
}


