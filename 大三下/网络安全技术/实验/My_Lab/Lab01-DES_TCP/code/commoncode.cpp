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

// ��ʼ��WinSock��
/* int initializeWinSock()
{
	cout << "commoncode: Initializing WinSock..." << endl;
	WSADATA wsaData; //wsaData�����洢ϵͳ���صĹ���WINSOCK������.
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)//MAKEWORD(2, 2)��ʾʹ��WINSOCK2�汾.
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

// ����WinSock����Դ
/* void cleanupWinSock()
{
	cout << "commoncode: Cleaning up WinSock..." << endl;
	WSACleanup();
}*/

// �ر��׽���
void closeSocket(int socket)
{
	cout << "commoncode: Closing socket..." << endl;
	close(socket);
}

// �����������׽���
int createServerSocket(int port)
{
	cout << "commoncode: Creating server socket..." << endl;
	// ����һ���׽��֣�ʹ��IPv4��ַ��(AF_INET)����ʽ�׽�������(SOCK_STREAM)����TCPЭ��(IPPROTO_TCP)
	int serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (serverSocket < 0)
	{
		cerr << "commoncode: Create socket failed. Error: " << strerror(errno) << endl;
		return -1;// ����-1��ʾ�����׽���ʧ��
	}

	// ���÷�������ַ�ṹ
	sockaddr_in serverAddr;//ͨ���׽��ֵ�ַ
	serverAddr.sin_family = AF_INET;// ��ַ�壬IPv4
	serverAddr.sin_port = htons(port);//�˿ںţ���Ҫʹ��htons�������˿ںŴ������ֽ���ת��Ϊ�����ֽ���
	serverAddr.sin_addr.s_addr = INADDR_ANY;//ip��ַ��INADDR_ANY ��ʾ�׽��ֿ��Խ������Ա����ϵ���������ӿڣ������п��õ� IP ��ַ������������

	// ����SO_REUSEADDRѡ������������͡�����״̬�׽ӿڵ�����ѡ��ֵ���������׽��ֵ�ַ������
	/*int reuse = 1;
	if (setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(int)) == -1)
	{
		cerr << "commoncode: Setsockopt failed. Error: " << strerror(errno) << endl;
		closeSocket(serverSocket);
		return -4;// ����-4��ʾ����ѡ��ʧ��
	}
	else
	{
		cout << "commoncode: Setsockopt." << endl;
	}*/

	//���׽��ְ󶨵�ָ���Ķ˿ں�IP��ַ
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

	// ��ʼ�������������������10���ȴ�����
	if (listen(serverSocket, 10) < 0)
	{
		cerr << "commoncode: Listen failed. Error: " << strerror(errno) << endl;
		closeSocket(serverSocket);
		return -3;// ����-3��ʾ����ʧ��
	}
	else
	{
		cout << "commoncode: Listen." << endl;
	}

	return serverSocket;// ���ش����ķ������׽���
}

// �����ͻ����׽���
int createClientSocket(const char* serverIP, int serverPort)
{
	int clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (clientSocket == -1)
	{
		cerr << "commoncode: Failed to create a client socket. Error: " << strerror(errno) << endl;
		// cleanupWinSock();
		return -2;
	}

	// ���ӷ�����
	sockaddr_in serverAddr{};
	bzero(&serverAddr, sizeof(serverAddr));
	serverAddr.sin_family = AF_INET;
	inet_pton(AF_INET, serverIP, &serverAddr.sin_addr);
	serverAddr.sin_port = htons(serverPort);

	// �������ӵ�������
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

// ���ܿͻ�������
int acceptClient(int serverSocket, sockaddr_in* clientAddr, socklen_t* clientAddrLen)
{
	cout << "commoncode: Accepting client connection..." << endl;
	// ʹ�� accept �������ܿͻ��˵��������󣬷���һ���µ��׽���������ͻ���ͨ��
	int clientSocket = accept(serverSocket, (struct sockaddr*)clientAddr, clientAddrLen);
	// ����Ƿ������������ʧ��
	if (clientSocket < 0)
	{
		cerr << "commoncode: Accept failed. Error: " << strerror(errno) << endl;
		return -1;// ����-1��ʾ��������ʧ��
	}
	cout << "commoncode: Client connected." << endl;
	return clientSocket;// ������ͻ���ͨ�ŵ��׽���
}

// ������Ϣ���׽���
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

// ���׽��ֽ�����Ϣ
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
	// ��ȡϵͳʱ��
	time_t rawtime;
	// struct tm timeInfo;
	char time_buffer[64];
	time(&rawtime);
	// localtime_s(&timeInfo, &rawtime);
	strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", localtime(&rawtime));

	return string(time_buffer);
}


