// CLIENT.CPP

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <thread>
#include <cmath>
#include <limits>
#include <ios>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <errno.h>
#include <pthread.h>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
// #include <WinSock2.h>
// #include <ws2tcpip.h>
#include "commoncode.h"
#include "DES_handle.h"

// #pragma comment(lib, "ws2_32.lib")


using namespace std;
//HWND hEdit;
//HWND hButton;
wstring chatHistory;

int clientSocket;// ��clientSocket����Ϊȫ�ֱ���
bool exit_flag = false;
Block bkey; // ��Կ

/*
// �ͻ����̣߳����ڽ�����Ϣ
void *Client_Receive(void *)
{
	// int Client = (int)(LPVOID)lparam;
	int Client;
	char messagebuffer[8192] = { 0 };
	int recv_buffer = 0;

	while (true)
	{
		memset(messagebuffer, 0, sizeof(messagebuffer));
		//int rec=receiveMessage(Client, messagebuffer);
		recv_buffer = recv(Client, messagebuffer, sizeof(messagebuffer), 0);

		char* recv_dealMsg_buffer = DescyptMsg(messagebuffer, bkey);

		if (recv_buffer < 0)
		{
			cout << "Client: Failed to receive the message. Error: " << strerror(errno) << endl;
			break;
		}
		else
		{
			// if (strcmp(messagebuffer, "exit_request") == 0)
			if (strcmp(recv_dealMsg_buffer, "exit_request") == 0)
			{
				cout << "Client: Server confirmed exit. Closing the connection..." << endl;
				closeSocket(Client);
				// cleanupWinSock();
				break;
				return 0;
			}

			else
			{
				// cout << "Client: " << messagebuffer << "��Broadcast��" << endl;
				cout << "Client: " << recv_dealMsg_buffer << "��Broadcast��" << endl;
			}
		}

	}
	// �رտͻ����׽��ֺ��߳��˳�
	//closeSocket(Client);
	return 0;
}

char clientname[BUFFER_SIZE]{};

// �ͻ��˷����߳�
void* Client_Send(void*)
{
	// int clientSocket = (int)(LPVOID)lparam;
	int clientSocket;
	char message[256] = { 0 };
	char SendTime[64] = {};
	int send_time = 0;
	int send_message = 0;
	int send_exit = 0;
	char Exit[15];
	// strcpy_s(Exit, sizeof(Exit), "exit_request");
	memcpy(Exit, "exit_request", sizeof(Exit));

	while (true && (exitRequested == false))
	{
		memset(message, 0, sizeof(message));
		// memset(clientname, 0, sizeof(clientname));
		// memset(SendTime, 0, sizeof(SendTime));

		// string time = getTime();//��ȡ��ǰʱ��
		// strcpy_s(SendTime, sizeof(SendTime), time.c_str());
		// memcpy(SendTime, time.c_str(), sizeof(SendTime));

		// cin.ignore((numeric_limits<streamsize>::max)(), '\n');
		// cin.getline(message, 0, 256 - 1);

		// cout << SendTime << " " << clientname << ": ";
		cout << "Client: Enter your message (type 'exit' if you want to exit): ";
		cin.getline(message, sizeof(message) - 1);

		// cin.ignore((numeric_limits<streamsize>::max)(), '\n');

		if (strcmp(message, "exit") == 0)
		{
			// encryptMsg(SendTime, bkey);
			// send_time = send(clientSocket, SendTime, strlen(SendTime), 0);
			//send_message = send(clientSocket, message, strlen(message), 0);
			// ����������˳�������Ϣ��������
			encryptMsg(Exit, bkey);
			send_exit = send(clientSocket, Exit, strlen(Exit), 0);
			if (send_exit == -1)
			{
				cerr << "Client: Failed to send exit request. Error: " << strerror(errno) << endl;
				// ������Կ��ǽ��д������������·��ͻ�ر�����
			}
			else
			{
				cout << "Exit request sent." << endl;
			}
			//send(clientSocket, "exit_request", strlen("exit_request"), 0);
			exitRequested = true;
			return 0;
		}
		else
		{
			// encryptMsg(SendTime, bkey);
			encryptMsg(message, bkey);
			// send_time = send(clientSocket, SendTime, strlen(SendTime), 0);
			send_message = send(clientSocket, message, strlen(message), 0);
		}
	}
	return 0;
}*/

void* Client_recv_thread(void*)
{
    char client_recv_buffer[BUFFER_SIZE]{};
	// char receive_decrypt_buffer[BUFFER_SIZE]{};

    while (!exit_flag)
    {
        // ���շ��������͵���Ϣ
        memset(client_recv_buffer, 0, BUFFER_SIZE);
		// memset(receive_decrypt_buffer, 0, BUFFER_SIZE);

        int recv_len = recv(clientSocket, (void*)client_recv_buffer, BUFFER_SIZE, 0);
        cout << "Client_recv_thread: The ciphertext that received from server: " << client_recv_buffer << endl;
        if (recv_len < 0)
        {
            cout << "Failed to receive message from server." << endl;
            break;
        }
        if (recv_len == 0)
        {
            cout << "Server closed the connection." << endl;
            exit(0);
            break;
        }

		// char *receive_decrypt_buffer= DecryptMsg(receive_buffer, bkey);

		// ��ʾ���������͵���Ϣ
		// cout << "Received message from server: plaintext: " << receive_decrypt_buffer << endl;
		
        // ��ʾ���������͵���Ϣ
		DecryptMsg(client_recv_buffer, bkey);
        cout << "Client_recv_thread: plaintext that the client received from server:";	
		cout << client_recv_buffer << endl;
    }
    pthread_exit(NULL);
}

void* Client_send_thread(void*)
{
    char client_send_buffer[BUFFER_SIZE];

    while (!exit_flag)
    {
        // ������Ϣ��������
        memset(client_send_buffer, 0, BUFFER_SIZE);
        cout << "Please input the message that the client will transmit: " << endl;
        cin.getline(client_send_buffer, BUFFER_SIZE - 1);
        if (strcmp(client_send_buffer, "exit") == 0)
        {
            exit_flag = true;
            break;
        }
        EncryptMsg(client_send_buffer, bkey);

        cout << "Client_send_thread: ciphertext that the client will transmit: " << client_send_buffer << endl;

        if (send(clientSocket, (void*)client_send_buffer, BUFFER_SIZE, 0) < 0)
        {
            cout << "Error: failed to send message to server." << endl;
            break;
        }
    }
    pthread_exit(NULL);
}


int main(int argc, char* argv[])
{
	// ����Socket���ڿͻ��˴���һ��Socket�����ӷ�������
	// int clientSocket = createClientSocket("127.0.0.1", 5500);
	/*if ((clientSocket == -2) || (clientSocket == -3) || (clientSocket == -4))
	{
		closeSocket(clientSocket);// �رտͻ����׽���
		// cleanupWinSock();// ����WinSock��
		return 0;
	}*/

    const char* ip = "127.0.0.1";
    int Server_port = 5500;

    clientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (clientSocket < 0)
    {
        cout << "Error: failed to create socket." << endl;
        return -1;
    }

    // ���ӷ�����
    sockaddr_in serverAddr{};
    bzero(&serverAddr, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    inet_pton(AF_INET, ip, &serverAddr.sin_addr);
    serverAddr.sin_port = htons(Server_port);
    if (connect(clientSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
    {
        cout << "Error: failed to connect to server." << endl;
        close(clientSocket);
    }
    else
    {
        cout << "commoncode: Connect to the server." << endl;
    }

    cout << "########## message ##########" << endl;

    BlockFromStr(bkey, "deskey"); // ��ȡ��Կ

	//memset(clientname, 0, sizeof(clientname));
	/*cout << "Client: Enter your name: ";
	memset(clientname, 0, BUFFER_SIZE);
	cin.getline(clientname, BUFFER_SIZE - 1);
	// cin.ignore((numeric_limits<streamsize>::max)(), '\n');

	cout << "Client: cipher_name: " << clientname << endl;
	send(clientSocket, (void *)clientname, BUFFER_SIZE, 0);*/

	/*
	HANDLE  ThreadReceive, ThreadSend;

	ThreadReceive = CreateThread(NULL, 0, Client_Receive, (LPVOID)clientSocket, 0, NULL);
	ThreadSend = CreateThread(NULL, 0, Client_Send, (LPVOID)clientSocket, 0, NULL);

	// ʹ���߳�ͬ���ȴ��߳��˳�
	WaitForSingleObject(ThreadReceive, INFINITE);
	WaitForSingleObject(ThreadSend, INFINITE);

	//���̹رգ�����������
	if (WaitForSingleObject(ThreadReceive, INFINITE) == WAIT_OBJECT_0 ||
		WaitForSingleObject(ThreadSend, INFINITE) == WAIT_OBJECT_0)
	{
		CloseHandle(ThreadReceive);
		CloseHandle(ThreadSend);
		return 0;
	}*/

    pthread_t cli_recv, cli_send;
    pthread_create(&cli_recv, NULL, Client_recv_thread, NULL);
    pthread_create(&cli_send, NULL, Client_send_thread, NULL);

    pthread_join(cli_recv, NULL);
    pthread_join(cli_send, NULL);

	//�����˳���ȷ���ͻ����ܹ������˳����ͷ���Դ��
   
    close(clientSocket); // �ر����Ӻ�socket 
    //closeSocket(clientSocket);// �رտͻ����׽���

	// cleanupWinSock();// ����WinSock��

    return 0;
}


