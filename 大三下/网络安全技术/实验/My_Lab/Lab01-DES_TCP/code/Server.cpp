// SERVER.CPP

#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <cstring>
#include <mutex>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <ctime>
#include <sys/types.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <cstdlib>
#include <stdlib.h>
// #include <WinSock2.h>
// #include <ws2tcpip.h>

#include "DES_handle.h"
#include"commoncode.h"
// #pragma comment(lib, "ws2_32.lib")

using namespace std;

typedef struct CLIENT
{
    int client_Socket;//�ͻ����׽���  
    char clientname[2048];//�ͻ���
    char time_stamp[64];//��ǰʱ����ַ���
    char content_buf[256];//�ͻ����յ���Ϣ�����������ʾ�����͵��ַ���
    int flag;//��ǰ�ͻ��˱��
}CLIENT;

//vector<int> clientSockets;// �洢���ӵĿͻ����׽���
CLIENT clients[20];//���������20��
//mutex mtx; // �����������ڱ��� clientSockets ������ȷ�����̷߳���ʱ��ͬ����
//vector<CLIENT> clients;

// int num = 0;//��ǰ�ͻ��˵ı��

int client_Socket;
bool exit_flag = false;
Block bkey; // ��Կ

void* Server_recv_thread(void*) 
{
	/*
	CLIENT* Client = new CLIENT; // ��������ʼ��һ�� CLIENT ָ��
	char timestamp[8196] = {};  // ʱ������������ַ�����ʱ���ʽ
	char content[BUFFER_SIZE];   // ���յ�������Ϣ����
	int i = 0;
	char tmp1[] = "[ ";
	char tmp2[] = " ]";
	char tmp3[] = " Received message from client: ";
	char tmp4[] = " message: ";
	char tmp5[] = "Server: ";

	while (!exit_flag)
	// {
		memset(content, 0, sizeof(content));
		//memset(timestamp, 0, sizeof(timestamp));

		//��������
		// int recv_time = recv(Client->client_Socket, timestamp, sizeof(timestamp), 0);
		// char* TimeStamp = dealMsg(timestamp, bkey);
		int recv_con = recv(Client->client_Socket, content, sizeof(content), 0);
		char* Content = dealMsg(content, bkey);
		//int recv_time = receiveMessage(Client->client, timestamp);
		//int ret_con = receiveMessage(Client->client, content);

		//ƴ�ӷ��������յ���Ϣ����ʾ��ʽ
		if (recv_con > 0)
		{
			memcpy(Client->content_buf, tmp5, sizeof(Client->content_buf));
			// memcpy(Client->content_buf, tmp1, sizeof(Client->content_buf));
			// memcpy(Client->content_buf, timestamp, sizeof(Client->content_buf));
			// memcpy(Client->content_buf, tmp2, sizeof(Client->content_buf));
			memcpy(Client->content_buf, tmp3, sizeof(Client->content_buf));
			// memcpy(Client->content_buf, Client->clientname, sizeof(Client->content_buf));
			memcpy(Client->content_buf, tmp4, sizeof(Client->content_buf));
			memcpy(Client->content_buf, Content, sizeof(Client->content_buf));
			//�������˵���ʾ
			cout << Client->content_buf << endl;

			//lock_guard<mutex> lock(mutex);//ʹ�û�������ȷ������̰߳�ȫ�ز��� clients vector,����������Դ clients
			if (strcmp(Content, "exit_request") != 0)
			{
				for (i = 0; i < num; i++)
				{
					if (i != Client->flag)
					{
						// ���ͻظ���Ϣ���ͻ���
						encryptMsg(Client->content_buf, bkey);
						int send_result = send(clients[i].client_Socket, Client->content_buf, strlen(Client->content_buf), 0);
						//sendMessage(Client->client, Client->buf);
						if (send_result == -1)// �����ʹ���
						{
							//cerr << "1Server: Failed to send message to client " << Client->clientname <<  ". Error: " << WSAGetLastError() << endl;
						}
						exit_flag = true;
					}

				}
			}
			// if (strcmp(content, "exit_request") != 0)
			if (strcmp(Content, "exit_request") != 0)
			{
				for (i = 0; i < num; i++)
				{
					if (i != Client->flag)
					{
						// ���ͻظ���Ϣ���ͻ���
						encryptMsg(Client->content_buf, bkey);
						int send_result = send(clients[i].client_Socket, Client->content_buf, strlen(Client->content_buf), 0);
						//sendMessage(Client->client, Client->buf);
						if (send_result == -1)// �����ʹ���
						{
							//cerr << "1Server: Failed to send message to client " << Client->clientname <<  ". Error: " << WSAGetLastError() << endl;
						}
					}

				}
			}
			else
			{
				for (i = 0; i < num; i++)
				{
					if (i == Client->flag)
					{
						encryptMsg(Content, bkey);
						int send_result = send(clients[i].client_Socket, Content, strlen(content), 0);
						if (send_result == -1)// �����ʹ���
						{
							cerr << "Server: Failed to send message to client " << Client->clientname << ". Error: " << strerror(errno) << endl;

						}
					}
				}
			}
		}
		else
		{
			cout << "Server: Failed to receive message from client " << Client->clientname << ". Error: " << strerror(errno) << endl;
			break;
		}
	// }
	// return 0;
	*/
    
    char server_recv_buffer[BUFFER_SIZE]{};

    while (!exit_flag) 
    {
        // ���տͻ��˷��͵���Ϣ
        memset(server_recv_buffer, 0, BUFFER_SIZE);
        int recv_len = recv(client_Socket, (void*)server_recv_buffer, BUFFER_SIZE, 0);

        cout << "Server_recv_thread: The ciphertext that received from client: " << server_recv_buffer << endl;

        if (recv_len < 0)
        {
            cout << "Failed to receive message from client." << endl;
            break;
        }
        if (recv_len == 0) 
        {
            cout << "Client closed the connection." << endl;
            exit(0);
            break;
        }

		// char* recv_decrypt_buffer = DecryptMsg(Server_recv_buffer, bkey);
		// ��ʾ�ͻ��˷��͵���Ϣ
		// cout << "Received message from client: plaintext: " << recv_decrypt_buffer << endl;
		
        // ��ʾ�ͻ��˷��͵���Ϣ
		DecryptMsg(server_recv_buffer, bkey);
        cout << "Server_recv_thread: plaintext that the server received from client: ";		
		cout << server_recv_buffer << endl;
    }
    pthread_exit(NULL);
}

void* Server_send_thread(void*) 
{
    char server_send_buffer[BUFFER_SIZE];

    while (!exit_flag) 
    {
        // ������Ϣ���ͻ���
        memset(server_send_buffer, 0, BUFFER_SIZE);
        cout << "Please input the message that the server will transmit: " << endl;
        cin.getline(server_send_buffer, BUFFER_SIZE - 1);

        if (strcmp(server_send_buffer, "exit") == 0) 
        {
            exit_flag = true;
            break;
        }
        EncryptMsg(server_send_buffer, bkey);

        cout << "Server_send_thread: ciphertext that the server will transmit: " << server_send_buffer << endl;

        if (send(client_Socket, (void*)server_send_buffer, BUFFER_SIZE, 0) < 0) 
        {
            cout << "Error: failed to send message to client." << endl;
            break;
        }
    }
    pthread_exit(NULL);
}


int main() 
{
    //��ʼ��WinSock��
     /*if (initializeWinSock() != 0)
     {
         cerr << "Server: WSAStartup failed. Error: " << WSAGetLastError() << endl;
         return 1;
     }*/

     //����һ��Socket�������ͻ��˵���������
    int serverPort = 5500;
    int serverSocket = createServerSocket(serverPort);

    if (serverSocket < 0)
    {
        cerr << "Server: Failed to create a server socket. Error: " << strerror(errno) << endl;
        // cleanupWinSock();
        return 1;
    }
    else
    {
        cerr << "Server: Create a server socket. " << endl;
    }

    //�ȴ��ͻ��˵�����
    sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);

    client_Socket = acceptClient(serverSocket, &clientAddr, &clientAddrLen);

    BlockFromStr(bkey, "deskey"); // ��ȡ��Կ

    pthread_t sev_recv, sev_send;
    pthread_create(&sev_recv, NULL, Server_recv_thread, NULL);
    pthread_create(&sev_send, NULL, Server_send_thread, NULL);

    pthread_join(sev_recv, NULL);
    pthread_join(sev_send, NULL);

	/*
	while (true)
	{
		//CLIENT newClient;
		//newClient.client = acceptClient(serverSocket, &clientAddr, &clientAddrLen);
		//���ܿͻ��˵�����
		clients[num].client_Socket = acceptClient(serverSocket, &clientAddr, &clientAddrLen);
		clients[num].flag = num;

		BlockFromStr(bkey, "aeskey"); // ��ȡ��Կ

		//�����û����������Ƿ��ܳɹ�����
		/*char clientname_temp[1024];
		memset(clientname_temp , 0, sizeof(clientname_temp));

		// recv(clients[num].client_Socket, clients[num].clientname, sizeof(clients[num].clientname), 0); //�����û���
		int recv_name_length = recv(clients[num].client_Socket, (void*)clientname_temp, sizeof(clientname_temp), 0);

		cout << "Server: cipher_name_text: " << clientname_temp << endl;
		// clients[num].clientname = dealMsg(clients[num].clientname, bkey);
		strcpy(clients[num].clientname, dealMsg(clientname_temp, bkey));

		cout << "Server: plaintext: " << clients[num].clientname << " connected" << endl;
		//receiveMessage(clients[num].client, clients[num].clientname);

		//clients.push_back();
		// DWORD  lpThreadID;
		// HANDLE clientThread;

		//�����̴߳�����Ϣ������ת��
		// clientThread = CreateThread(nullptr, 0, (LPTHREAD_START_ROUTINE)handleClient, &clients[num], 0, &lpThreadID);

		pthread_t tid_recv, tid_send;

		pthread_create(&tid_recv, NULL, Server_recv_thread, NULL);
		// pthread_create(&tid_send, NULL, Server_send_thread, NULL);

		pthread_join(tid_recv, NULL);
		// pthread_join(tid_send, NULL);

		/*if (clientThread == nullptr)
		{
			cerr << "Server: Failed to create a thread. Error: " << GetCurrentThreadId() << endl;
		}
		else
		{
			cout << "Server:  a thread." << endl;
		}*/
		// ��Ǽ�һ
		// num++;
		//lock_guard<mutex> lock(mtx); // ������������ȷ���̰߳�ȫ
		//clients.push_back(newClient);
	// }

    // �ر����Ӻ�socket
    close(client_Socket);
    close(serverSocket);

	// cleanupWinSock(); // ����WinSock��

    return 0;
}