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
#include <sys/epoll.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <errno.h>
#include <pthread.h>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <limits>
// #include <time>
#include <ctime>
#include <aio.h>
#include <future>
// #include <WinSock2.h>
// #include <ws2tcpip.h>
#include "commoncode.h"
#include "DES_handle.cpp"
#include "RSA.cpp"

// #pragma comment(lib, "ws2_32.lib")


using namespace std;
//HWND hEdit;
//HWND hButton;
wstring chatHistory;

int clientSocket;// ��clientSocket����Ϊȫ�ֱ���
bool exit_flag = false;
Block bkey; // ��Կ
const static int epoll_timeout = -1;
bool getrsakey = false;

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

void Client_recv_thread(int clientSocket)
{
	char client_recv_buffer[BUFFER_SIZE]{};
	// char receive_decrypt_buffer[BUFFER_SIZE]{};

	// while (!exit_flag)
	// {
		// ���շ��������͵���Ϣ
	memset(client_recv_buffer, 0, BUFFER_SIZE);
	// memset(receive_decrypt_buffer, 0, BUFFER_SIZE);

	int recv_len = recv(clientSocket, (void*)client_recv_buffer, BUFFER_SIZE, 0);
	cout << "Client_recv_thread: The ciphertext that received from server: " << client_recv_buffer << endl;
	if (recv_len < 0)
	{
		cout << "Failed to receive message from server." << endl;
		// break;
		return;
	}
	if (recv_len == 0)
	{
		cout << "Server closed the connection." << endl;
		exit(0);
		// break;
		return;
	}

	// char *receive_decrypt_buffer= DecryptMsg(receive_buffer, bkey);

	// ��ʾ���������͵���Ϣ
	// cout << "Received message from server: plaintext: " << receive_decrypt_buffer << endl;

	// ��ʾ���������͵���Ϣ
	DecryptMsg(client_recv_buffer, bkey);
	cout << "Client_recv_thread: plaintext that the client received from server:";
	cout << client_recv_buffer << endl;
	// }
	// pthread_exit(NULL);
}

void Client_send_thread(int clientSocket)
{
	char client_send_buffer[BUFFER_SIZE]{};

	// while (!exit_flag)
	// {
		// ������Ϣ��������
	memset(client_send_buffer, 0, BUFFER_SIZE);
	// cout << "Please input the message that the client will transmit: " << endl;
	// cin.getline(client_send_buffer, BUFFER_SIZE - 1);

	int count = 0;
	int sum = 0;
	while ((count = read(STDIN_FILENO, client_send_buffer, BUFFER_SIZE)) > 0)
	{
		sum += count;
	}
	if (count == -1 && errno != EAGAIN)
	{
		cout << "Send_send_thread: Get_message wrong!" << endl;
		return; //������Ϣ���ͻ��˶�ȡ����
	}

	if (strcmp(client_send_buffer, "exit") == 0)
	{
		exit_flag = true;
		cout << "Client_send_thread: exit" << endl;
		// break;
		return;
	}

	EncryptMsg(client_send_buffer, bkey);
	cout << "Client_send_thread: ciphertext that the client will transmit: " << client_send_buffer << endl;

	if (send(clientSocket, (void*)client_send_buffer, BUFFER_SIZE, 0) < 0)
	{
		cout << "Error: failed to send message to server." << endl;
		// break;
		return;
	}
	// }
	// pthread_exit(NULL);
}

// ��ԿЭ�̣�
// ��1���������DES��Կ
// ��2��ʹ�ù�Կ����DES��Կ�����͸������
/*void DES_Key_Aggrement(int clientSocket, char* str)
{
	string des_key;
	string str_rsa_keypair;
	char server_rsa_keypair[BUFFER_SIZE]{};
	memset(server_rsa_keypair, 0, BUFFER_SIZE);

	// �������DES��Կ
	srand((unsigned) time(NULL));
	for (int i = 0; i < 8; i++)
	{
		char temp = 65 + rand() % 26;
		des_key += temp;
	}
	cout << "Client: The plaintext of the DES key: " << des_key << endl;

	// �ַ���תchar
	for (int i = 0; i < des_key.size(); i++)
	{
		str[i] = des_key[i];
	}
	str[des_key.size()] = '\0';

	// ���շ������Ĺ�Կ��
	int len_n = recv(clientSocket, (void*)server_rsa_keypair, sizeof(server_rsa_keypair), 0);
	for (int i = 0; i < sizeof(server_rsa_keypair); i++)
	{
		str_rsa_keypair[i] = server_rsa_keypair[i];// string�����ַ���
	}
	cout << "Client: RSA_keypair from the server: " << server_rsa_keypair << endl;
	cout << "Client: RSA_keypair from the server: " << str_rsa_keypair << endl;


	int pos = str_rsa_keypair.find(",", 0);
	unsigned long long rsa_public_key_e = str2uint(str_rsa_keypair.substr(0, pos));
	unsigned long long n = str2uint(str_rsa_keypair.substr(pos + 1, str_rsa_keypair.size()));
	cout << "Client: RSA_keypair_publickey_e is : " << rsa_public_key_e << endl;
	cout << "Client:RSA_keypair_publickey_n is:" << n << endl;

	// ʹ��RSA�Ĺ�Կ����DES��Կ
	string encrypted_DesKey = client_encry(des_key, rsa_public_key_e, n);
	cout << "Client: The ciphertext of the DES key: " << encrypted_DesKey << endl;

	char encrypted_DESKey_[8192];
	for (int i = 0; i < encrypted_DesKey.size(); i++)
	{
		encrypted_DESKey_[i] = encrypted_DesKey[i];
	}
	encrypted_DESKey_[encrypted_DesKey.size()] = '\0';

	// �����������ͼ��ܹ���DES��Կ
	if (send(clientSocket, (void *)encrypted_DESKey_, sizeof(encrypted_DESKey_), 0) < 0)
	{
		cout << "Client: Fail to send the ciphertext of DES key successfully! " << endl;
		return;
	}
}*/


int main()
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

	int epoll_fd = epoll_create(256);//����һ��ר�õ�epoll�ļ�������
	if (epoll_fd < 0)
	{
		cout << "Client: Client epoll_create failed ! Error: " << strerror(errno) << endl;
		return -1;
	}

	struct epoll_event ep_event, ep_input; // ��Լ�����socket������2��epoll_event
	struct epoll_event ret_events[20]; // �������ڻش�Ҫ������¼�

	fcntl(clientSocket, F_SETFL, O_NONBLOCK); // ���÷�����
	// ep_event����ע���¼�
	ep_event.data.fd = clientSocket;
	ep_event.events = EPOLLIN | EPOLLET;
	//���ڿ���ĳ���ļ��������ϵ��¼���ע�ᣬ�޸ģ�ɾ����
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, clientSocket, &ep_event) < 0)
	{
		cout << "Client: Epoll_ctl ep_event failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	// ��clientSocket�󶨼�����׼������ļ�������
	fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
	ep_input.data.fd = STDIN_FILENO;
	ep_input.events = EPOLLIN | EPOLLET;
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, STDIN_FILENO, &ep_input) < 0)
	{
		cout << "Client: Epoll_ctl ep_input failed . Error: " << strerror(errno) << endl;
		return -1;
	}
	int ret_num = 20;

	cout << "############### DES_KEY_GEN ###############" << endl;
	// ���des��Կ����
	string des_key = DES_key_gen();
	cout << "Client: The plaintext of the DES key: " << des_key << endl;
	// string תchar
	char des_key_[des_key.length() + 1];
	strcpy(des_key_, des_key.data());
	// cout << "Client: The plaintext of the DES key(char): " << des_key_ << endl;
	BlockFromStr(bkey, des_key_);// ��ȡ��Կ

	char server_rsa_keypair[BUFFER_SIZE]{};
	memset(server_rsa_keypair, 0, BUFFER_SIZE);
	string str_rsa_keypair;

	cin.ignore(2048, '\n');
	while (1)
	{
		//������ѰI/O�¼��ķ���
		switch (int event_num = epoll_wait(epoll_fd, ret_events, ret_num, epoll_timeout))
		{
			// ����ֵ����0��ʾ��ʱ
		case 0:
			cout << "Client: Epoll_wait: Time out!" << endl;
			break;
			// ����ֵС��0��ʾ����
		case -1:
			cout << "Client: Eppoll_wait: Wrong!" << endl;
			break;
			// ����ֵ����0��ʾ�¼��ĸ���
		default:
		{
			for (int i = 0; i < event_num; ++i)
			{
				// ���յ����ݣ���socket
				if (ret_events[i].events == EPOLLIN)
				{
					// ��׼����
					if (ret_events[i].data.fd == STDIN_FILENO)
					{
						Client_send_thread(clientSocket); // ������������Ϣ
					}
					else
					{
						if (getrsakey == false)
						{
							// ���շ������Ĺ�Կ��
							int len_n = recv(clientSocket, (void*)server_rsa_keypair, BUFFER_SIZE, 0);
							cout << "Client: RSA_keypair from the server: " << server_rsa_keypair << endl;
							// charתstring
							str_rsa_keypair = server_rsa_keypair;

							// cout << "Client: RSA_keypair from the server: " << str_rsa_keypair << endl;

							int pos = str_rsa_keypair.find(",", 0);
							unsigned long long rsa_public_key_e = str2uint(str_rsa_keypair.substr(0, pos));
							unsigned long long n = str2uint(str_rsa_keypair.substr(pos + 1, str_rsa_keypair.size()));
							// cout << "Client: RSA_keypair_publickey_e is : " << rsa_public_key_e << endl;
							// cout << "Client: RSA_keypair_publickey_n is:" << n << endl;
							getrsakey = true;

							// ʹ��RSA�Ĺ�Կ����DES��Կ
							string encrypted_DesKey = client_encry(des_key, rsa_public_key_e, n);
							cout << "Client: The ciphertext of the DES key: " << encrypted_DesKey << endl;

							char encrypted_DESKey_[8192];
							for (int i = 0; i < encrypted_DesKey.size(); i++)
							{
								encrypted_DESKey_[i] = encrypted_DesKey[i];
							}
							encrypted_DESKey_[encrypted_DesKey.size()] = '\0';

							// �����������ͼ��ܹ���DES��Կ
							if (send(clientSocket, (void*)encrypted_DESKey_, sizeof(encrypted_DESKey_), 0) < 0)
							{
								cout << "Client: Fail to send the ciphertext of DES key successfully! " << endl;
							}
							else
							{
								cout << "Client: Send the ciphertext of DESkey successfully!!" << endl;
							}

							cout << "################# Let's chat ###################" << endl;
						}

						else
						{
							Client_recv_thread(clientSocket); // ���շ�������Ϣ 
						}
					}
				}
			}
		}

		}
	}
	/*
		// ���շ������Ĺ�Կ��
		int len_n = recv(clientSocket, (void*)server_rsa_keypair, BUFFER_SIZE, 0);
		cout << "Client: RSA_keypair from the server: " << server_rsa_keypair << endl;
		// charתstring
		str_rsa_keypair = server_rsa_keypair;

		// cout << "Client: RSA_keypair from the server: " << str_rsa_keypair << endl;

		int pos = str_rsa_keypair.find(",", 0);
		unsigned long long rsa_public_key_e = str2uint(str_rsa_keypair.substr(0, pos));
		unsigned long long n = str2uint(str_rsa_keypair.substr(pos + 1, str_rsa_keypair.size()));
		// cout << "Client: RSA_keypair_publickey_e is : " << rsa_public_key_e << endl;
		// cout << "Client: RSA_keypair_publickey_n is:" << n << endl;

		// ʹ��RSA�Ĺ�Կ����DES��Կ
		string encrypted_DesKey = client_encry(des_key, rsa_public_key_e, n);
		cout << "Client: The ciphertext of the DES key: " << encrypted_DesKey << endl;

		char encrypted_DESKey_[8192];
		for (int i = 0; i < encrypted_DesKey.size(); i++)
		{
			encrypted_DESKey_[i] = encrypted_DesKey[i];
		}
		encrypted_DESKey_[encrypted_DesKey.size()] = '\0';

		// �����������ͼ��ܹ���DES��Կ
		if (send(clientSocket, (void*)encrypted_DESKey_, sizeof(encrypted_DESKey_), 0) < 0)
		{
			cout << "Client: Fail to send the ciphertext of DES key successfully! " << endl;
		}
		else
		{
			cout << "Client: Send the ciphertext of DESkey successfully!!" << endl;
		}*/

		// char des_key_[8192];
		// DES_Key_Aggrement(clientSocket, des_key_);
		// cout << "des_key_: " << des_key_ << endl;

		// cout << "############ message ############" << endl;
		// BlockFromStr(bkey, des_key_);// ��ȡ��Կ
		// fd_set read_fds; // ��ȡ�ļ�����������
		/*while (!exit_flag)
		{
			FD_ZERO(&read_fds); // ����ļ�����������
			FD_SET(STDIN_FILENO, &read_fds); // ��ӱ�׼���뵽����
			FD_SET(clientSocket, &read_fds); // ��ӿͻ����׽��ֵ�����

			// ʹ�� select �����ļ�������
			int max_fd = max(clientSocket, STDIN_FILENO) + 1;
			int activity = select(max_fd, &read_fds, NULL, NULL, NULL);

			if (activity < 0)
			{
				cout << "Error in select." << endl;
				break;
			}

			pthread_t cli_recv, cli_send;

			// ��׼���������ݿɶ�
			if (FD_ISSET(STDIN_FILENO, &read_fds))
			{
				pthread_create(&cli_send, NULL, Client_send_thread, NULL);

				pthread_join(cli_send, NULL);

			}

			// �ͻ����׽��������ݿɶ�
			if (FD_ISSET(clientSocket, &read_fds))
			{
				pthread_create(&cli_recv, NULL, Client_recv_thread, NULL);

				pthread_join(cli_recv, NULL);
			}

		}*/

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

		//�����˳���ȷ���ͻ����ܹ������˳����ͷ���Դ��

	close(clientSocket); // �ر����Ӻ�socket 
	//closeSocket(clientSocket);// �رտͻ����׽���

	// cleanupWinSock();// ����WinSock��

	return 0;
}


