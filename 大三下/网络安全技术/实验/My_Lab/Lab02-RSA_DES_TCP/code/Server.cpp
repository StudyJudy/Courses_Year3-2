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
#include <sys/epoll.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <limits>
#include <aio.h>
#include <sys/select.h>
#include <future>
// #include <WinSock2.h>
// #include <ws2tcpip.h>

#include "DES_handle.cpp"
#include "commoncode.h"
#include "RSA.cpp"
// #pragma comment(lib, "ws2_32.lib")
#define MAX_CLIENTS 20
const static int epoll_timeout = -1;

using namespace std;

struct CLIENT
{
	int client_Socket;//�ͻ����׽���  
	// char clientname[2048];//�ͻ���
	// char time_stamp[64];//��ǰʱ����ַ���
	// char content_buf[256];//�ͻ����յ���Ϣ�����������ʾ�����͵��ַ���
	// int flag;//��ǰ�ͻ��˱��
	sockaddr_in* client_addr;
	string des_key = "";
	RSA_key_pair rsa_keypair;
};

CLIENT* client;

//vector<int> clientSockets;// �洢���ӵĿͻ����׽���
// CLIENT clients[20];//���������20��
//mutex mtx; // �����������ڱ��� clientSockets ������ȷ�����̷߳���ʱ��ͬ����
//vector<CLIENT> clients;

// int num = 0;//��ǰ�ͻ��˵ı��

int client_Socket;
bool exit_flag = false;
Block bkey; // ��Կ
bool getdeskey = false;

void Server_recv_thread(int client_Socket)
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

	// while (!exit_flag)
	// {
		// ���տͻ��˷��͵���Ϣ
	memset(server_recv_buffer, 0, BUFFER_SIZE);

	int recv_len = recv(client_Socket, (void*)server_recv_buffer, BUFFER_SIZE, 0);

	cout << "Server_recv_thread: The ciphertext that received from client: " << server_recv_buffer << endl;

	if (recv_len < 0)
	{
		cout << "Failed to receive message from client." << endl;
		// break;
		return;
	}
	if (recv_len == 0)
	{
		cout << "Client closed the connection." << endl;
		exit(0);
		// break;
		return;
	}

	// char* recv_decrypt_buffer = DecryptMsg(Server_recv_buffer, bkey);
	// ��ʾ�ͻ��˷��͵���Ϣ
	// cout << "Received message from client: plaintext: " << recv_decrypt_buffer << endl;

	// ��ʾ�ͻ��˷��͵���Ϣ
	DecryptMsg(server_recv_buffer, bkey);
	cout << "Server_recv_thread: The plaintext that the server received from client: ";
	cout << server_recv_buffer << endl;
	// }
	// pthread_exit(NULL);
}

void Server_send_thread(int client_Socket)
{
	char server_send_buffer[BUFFER_SIZE];

	// while (!exit_flag)
	// {
		// ������Ϣ���ͻ���
	memset(server_send_buffer, 0, BUFFER_SIZE);
	// cout << "Please input the message that the server will transmit: " << endl;
	// cin.getline(server_send_buffer, BUFFER_SIZE - 1);
	int count = 0;
	int sum = 0;
	while ((count = read(STDIN_FILENO, server_send_buffer, BUFFER_SIZE)) > 0)
	{
		sum += count;
	}
	if (count == -1 && errno != EAGAIN)
	{
		cout << "Server_send_thread: Get_message wrong!" << endl;
		return; //������Ϣ���ͻ��˶�ȡ����
	}

	if (strcmp(server_send_buffer, "exit") == 0)
	{
		exit_flag = true;
		// break;
		return;
	}
	EncryptMsg(server_send_buffer, bkey);

	cout << "Server_send_thread: The ciphertext that the server will transmit: " << server_send_buffer << endl;

	if (send(client_Socket, (void*)server_send_buffer, BUFFER_SIZE, 0) < 0)
	{
		cout << "Error: failed to send message to client." << endl;
		// break;
		return;
	}
	// }
	// pthread_exit(NULL);
}

/*static int epoll_server(int serverSocket)
{

	int epoll_fd = epoll_create(256);//����һ��ר�õ�epoll�ļ�������
	if (epoll_fd < 0)
	{
		cout << "Server: Server epoll_create failed ! Error: " << strerror(errno) << endl;
		return -1;
	}

	struct epoll_event ep_event, ep_input; // ��Լ�����fd_skt������2��epollevent
	struct epoll_event events[max_line]; // �������ڻش�Ҫ������¼�

	fcntl(serverSocket, F_SETFL, O_NONBLOCK); // ���÷�����
	// ep_event����ע���¼�
	ep_event.data.fd = serverSocket;
	ep_event.events = EPOLLIN | EPOLLET;
	//���ڿ���ĳ���ļ��������ϵ��¼���ע�ᣬ�޸ģ�ɾ����
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serverSocket, &ep_event) < 0)
	{
		cout << "Server: Epoll_ctl failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	// ��serverSocket�󶨼�����׼������ļ�������
	fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
	ep_input.data.fd = STDIN_FILENO;
	ep_input.events = EPOLLIN | EPOLLET;
	if (epoll_ctl(fd_ep, EPOLL_CTL_ADD, STDIN_FILENO, &ep_input) < 0)
	{
		printf("server epoll_ctl-2 error!\n");
		return -1;
	}

}*/

RSA_key_pair key_pair;

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

	int epoll_fd = epoll_create(256);//����һ��ר�õ�epoll�ļ�������
	if (epoll_fd < 0)
	{
		cout << "Server: Server epoll_create failed ! Error: " << strerror(errno) << endl;
		return -1;
	}

	struct epoll_event ep_event, ep_input; // ��Լ�����fd_skt������2��epoll_event
	struct epoll_event ret_events[20]; // �������ڻش�Ҫ������¼�

	fcntl(serverSocket, F_SETFL, O_NONBLOCK); // ���÷�����
	// ep_event����ע���¼�
	ep_event.data.fd = serverSocket;
	ep_event.events = EPOLLIN | EPOLLET;
	//���ڿ���ĳ���ļ��������ϵ��¼���ע�ᣬ�޸ģ�ɾ����
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serverSocket, &ep_event) < 0)
	{
		cout << "Server: Epoll_ctl ep_event failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	// ��serverSocket�󶨼�����׼������ļ�������
	fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
	ep_input.data.fd = STDIN_FILENO;
	ep_input.events = EPOLLIN | EPOLLET;
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, STDIN_FILENO, &ep_input) < 0)
	{
		cout << "Server: Epoll_ctl ep_input failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	int ret_num = 20;
	char server_des_cipherkey[BUFFER_SIZE]{};
	memset(server_des_cipherkey, 0, sizeof(server_des_cipherkey));

	cin.ignore(2048, '\n');
	while (1)
	{
		//������ѰI/O�¼��ķ���
		switch (int event_num = epoll_wait(epoll_fd, ret_events, ret_num, epoll_timeout))
		{
			// ����ֵ����0��ʾ��ʱ
		case 0:
			cout << "Server: Epoll_wait: Time out!" << endl;
			break;
			// ����ֵС��0��ʾ����
		case -1:
			cout << "Server: Eppoll_wait: Wrong!" << endl;
			break;
			// ����ֵ����0��ʾ�¼��ĸ���
		default:
		{
			for (int i = 0; i < event_num; ++i)
			{
				// �����Կͻ����µ�����
				if (ret_events[i].data.fd == serverSocket)
				{
					//�ȴ��ͻ��˵�����

					sockaddr_in* clientAddr = new sockaddr_in();
					socklen_t clientAddrLen = sizeof(clientAddr);
					int client_Socket = acceptClient(serverSocket, (sockaddr_in*)clientAddr, &clientAddrLen);

					// ����RSA��Կ˽Կ��
					cout << "########## RSA_KEY_PAIR_GEN ############" << endl;
					// RSA_key_pair key_pair;
					/*while (1)
					{
						if (RSA_key_pair_gen(key_pair) == 0)
						{
							// cout << "True!" << endl;
							break;
						}
					}*/

					key_pair.private_key_d = 922628489;
					key_pair.n = 1915221097;
					key_pair.public_key_e = 41413;
					key_pair.en = "41413,1915221097";

					cout << "Server: RSA_publickey is : " << "{ e: " << key_pair.public_key_e << " , n: " << key_pair.n << "} " << endl;
					cout << "Server: RSA_privatekey is : " << "{ d: " << key_pair.private_key_d << " , n: " << key_pair.n << "} " << endl;

					char rsa_keypair[BUFFER_SIZE]{};
					memset(rsa_keypair, 0, BUFFER_SIZE);

					// ��string���͵Ĺ�ԿתΪchar����
					for (int i = 0; i < key_pair.en.size(); i++)
					{
						rsa_keypair[i] = key_pair.en[i];
					}

					rsa_keypair[key_pair.en.size()] = '\0';
					cout << "Server: rsa_keypair(char): " << rsa_keypair << endl;

					client = new CLIENT();
					client->client_Socket = client_Socket;
					client->client_addr = clientAddr;
					client->rsa_keypair = key_pair;

					fcntl(client_Socket, F_SETFL, O_NONBLOCK);
					struct epoll_event new_client_event;
					new_client_event.events = EPOLLIN | EPOLLET;
					new_client_event.data.ptr = client;

					// ��RSA��Կ�Է��͸��ͻ���
					if (send(client_Socket, (void*)rsa_keypair, sizeof(rsa_keypair), 0) < 0)
					{
						cout << "Server: Failed to send RSA_key_pair_public_key." << endl;
						return -1;
					}
					else
					{
						cout << "Server: Send RSA_key_pair_public_key successfully!!" << endl;
					}

					// epoll�����ͻ��˷�������Ϣ
					if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_Socket, &new_client_event) < 0)
					{
						cout << "Epoll_ctl error!" << endl;
						return -1;
					}

				}
				// ���յ����ݣ���socket
				else if (ret_events[i].events == EPOLLIN)
				{
					// ��׼����
					if (ret_events[i].data.fd == STDIN_FILENO)
					{
						Server_send_thread(client->client_Socket);
					}
					else
					{
						if (getdeskey == false)
						{
							// ��������ͨ��
							CLIENT* client = (CLIENT*)ret_events[i].data.ptr;

							if (recv(client->client_Socket, (void*)server_des_cipherkey, BUFFER_SIZE, 0) < 0)
							{
								cout << "Server: Fail to receive the ciphertext of DES key from the client! " << endl;
							}
							else
							{
								cout << "Server : The ciphertext of DES key is: " << server_des_cipherkey << endl;
							}
							// charתstring
							string str_des_cipherkey = server_des_cipherkey;
							// cout << "Server : The ciphertext of DES key is(string): " << server_des_cipherkey << endl;

							string des_key = "";
							des_key = server_decry(str_des_cipherkey, key_pair.private_key_d, key_pair.n);
							cout << "Server: The plaintext of DES key is: " << des_key << endl;

							getdeskey = true;

							char des_key_[8192];
							for (int i = 0; i < des_key.size(); i++)
							{
								des_key_[i] = des_key[i];
							}
							des_key_[des_key.size()] = '\0';

							BlockFromStr(bkey, des_key_); // ��ȡ��Կ

							char getdeskeysuccess[BUFFER_SIZE]{};
							memset(getdeskeysuccess, 0, BUFFER_SIZE);

							strcpy(getdeskeysuccess, "GetDESKeySuccess");
							EncryptMsg(getdeskeysuccess, bkey);
							send(client->client_Socket, (void*)getdeskeysuccess, sizeof(getdeskeysuccess), 0);

							cout << "################# Let's chat ###################" << endl;
						}
						else
						{
							Server_recv_thread(client->client_Socket);
						}

						// pthread_t sev_recv, sev_send;
						// pthread_create(&sev_recv, NULL, Server_recv_thread, NULL);
						// pthread_create(&sev_send, NULL, Server_send_thread, NULL);
					}
				}


			}

		}

		}
	}


	/*
	// ����洢���ӵĿͻ����׽��ּ���++
	// fd_set readfds;
	// int max_sd = serverSocket;*/

	/*// ����ͻ����׽���++
	// int clientSocket[MAX_CLIENTS];
	memset(clientSocket, 0, sizeof(clientSocket));

	//�ȴ��ͻ��˵�����
	sockaddr_in clientAddr;
	socklen_t clientAddrLen = sizeof(clientAddr);

	client_Socket = acceptClient(serverSocket, &clientAddr, &clientAddrLen);*/


	// ++����RSA��Կ˽Կ��

	// cout << "########## RSA_KEY_PAIR_GEN ############" << endl;
	// RSA_key_pair key_pair;
	/*while (1)
	{
		if (RSA_key_pair_gen(key_pair) == 0)
		{
			cout << "True!" << endl;
			break;
		}
	}*/

	/* key_pair.private_key_d = 922628489;
	key_pair.n = 1915221097;
	key_pair.public_key_e = 41413;
	key_pair.en = "41413,1915221097";

	cout << "Server: RSA_publickey is : " << "{ e: " << key_pair.public_key_e << " , n: " << key_pair.n << "} " << endl;
	cout << "Server: RSA_privatekey is : " << "{ d: " << key_pair.private_key_d << " , n: " << key_pair.n << "} " << endl;

	char rsa_keypair[BUFFER_SIZE]{};
	memset(rsa_keypair, 0, BUFFER_SIZE);

	// ��string���͵Ĺ�ԿתΪchar����
	for (int i = 0; i < key_pair.en.size(); i++)
	{
		rsa_keypair[i] = key_pair.en[i];
	}

	rsa_keypair[key_pair.en.size()] = '\0';
	cout << "Server: rsa_keypair: " << rsa_keypair << endl;*/

	// ����Կ�Է��͸��ͻ���
	/*if (send(client_Socket, (void*)rsa_keypair, sizeof(rsa_keypair), 0) < 0)
	{
		return -1;
	}
	else
	{
		cout << "Server: Send RSA_key_pair_public_key successfully!!" << endl;
	}*/

	/*char server_des_cipherkey[8192]{};
	memset(server_des_cipherkey, 0, sizeof(server_des_cipherkey));
	if (recv(client_Socket, (void*)server_des_cipherkey, sizeof(server_des_cipherkey), 0) < 0)
	{
		cout << "Server: Fail to receive the ciphertext of DES key from the client! " << endl;
	}
	else
	{
		cout << "Server : The ciphertext of DES key is: " << server_des_cipherkey << endl;
	}
	// charתstring
	string str_des_cipherkey = server_des_cipherkey;
	// cout << "Server : The ciphertext of DES key is(string): " << server_des_cipherkey << endl;

	string des_key = "";
	des_key = server_decry(str_des_cipherkey, key_pair.private_key_d, key_pair.n);
	cout << "Server: The plaintext of DES key is: " << des_key << endl;

	char des_key_[8192];
	for (int i = 0; i < des_key.size(); i++)
	{
		des_key_[i] = des_key[i];
	}
	des_key_[des_key.size()] = '\0';

	BlockFromStr(bkey, des_key_); // ��ȡ��Կ


	cout << "################# message ###################" << endl;

	pthread_t sev_recv, sev_send;
	pthread_create(&sev_recv, NULL, Server_recv_thread, NULL);
	pthread_create(&sev_send, NULL, Server_send_thread, NULL);


	/*while (true)
	{
		// ����׽��ּ���
		FD_ZERO(&readfds);

		// ���������׽�����ӵ�������
		FD_SET(serverSocket, &readfds);

		// ���ͻ����׽�����ӵ�������
		for (int i = 0; i < MAX_CLIENTS; ++i)
		{
			int sd = clientSocket[i];

			if (sd > 0)
				FD_SET(sd, &readfds);

			if (sd > max_sd)
				max_sd = sd;
		}

		// ���� select �����Լ���׽���״̬
		int activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);

		if ((activity < 0) && (errno != EINTR))
		{
			cout << "Server: select error." << endl;
		}

		// ����������׽����пɶ��¼�����ʾ���µĿͻ�������
		if (FD_ISSET(serverSocket, &readfds))
		{
			int newSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
			if (newSocket < 0)
			{
				cout << "Server: Failed to accept new connection. Error: " << strerror(errno) << endl;
				continue;
			}

			// ���µĿͻ����׽�����ӵ�������
			for (int i = 0; i < MAX_CLIENTS; ++i)
			{
				if (clientSocket[i] == 0)
				{
					clientSocket[i] = newSocket;
					cout << "Server: New connection, socket fd is " << newSocket << ", ip is " << inet_ntoa(clientAddr.sin_addr) << ", port is " << ntohs(clientAddr.sin_port) << endl;
					break;
				}
			}
		}

		pthread_t sev_recv, sev_send;
		pthread_create(&sev_recv, NULL, Server_recv_thread, NULL);
		pthread_create(&sev_send, NULL, Server_send_thread, NULL);

		pthread_join(sev_recv, NULL);
		pthread_join(sev_send, NULL);
	}*/

	// pthread_join(sev_recv, NULL);
	// pthread_join(sev_send, NULL);

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