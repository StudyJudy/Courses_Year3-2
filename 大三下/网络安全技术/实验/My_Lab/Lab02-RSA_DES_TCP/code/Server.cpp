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
	int client_Socket;//客户端套接字  
	// char clientname[2048];//客户名
	// char time_stamp[64];//当前时间的字符串
	// char content_buf[256];//客户端收到信息后服务器端显示及发送的字符串
	// int flag;//当前客户端标号
	sockaddr_in* client_addr;
	string des_key = "";
	RSA_key_pair rsa_keypair;
};

CLIENT* client;

//vector<int> clientSockets;// 存储连接的客户端套接字
// CLIENT clients[20];//聊天室最多20人
//mutex mtx; // 互斥锁，用于保护 clientSockets 向量，确保多线程访问时的同步。
//vector<CLIENT> clients;

// int num = 0;//当前客户端的标记

int client_Socket;
bool exit_flag = false;
Block bkey; // 密钥
bool getdeskey = false;

void Server_recv_thread(int client_Socket)
{
	/*
	CLIENT* Client = new CLIENT; // 声明并初始化一个 CLIENT 指针
	char timestamp[8196] = {};  // 时间戳，可以是字符串或时间格式
	char content[BUFFER_SIZE];   // 接收的聊天消息内容
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

		//接收数据
		// int recv_time = recv(Client->client_Socket, timestamp, sizeof(timestamp), 0);
		// char* TimeStamp = dealMsg(timestamp, bkey);
		int recv_con = recv(Client->client_Socket, content, sizeof(content), 0);
		char* Content = dealMsg(content, bkey);
		//int recv_time = receiveMessage(Client->client, timestamp);
		//int ret_con = receiveMessage(Client->client, content);

		//拼接服务器端收到消息的显示格式
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
			//服务器端的显示
			cout << Client->content_buf << endl;

			//lock_guard<mutex> lock(mutex);//使用互斥锁以确保多个线程安全地操作 clients vector,保护共享资源 clients
			if (strcmp(Content, "exit_request") != 0)
			{
				for (i = 0; i < num; i++)
				{
					if (i != Client->flag)
					{
						// 发送回复消息给客户端
						encryptMsg(Client->content_buf, bkey);
						int send_result = send(clients[i].client_Socket, Client->content_buf, strlen(Client->content_buf), 0);
						//sendMessage(Client->client, Client->buf);
						if (send_result == -1)// 处理发送错误
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
						// 发送回复消息给客户端
						encryptMsg(Client->content_buf, bkey);
						int send_result = send(clients[i].client_Socket, Client->content_buf, strlen(Client->content_buf), 0);
						//sendMessage(Client->client, Client->buf);
						if (send_result == -1)// 处理发送错误
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
						if (send_result == -1)// 处理发送错误
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
		// 接收客户端发送的消息
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
	// 显示客户端发送的消息
	// cout << "Received message from client: plaintext: " << recv_decrypt_buffer << endl;

	// 显示客户端发送的消息
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
		// 发送消息给客户端
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
		return; //发送信息给客户端读取错误
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

	int epoll_fd = epoll_create(256);//生成一个专用的epoll文件描述符
	if (epoll_fd < 0)
	{
		cout << "Server: Server epoll_create failed ! Error: " << strerror(errno) << endl;
		return -1;
	}

	struct epoll_event ep_event, ep_input; // 针对监听的fd_skt，创建2个epollevent
	struct epoll_event events[max_line]; // 数组用于回传要处理的事件

	fcntl(serverSocket, F_SETFL, O_NONBLOCK); // 设置非阻塞
	// ep_event用于注册事件
	ep_event.data.fd = serverSocket;
	ep_event.events = EPOLLIN | EPOLLET;
	//用于控制某个文件描述符上的事件（注册，修改，删除）
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serverSocket, &ep_event) < 0)
	{
		cout << "Server: Epoll_ctl failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	// 给serverSocket绑定监听标准输入的文件描述符
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
	//初始化WinSock库
		 /*if (initializeWinSock() != 0)
		 {
			 cerr << "Server: WSAStartup failed. Error: " << WSAGetLastError() << endl;
			 return 1;
		 }*/

		 //创建一个Socket来监听客户端的连接请求。
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

	int epoll_fd = epoll_create(256);//生成一个专用的epoll文件描述符
	if (epoll_fd < 0)
	{
		cout << "Server: Server epoll_create failed ! Error: " << strerror(errno) << endl;
		return -1;
	}

	struct epoll_event ep_event, ep_input; // 针对监听的fd_skt，创建2个epoll_event
	struct epoll_event ret_events[20]; // 数组用于回传要处理的事件

	fcntl(serverSocket, F_SETFL, O_NONBLOCK); // 设置非阻塞
	// ep_event用于注册事件
	ep_event.data.fd = serverSocket;
	ep_event.events = EPOLLIN | EPOLLET;
	//用于控制某个文件描述符上的事件（注册，修改，删除）
	if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, serverSocket, &ep_event) < 0)
	{
		cout << "Server: Epoll_ctl ep_event failed . Error: " << strerror(errno) << endl;
		return -1;
	}

	// 给serverSocket绑定监听标准输入的文件描述符
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
		//用于轮寻I/O事件的发生
		switch (int event_num = epoll_wait(epoll_fd, ret_events, ret_num, epoll_timeout))
		{
			// 返回值等于0表示超时
		case 0:
			cout << "Server: Epoll_wait: Time out!" << endl;
			break;
			// 返回值小于0表示出错
		case -1:
			cout << "Server: Eppoll_wait: Wrong!" << endl;
			break;
			// 返回值大于0表示事件的个数
		default:
		{
			for (int i = 0; i < event_num; ++i)
			{
				// 有来自客户端新的连接
				if (ret_events[i].data.fd == serverSocket)
				{
					//等待客户端的连接

					sockaddr_in* clientAddr = new sockaddr_in();
					socklen_t clientAddrLen = sizeof(clientAddr);
					int client_Socket = acceptClient(serverSocket, (sockaddr_in*)clientAddr, &clientAddrLen);

					// 生成RSA公钥私钥对
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

					// 将string类型的公钥转为char类型
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

					// 将RSA公钥对发送给客户端
					if (send(client_Socket, (void*)rsa_keypair, sizeof(rsa_keypair), 0) < 0)
					{
						cout << "Server: Failed to send RSA_key_pair_public_key." << endl;
						return -1;
					}
					else
					{
						cout << "Server: Send RSA_key_pair_public_key successfully!!" << endl;
					}

					// epoll监听客户端发来的消息
					if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_Socket, &new_client_event) < 0)
					{
						cout << "Epoll_ctl error!" << endl;
						return -1;
					}

				}
				// 接收到数据，读socket
				else if (ret_events[i].events == EPOLLIN)
				{
					// 标准输入
					if (ret_events[i].data.fd == STDIN_FILENO)
					{
						Server_send_thread(client->client_Socket);
					}
					else
					{
						if (getdeskey == false)
						{
							// 进行正常通信
							CLIENT* client = (CLIENT*)ret_events[i].data.ptr;

							if (recv(client->client_Socket, (void*)server_des_cipherkey, BUFFER_SIZE, 0) < 0)
							{
								cout << "Server: Fail to receive the ciphertext of DES key from the client! " << endl;
							}
							else
							{
								cout << "Server : The ciphertext of DES key is: " << server_des_cipherkey << endl;
							}
							// char转string
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

							BlockFromStr(bkey, des_key_); // 获取密钥

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
	// 定义存储连接的客户端套接字集合++
	// fd_set readfds;
	// int max_sd = serverSocket;*/

	/*// 定义客户端套接字++
	// int clientSocket[MAX_CLIENTS];
	memset(clientSocket, 0, sizeof(clientSocket));

	//等待客户端的连接
	sockaddr_in clientAddr;
	socklen_t clientAddrLen = sizeof(clientAddr);

	client_Socket = acceptClient(serverSocket, &clientAddr, &clientAddrLen);*/


	// ++生成RSA公钥私钥对

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

	// 将string类型的公钥转为char类型
	for (int i = 0; i < key_pair.en.size(); i++)
	{
		rsa_keypair[i] = key_pair.en[i];
	}

	rsa_keypair[key_pair.en.size()] = '\0';
	cout << "Server: rsa_keypair: " << rsa_keypair << endl;*/

	// 将公钥对发送给客户端
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
	// char转string
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

	BlockFromStr(bkey, des_key_); // 获取密钥


	cout << "################# message ###################" << endl;

	pthread_t sev_recv, sev_send;
	pthread_create(&sev_recv, NULL, Server_recv_thread, NULL);
	pthread_create(&sev_send, NULL, Server_send_thread, NULL);


	/*while (true)
	{
		// 清空套接字集合
		FD_ZERO(&readfds);

		// 将服务器套接字添加到集合中
		FD_SET(serverSocket, &readfds);

		// 将客户端套接字添加到集合中
		for (int i = 0; i < MAX_CLIENTS; ++i)
		{
			int sd = clientSocket[i];

			if (sd > 0)
				FD_SET(sd, &readfds);

			if (sd > max_sd)
				max_sd = sd;
		}

		// 调用 select 函数以检查套接字状态
		int activity = select(max_sd + 1, &readfds, NULL, NULL, NULL);

		if ((activity < 0) && (errno != EINTR))
		{
			cout << "Server: select error." << endl;
		}

		// 如果服务器套接字有可读事件，表示有新的客户端连接
		if (FD_ISSET(serverSocket, &readfds))
		{
			int newSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);
			if (newSocket < 0)
			{
				cout << "Server: Failed to accept new connection. Error: " << strerror(errno) << endl;
				continue;
			}

			// 将新的客户端套接字添加到数组中
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
		//接受客户端的连接
		clients[num].client_Socket = acceptClient(serverSocket, &clientAddr, &clientAddrLen);
		clients[num].flag = num;

		BlockFromStr(bkey, "aeskey"); // 获取密钥

		//接收用户名，检验是否能成功接收
		/*char clientname_temp[1024];
		memset(clientname_temp , 0, sizeof(clientname_temp));

		// recv(clients[num].client_Socket, clients[num].clientname, sizeof(clients[num].clientname), 0); //接收用户名
		int recv_name_length = recv(clients[num].client_Socket, (void*)clientname_temp, sizeof(clientname_temp), 0);

		cout << "Server: cipher_name_text: " << clientname_temp << endl;
		// clients[num].clientname = dealMsg(clients[num].clientname, bkey);
		strcpy(clients[num].clientname, dealMsg(clientname_temp, bkey));

		cout << "Server: plaintext: " << clients[num].clientname << " connected" << endl;
		//receiveMessage(clients[num].client, clients[num].clientname);

		//clients.push_back();
		// DWORD  lpThreadID;
		// HANDLE clientThread;

		//创建线程处理消息接收与转发
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
		// 标记加一
		// num++;
		//lock_guard<mutex> lock(mtx); // 锁定互斥锁，确保线程安全
		//clients.push_back(newClient);
	// }

	// 关闭连接和socket
	close(client_Socket);
	close(serverSocket);

	// cleanupWinSock(); // 清理WinSock库

	return 0;
}