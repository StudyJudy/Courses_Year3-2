#include "TCPFINScan.h"

void* Thread_TCPFINHost(void* param)
{
	// 变量声明
	struct TCPFINHostThrParam* p;
	string HostIP, SrcIP, DstIP, LocalIP;
	unsigned HostPort, LocalPort, SrcPort, DstPort, LocalHostIP;
	struct sockaddr_in FINScanHostAddr, FromAddr, FinRevAddr;
	struct in_addr in_LocalhostIP;
	int FinSock, FinRevSock;
	int len, FromAddrLen;
	char sendbuf[8192];
	char recvbuf[8192];
	struct timeval TpStart, TpEnd;
	float TimeUse;
	//获得目标主机的IP地址和扫描端口号，以及本机的IP地址和端口
	p = (struct TCPFINHostThrParam*)param;
	HostIP = p->HostIP;
	HostPort = p->HostPort;
	LocalPort = p->LocalPort;
	LocalHostIP = p->LocalHostIP;
	//设置TCP SYN扫描的套接字地址
	memset(&FINScanHostAddr, 0, sizeof(FINScanHostAddr));
	FINScanHostAddr.sin_family = AF_INET;
	FINScanHostAddr.sin_addr.s_addr = inet_addr(&HostIP[0]);
	FINScanHostAddr.sin_port = htons(HostPort);
	//创建套接字
	FinSock = socket(PF_INET, SOCK_RAW, IPPROTO_TCP);
	if (FinSock < 0)
	{
		pthread_mutex_lock(&TCPFinPrintlocker);
		cout << "Can't creat raw socket !" << endl;
		pthread_mutex_unlock(&TCPFinPrintlocker);
	}
	FinRevSock = socket(PF_INET, SOCK_RAW, IPPROTO_TCP);
	if (FinRevSock < 0)
	{
		pthread_mutex_lock(&TCPFinPrintlocker);
		cout << "Can't creat raw socket !" << endl;
		pthread_mutex_unlock(&TCPFinPrintlocker);
	}
	//填充TCP FIN数据包
	struct pseudohdr* ptcph = (struct pseudohdr*)sendbuf;
	struct tcphdr* tcph = (struct tcphdr*)(sendbuf + sizeof(struct pseudohdr));
	//填充TCP伪头部，用于计算校验和
	ptcph->saddr = LocalHostIP;
	ptcph->daddr = inet_addr(&HostIP[0]);
	ptcph->useless = 0;
	ptcph->protocol = IPPROTO_TCP;
	ptcph->length = htons(sizeof(struct tcphdr));
	//填充TCP头
	tcph->th_sport = htons(LocalPort);
	tcph->th_dport = htons(HostPort);
	tcph->th_seq = htonl(123456);
	tcph->th_ack = 0;
	tcph->th_x2 = 0;
	tcph->th_off = 5;
	tcph->th_flags = TH_FIN; //TCP头flags字段的FIN位置1
	tcph->th_win = htons(65535);
	tcph->th_sum = 0;
	tcph->th_urp = 0;
	tcph->th_sum = in_cksum((unsigned short*)ptcph, 20 + 12);
	//发送TCP FIN数据包
	len = sendto(FinSock, tcph, 20, 0, (struct sockaddr*)&FINScanHostAddr, sizeof(FINScanHostAddr));
	if (len < 0)
	{
		pthread_mutex_lock(&TCPFinPrintlocker);
		cout << "Send TCP FIN Packet error !" << endl;
		pthread_mutex_unlock(&TCPFinPrintlocker);
	}
	// 将套接字设置为非阻塞模式
	if (fcntl(FinRevSock, F_SETFL, O_NONBLOCK) == -1)
	{
		pthread_mutex_lock(&TCPFinPrintlocker);
		cout << "Set socket in non-blocked model fail !" << endl;
		pthread_mutex_unlock(&TCPFinPrintlocker);
	}
	//接收 TCP 响应数据包循环
	FromAddrLen = sizeof(struct sockaddr_in);
	gettimeofday(&TpStart, NULL);//获得开始接收时刻
	do
	{
		//调用 recvfrom 函数接收数据包
		len = recvfrom(FinRevSock, recvbuf, sizeof(recvbuf), 0, (struct sockaddr*)&FromAddr, (socklen_t*)&FromAddrLen);
		if (len > 0)
		{
			SrcIP = inet_ntoa(FromAddr.sin_addr);
			//判断响应数据包的源地址是否等于目标主机地址，目的地址是否等于本机
			//IP地址，源端口是否等于被扫描端口，目的端口是否等于本机端口号
			if (SrcIP == HostIP)// 响应数据包的源地址等于目标主机地址
			{
				struct ip* iph = (struct ip*)recvbuf;
				int i = iph->ip_hl * 4;
				struct tcphdr* tcph = (struct tcphdr*)&recvbuf[i];
				SrcIP = inet_ntoa(iph->ip_src);       // TCP响应数据包的源IP地址
				DstIP = inet_ntoa(iph->ip_dst);       // TCP响应数据包的目的IP地址
				in_LocalhostIP.s_addr = LocalHostIP;
				// LocalIP = inet_ntoa(in_LocalhostIP);  // 本机IP地址
				string LocalIP = "192.168.175.132";  // 本机地址
				unsigned SrcPort = ntohs(tcph->th_sport);    // TCP响应包的源端口号
				unsigned DstPort = ntohs(tcph->th_dport);    // TCP响应包的目的端口号

				// cout << "HostIP: " << HostIP << endl;
				// cout << "SrcIP: " << SrcIP << endl;		
				// cout << "DstIP: " << DstIP << endl;
				// cout << "LocalIP: " << LocalIP << endl;

				// cout << "HostPort: " << HostPort << endl;
				// cout << "SrcPort: " << SrcPort << endl;
				// cout << "DstPort: " << DstPort << endl;
				// cout << "LocalPort: " << LocalPort << endl;
				if (HostIP == SrcIP && LocalIP == DstIP && SrcPort == LocalPort && DstPort == HostPort)
				// if (HostIP == SrcIP && LocalIP == DstIP && SrcPort == HostPort && DstPort == LocalPort)
				{
					if (tcph->th_flags == TH_FIN)
					{
						pthread_mutex_lock(&TCPFinPrintlocker);
						cout << "Host: " << SrcIP << " Port: " << ntohs(tcph->th_sport) - 1024 << " closed !" << endl;
						pthread_mutex_unlock(&TCPFinPrintlocker);
					}
					if (tcph->th_flags == 0x14) //判断是否为 RST 数据包
					{
						pthread_mutex_lock(&TCPFinPrintlocker);
						cout << "Host: " << SrcIP << " Port: " << ntohs(tcph->th_sport) << " open !" << endl;
						pthread_mutex_unlock(&TCPFinPrintlocker);
					}
				}
				break;
			}
		}
		// 判断等待响应数据包时间是否超过 3 秒
		gettimeofday(&TpEnd, NULL);
		TimeUse = (1000000 * (TpEnd.tv_sec - TpStart.tv_sec) + (TpEnd.tv_usec - TpStart.tv_usec)) / 1000000.0;
		if (TimeUse < 3)
			continue;
		else //超时，扫描端口开启
		{
			pthread_mutex_lock(&TCPFinPrintlocker);
			cout << "Host: " << HostIP << " Port: " << HostPort << " open !" << endl;
			pthread_mutex_unlock(&TCPFinPrintlocker);
			break;
		}
	} while (true);
	//退出子线程
	delete p;
	close(FinSock);
	close(FinRevSock);
	pthread_mutex_lock(&TCPFinScanlocker);
	// 子线程数减1
	TCPFinThrdNum--;
	pthread_mutex_unlock(&TCPFinScanlocker);
}
//===============================================================================================================================
void* Thread_TCPFinScan(void* param)
{
	// 变量定义
	struct TCPFINThrParam* p;
	string HostIP;
	unsigned BeginPort, EndPort, TempPort, LocalPort, LocalHostIP;
	pthread_t listenThreadID, subThreadID;
	pthread_attr_t attr, lattr;
	int ret;
	//获得目标主机的IP地址和扫描的起始端口号，终止端口号，以及本机的IP地址
	p = (struct TCPFINThrParam*)param;
	HostIP = p->HostIP;
	BeginPort = p->BeginPort;
	EndPort = p->EndPort;
	LocalHostIP = p->LocalHostIP;
	//循环遍历扫描端口
	TCPFinThrdNum = 0;
	LocalPort = 1024;

	for (TempPort = BeginPort; TempPort <= EndPort; TempPort++)
	{
		//设置子线程参数
		struct TCPFINHostThrParam* pTCPFINHostParam = new TCPFINHostThrParam;
		pTCPFINHostParam->HostIP = HostIP;
		pTCPFINHostParam->HostPort = TempPort;
		pTCPFINHostParam->LocalPort = TempPort + LocalPort;
		pTCPFINHostParam->LocalHostIP = LocalHostIP;
		//将子线程设置为分离状态
		pthread_attr_init(&attr);
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
		//创建子线程
		ret = pthread_create(&subThreadID, &attr, Thread_TCPFINHost, pTCPFINHostParam);
		if (ret == -1)
			cout << "Can't create the TCP FIN Scan Host thread !" << endl;
		pthread_attr_destroy(&attr);
		pthread_mutex_lock(&TCPFinScanlocker);
		// 子线程数加1
		TCPFinThrdNum++;
		pthread_mutex_unlock(&TCPFinScanlocker);
		// 子线程数大于100休眠
		while (TCPFinThrdNum > 100)
			sleep(3);
	}
	// 等待所有子线程返回
	while (TCPFinThrdNum != 0)
		sleep(1);
	cout << "TCP FIN scan thread exit !" << endl;
	// 返回主流程
	pthread_exit(NULL);
}

