import enum
from re import T
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath('.')))
from distribute_monitor.logger import Logger
from distribute_monitor.network import UDPSender,get_host_ip
from distribute_monitor.timer import Timer
from distribute_monitor.data_process import LocalDataProcess
import distribute_monitor
import random


class NodeStatus(enum.Enum):
    LEADER = 0,
    FOLLOWER = 1,
    CANDIDATE = 2,
    DEAD = 3,
    UNKNOWN = 4

    def __str__(self) -> str:
        return self.name

class Node:
    __heartbeat_interval = 1
    __election_interval = 5
    @staticmethod
    def get_election_interval():
        return Node.__election_interval + random.randint(0, Node.__election_interval)
    
    def get_heartbeat_interval():
        return Node.__heartbeat_interval


    def send_heartbeat(self):
        assert self.status == NodeStatus.LEADER, 'only leader can send heartbeat'
        self.election_timer.start()
        self.node_logger.debug(self.identity, 'send heartbeat')
        for node in distribute_monitor.Nodes:
            if node.identity != self.identity:
                self.udp_sender.send('heartbeat'.encode(), node.ip, node.port)

    def send_election(self, candidate = None):
        if self.status == NodeStatus.LEADER:
            candidate = None
        if candidate is None:
            if self.status == NodeStatus.FOLLOWER:
                self.status = NodeStatus.CANDIDATE
            candidate = (self.ip, self.port)
            self.elect = True
            self.election_cnt = 1
            self.node_logger.debug(self.identity, 'send election to all, election cnt = {}'.format(self.election_cnt))
            for node in distribute_monitor.Nodes:
                if node.identity != self.identity:
                    self.udp_sender.send('election'.encode(), node.ip, node.port)
        else:
            self.node_logger.debug(self.identity, 'receive election from {}'.format(candidate))
            if self.election_cnt >= 1 and not self.elect:
                return
            self.status = NodeStatus.FOLLOWER
            self.elect = True
            self.election_cnt = 0
            self.udp_sender.send('vote'.encode(), candidate[0], candidate[1])
    
    def receive(self):
        data, from_addr = self.udp_sender.receive()
        data = data.decode()
        # print('receive data: {}'.format(data))
        if data == 'heartbeat':
            self.node_logger.debug(self.identity, 'receive heartbeat from {}'.format(from_addr))
            self.leader = from_addr
            if self.status == NodeStatus.CANDIDATE:
                self.status = NodeStatus.FOLLOWER
            elif self.status == NodeStatus.LEADER:
                self.status = NodeStatus.FOLLOWER
                self.heartbeat_timer.stop()
            self.election_timer.reset()
        elif data == 'election':
            self.node_logger.debug(self.identity, 'receive election from {}'.format(from_addr))
            self.send_election(from_addr)
            self.election_timer.reset()
        elif data == 'vote':
            self.node_logger.debug(self.identity, 'receive vote from {}'.format(from_addr))
            self.election_cnt += 1
            self.election_timer.reset()
            if self.election_cnt >= len(distribute_monitor.Nodes) // 2 + 1:
                self.status = NodeStatus.LEADER
                self.heartbeat_timer.start()
                self.election_timer.stop()
                self.node_logger.debug(self.identity, 'become leader')
        else:
            self.node_logger.warning(self.identity, 'receive unknown message from {}'.format(from_addr))

    def get_status(self):
        return (self.identity, str(self.status))

        
    def __init__(self, ip:str, port:int) -> None:
        self.ip = ip
        self.port = port
        self.leader = None
        self.identity = ':'.join((self.ip, str(self.port)))
        self.udp_sender = UDPSender(ip, port)
        self.status = NodeStatus.FOLLOWER
        self.heartbeat_timer = Timer(Node.get_heartbeat_interval, self.send_heartbeat)
        self.election_timer = Timer(Node.get_election_interval, self.send_election)
        self.receive_timer = Timer(1, self.receive)
        self.election_cnt = 0
        self.elect = False
        self.term = 0
        self.node_logger = Logger(log_file=self.identity+'.log', mode='a')
        self.election_timer.start()
        self.receive_timer.start()



if __name__ == '__main__':
    import time
    while True:
        time.sleep(2)
        print(distribute_monitor.get_all_nodes_status())