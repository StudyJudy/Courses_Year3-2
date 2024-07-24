import socket
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))

def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    
    return ip

class UDPSender:
    loop_ip = '127.0.0.1'
    def __init__(self, ip:str, port:int) -> None:
        self.is_local = False
        self.ip = get_host_ip()
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        if ip == UDPSender.loop_ip:
            self.is_local = True
            self.socket.bind((UDPSender.loop_ip, int(port)))
        else:
            self.socket.bind((self.port, port))
            
    def send(self, data, ip, port):
        if self.is_local:
            if ip != UDPSender.loop_ip:
                return
        ret = self.socket.sendto(data, (ip, int(port)))
        return ret
    
    def receive(self):
        while(True):
            try:
                data, from_addr = self.socket.recvfrom(1024)
            except BlockingIOError:
                continue
            else:
                break
        return data, from_addr
    
    def set_timeout(self, timeout):
        self.socket.settimeout(timeout)
    
    def __del__(self):
        if self.socket is not None:
            self.socket.close()

if __name__ == '__main__':
    sender = UDPSender(UDPSender.loop_ip, 5820)
    receiver = UDPSender(UDPSender.loop_ip, 5821)
    sender.send('hello'.encode(), UDPSender.loop_ip, 5821)
    data, from_addr = receiver.receive()
    print(data.decode())

        
    
        