'''
本机数据处理 + 数据接收发送
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from distribute_monitor.logger import LOGGER
import re

fs_path_config = {
    'rtc': '/proc/driver/rtc',
    'buddyinfo': '/proc/buddyinfo',
    'cpuinfo': '/proc/cpuinfo',
    'uptime': '/proc/uptime',
    'meminfo': '/proc/meminfo'
}

class LocalDataProcess:
    REGEX_PAT = '[\t ]*:[\t ]*'
    MULWHITE_PAT = '[\t ]+'
    def __init__(self):
        self.rtc_file = open(fs_path_config['rtc'], 'r')
        self.buddyinfo_file = open(fs_path_config['buddyinfo'], 'r')
        self.cpuinfo_file = open(fs_path_config['cpuinfo'], 'r')
        self.uptime_file = open(fs_path_config['uptime'], 'r')
        self.meminfo_file = open(fs_path_config['meminfo'], 'r')
        self.processor_num = os.cpu_count()
        self.data_map = {}
        self.attribute_map = {
            'rtc': {},
            'buddyinfo': {},
            'cpuinfo': {},
            'uptime': {},
            'meminfo': {}
        }
        self.buddyinfo_cnt = None
        self.__get_attribute_map()
    
    def __get_rtc_data(self):
        self.rtc_file.seek(0)
        rtc_data = self.rtc_file.read()
        self.rtc_data = rtc_data.split('\n')
        return self.rtc_data
    
    def __get_buddyinfo_data(self):
        self.buddyinfo_file.seek(0)
        buddyinfo_data = self.buddyinfo_file.read()
        self.buddyinfo_data = buddyinfo_data.split('\n')
        return self.buddyinfo_data
    
    def __get_cpuinfo_data(self):
        self.cpuinfo_file.seek(0)
        cpuinfo_data = self.cpuinfo_file.read()
        self.cpuinfo_data = cpuinfo_data.split('\n\n')[0]
        return self.cpuinfo_data
    
    def __get_uptime_data(self):
        self.uptime_file.seek(0)
        uptime_data = self.uptime_file.read()
        self.uptime_data = uptime_data.split(' ')
        assert len(self.uptime_data) == 2, 'uptime_data format error'
        return self.uptime_data

    def __get_meminfo_data(self):
        self.meminfo_file.seek(0)
        meminfo_data = self.meminfo_file.read()
        self.meminfo_data = meminfo_data.split('\n')
        return self.meminfo_data
    
    def __get_attribute_map(self):
        if self.data_map.get('rtc', None) == None:
            self.data_map['rtc'] = self.__get_rtc_data()
        if self.data_map.get('buddyinfo', None) == None:
            self.data_map['buddyinfo'] = self.__get_buddyinfo_data()
        if self.data_map.get('cpuinfo', None) == None:
            self.data_map['cpuinfo'] = self.__get_cpuinfo_data()
        if self.data_map.get('uptime', None) == None:
            self.data_map['uptime'] = self.__get_uptime_data()
        if self.data_map.get('meminfo', None) == None:
            self.data_map['meminfo'] = self.__get_meminfo_data()

        for name, datas in self.data_map.items():
            if name == 'uptime':
                self.attribute_map[name] = {}
                self.attribute_map[name]['uptime'] = float(datas[0])
                self.attribute_map[name]['idle'] = float(datas[1])
                continue
            elif name == 'rtc':
                self.attribute_map[name] = {}
                for data in datas:
                    if data == '':
                        continue
                    deal_data = re.sub(LocalDataProcess.REGEX_PAT, ':', data)
                    attribute = deal_data.split(':')
                    if len(attribute) == 2:
                        self.attribute_map[name][attribute[0]] = attribute[1]
                continue
            elif name == 'cpuinfo':
                self.attribute_map[name] = {}
                processor_datas = datas.split('\n')
                for data in processor_datas:
                    if data == '':
                        continue
                    deal_data = re.sub(LocalDataProcess.REGEX_PAT, ':', data)
                    attribute = deal_data.split(':')
                    if len(attribute) == 2:
                        self.attribute_map[name][attribute[0]] = attribute[1]
                continue
            elif name == 'buddyinfo':
                self.attribute_map[name] = {}
                for data in datas:
                    if data == '':
                        continue
                    split_data_format = data.split(',')
                    assert len(split_data_format) == 2, 'buddyinfo format error'
                    node_name = split_data_format[0]
                    zone = [ ele for ele in re.sub(LocalDataProcess.MULWHITE_PAT, ' ', split_data_format[1]).split(' ') if ele != '' ]
                    self.attribute_map[name][(node_name, zone[1])] = zone[2:]
                continue
            elif name == 'meminfo':
                self.attribute_map[name] = {}
                for data in datas:
                    if data == '':
                        continue
                    deal_data = re.sub(LocalDataProcess.REGEX_PAT, ':', data)
                    attribute = deal_data.split(':')
                    if len(attribute) == 2:
                        self.attribute_map[name][attribute[0]] = attribute[1]
                continue
            else:
                LOGGER.error('localhost', '[failure] unknown data name')
                continue
        return self.attribute_map

    #
    # cpu information
    #

    def get_cpu_count(self):
        return self.processor_num
    
    def get_vendor_id(self):
        return self.attribute_map['cpuinfo']['vendor_id']
    
    def get_model_name(self):
        return self.attribute_map['cpuinfo']['model name']
    
    def get_cpu_mhz(self):
        return self.attribute_map['cpuinfo']['cpu MHz']
    
    def get_cache_size(self):
        return self.attribute_map['cpuinfo']['cache size']

    def get_cache_alignment(self):
        return self.attribute_map['cpuinfo']['cache_alignment']
    
    #
    # momory information
    #
    
    def get_mem_total(self):
        return self.attribute_map['meminfo']['MemTotal']
    
    def get_mem_free(self):
        return self.attribute_map['meminfo']['MemFree']
    
    def get_mem_available(self):
        return self.attribute_map['meminfo']['MemAvailable']
    
    def get_buddy_system_info(self):
        if self.buddyinfo_cnt != None:
            return self.buddyinfo_cnt
        self.buddyinfo_cnt = {}
        page_cnt = [2**e for e in range(0,12)]
        for page_cnt_key in page_cnt:
            self.buddyinfo_cnt[page_cnt_key] = 0
        for node_key, node_pages in self.attribute_map['buddyinfo'].items():
            for idx, node_page in enumerate(node_pages):
                self.buddyinfo_cnt[page_cnt[idx]] += int(node_page)
        return self.buddyinfo_cnt

    #
    # use info
    # 
    def get_uptime(self):
        return self.attribute_map['uptime']['uptime']
    
    def get_idle(self):
        return self.attribute_map['uptime']['idle']

    def get_from_start_use_rate(self):
        return self.get_idle() / (self.get_uptime() * self.processor_num)

    def get_batt_status(self):
        return self.attribute_map['rtc']['batt_status']

    def __del__(self):
        self.rtc_file.close()
        self.buddyinfo_file.close()
        self.cpuinfo_file.close()
        self.uptime_file.close()
        self.meminfo_file.close()
    

    
if __name__ == '__main__':
    local_data_process = LocalDataProcess()
    print('cpu count: ',local_data_process.get_cpu_count())
    print('vendor id: ',local_data_process.get_vendor_id())
    print('model name: ',local_data_process.get_model_name())
    print('cpu mhz: ',local_data_process.get_cpu_mhz())
    print('cache size: ',local_data_process.get_cache_size())
    print('cache alignment: ',local_data_process.get_cache_alignment())
    print('mem total: ',local_data_process.get_mem_total())
    print('mem free: ',local_data_process.get_mem_free())
    print('mem available: ',local_data_process.get_mem_available())
    print('buddy system info: ',local_data_process.get_buddy_system_info())
    print('uptime: ',local_data_process.get_uptime())
    print('idle: ',local_data_process.get_idle())
    print('from start use rate: ',local_data_process.get_from_start_use_rate())
    print('batt status: ',local_data_process.get_batt_status())
