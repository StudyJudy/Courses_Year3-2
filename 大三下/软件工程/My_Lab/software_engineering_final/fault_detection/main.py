import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import distribute_monitor
from distribute_monitor.data_process import LocalDataProcess

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

    print(distribute_monitor.get_all_nodes_status())