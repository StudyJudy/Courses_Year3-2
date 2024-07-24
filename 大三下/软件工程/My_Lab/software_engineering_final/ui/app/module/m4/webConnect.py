import paramiko
import re
import time
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

host = "192.168.123.129"
port = 22
username = "judy"
password = "123456"

import re


def parse_info(info_str):
    # Initialize the dictionary to store the parsed data
    info_dict = {}

    # Extract key-value pairs
    patterns = [
        (r'cpu count:\s+(\d+)', 'cpu_count', int),
        (r'vendor id:\s+(.+)', 'vendor_id', str),
        (r'model name:\s+(.+)', 'model_name', str),
        (r'cpu mhz:\s+([\d.]+)', 'cpu_mhz', float),
        (r'cache size:\s+(\d+)\s+KB', 'cache_size_kb', int),
        (r'cache alignment:\s+(\d+)', 'cache_alignment', int),
        (r'mem total:\s+(\d+)\s+kB', 'mem_total_kb', int),
        (r'mem free:\s+(\d+)\s+kB', 'mem_free_kb', int),
        (r'mem available:\s+(\d+)\s+kB', 'mem_available_kb', int),
        (r'uptime:\s+([\d.]+)', 'uptime', float),
        (r'idle:\s+([\d.]+)', 'idle', float),
        (r'from start use rate:\s+([\d.]+)', 'from_start_use_rate', float),
        (r'batt status:\s+(.+)', 'batt_status', str)
    ]

    for pattern, key, cast in patterns:
        match = re.search(pattern, info_str)
        if match:
            info_dict[key] = cast(match.group(1))

    # Extract buddy system info
    buddy_system_match = re.search(r'buddy system info:\s+\{(.+?)\}', info_str)
    if buddy_system_match:
        buddy_system_str = buddy_system_match.group(1)
        buddy_system_pairs = buddy_system_str.split(',')
        buddy_system_dict = {}
        for pair in buddy_system_pairs:
            k, v = pair.split(':')
            buddy_system_dict[int(k.strip())] = int(v.strip())
        # sorted_buddy_system_info = dict(sorted(buddy_system_dict.items(), key=lambda item: item[0], reverse=True))
        info_dict['buddy_system_info'] = buddy_system_dict

    # Extract cluster status
    cluster_status_match = re.search(r'\[(.+?)\]', info_str)
    if cluster_status_match:
        cluster_status_str = cluster_status_match.group(1)
        cluster_status_list = eval(cluster_status_str)
        # Convert list to dictionary
        cluster_status_dict = {}
        for item in cluster_status_list:
            key, value = item
            cluster_status_dict[key] = value
        info_dict['cluster_status'] = cluster_status_dict

    return info_dict


def getInfo():
    try:
        ssh.connect(host, port, username, password)
    except Exception as e:
        print(f"Failed to connect to {host}:{port}, error: {e}")
        return

    try:
        stdin, stdout, stderr = ssh.exec_command('python3 ./桌面/m4/main.py')
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            output = stdout.read().decode()
            # print(output)
        else:
            error = stderr.read().decode()
            print(f"Error: {error}")
    except Exception as e:
        print(f"Failed to execute command, error: {e}")
    finally:
        ssh.close()
        # print("SSH connection closed")

    # print(parse_info(output))

    return parse_info(output)

if __name__ == '__main__':
    getInfo()
