#!/usr/bin/env python
# coding: utf-8

# In[2]:


def has_cycle(graph, v, visited, stack):
    visited[v] = True
    stack[v] = True
    
    for neighbor in graph.get(v, []):
        if not visited[neighbor]:
            if has_cycle(graph, neighbor, visited, stack):
                return True
        elif stack[neighbor]:
            return True
    
    stack[v] = False
    return False


# In[3]:


def detect_cycle(devices, N):
    graph = {}
    
    for i in range(1, N + 1):
        _, inputs = devices[i]
        for port in inputs:
            if port[0] == 'O':
                neighbor = int(port[1:])
                if neighbor not in graph:
                    graph[neighbor] = []
                graph[neighbor].append(i)
    
    visited = [False] * (N + 1)
    stack = [False] * (N + 1)
    
    for i in range(1, N + 1):
        if not visited[i]:
            if has_cycle(graph, i, visited, stack):
                return True
    
    return False


# In[11]:


if __name__ == "__main__":
    # 输入提示性语句
    print("请输入问题数量：")

    # 读取问题数量
    Q = int(input())

    # 处理每个问题
    for _ in range(Q):
        # 输入提示性语句
        print("请输入输入和器件的数量：")
    
        # 读取输入和器件的数量
        M, N = map(int, input().split())
    
        # 初始化存储器件的字典
        devices = {}
    
        # 读取器件信息
        for i in range(1, N + 1):
            # 输入提示性语句
            print(f"请输入第{i}个器件的信息：")
        
            # 读取器件信息
            device_info = input().split()
            func = device_info[0]
            k = int(device_info[1])
            inputs = device_info[2:]
            devices[i] = (func, inputs)
        
        # 检测是否存在环路
        if detect_cycle(devices, N):
            print("LOOP")
            continue
    
        # 输入提示性语句
        print("请输入运行次数：")
    
        # 读取运行次数
        S = int(input())
    
        # 存储所有运行次数的输入值
        all_inputs = []
        for i in range(1,S+1):
            # 输入提示性语句
            print( f"请输入第{i}次运行电路的输入值：")
        
            # 读取电路的输入值
            inputs = list(map(int, input().split()))
            all_inputs.append(inputs)
    
        # 读取所有运行次数的需要输出的器件编号及其对应的信号数量
        all_output_devices = []
        for i in range(1,S+1):
            # 输入提示性语句
            print(f"请输入第{i}次运行需要输出的器件编号及其对应的信号数量：")
            output_devices = list(map(int, input().split()))
            all_output_devices.append(output_devices)
        
        # 计算所有运行次数的结果
        all_results = []
        for inputs, output_devices in zip(all_inputs, all_output_devices):
            # 存储每个器件的输出结果
            outputs = {}
            for i in range(1, N + 1):
                func, in_ports = devices[i]
                in_values = []
                for port in in_ports:
                    if port[0] == 'I':  # 输入端连接的信号
                        in_values.append(inputs[int(port[1:]) - 1])
                    elif port[0] == 'O':  # 输入端连接的器件的输出信号
                        in_values.append(outputs[int(port[1:])])
                # 根据逻辑门的功能计算输出
                if func == 'AND':
                    outputs[i] = int(all(in_values))
                elif func == 'OR':
                    outputs[i] = int(any(in_values))
                elif func == 'XOR':
                    outputs[i] = int(sum(in_values) % 2)
                elif func == 'NAND':
                    outputs[i] = int(not all(in_values))
                elif func == 'NOR':
                    outputs[i] = int(not any(in_values))
                elif func == 'NOT':
                    outputs[i] = int(not in_values[0])
        
            # 存储当前运行次数的结果
            result = [outputs.get(dev, 0) for dev in output_devices[1:]]  # 检查器件是否存在，不存在则输出 0
            all_results.append(result)
    
        # 输出所有运行次数的结果
        for result in all_results:
            print(*result)