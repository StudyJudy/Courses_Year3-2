#!/usr/bin/env python
# coding: utf-8

"""
This module contains functions for detecting cycles in a circuit and
calculating its output for a given set of inputs.
"""

def has_cycle(graph, vertex, visited, stack):
    """
    Check if a graph has a cycle starting from vertex v.

    Args:
        graph (dict): The graph representation.
        vertex (int): The current vertex.
        visited (list): List to mark visited vertices.
        stack (list): List to track vertices in the current path.

    Returns:
        bool: True if a cycle is found, False otherwise.
    """
    visited[vertex] = True
    stack[vertex] = True
    for neighbor in graph.get(vertex, []):
        if not visited[neighbor]:
            if has_cycle(graph, neighbor, visited, stack):
                return True
        elif stack[neighbor]:
            return True
    stack[vertex] = False
    return False

def detect_cycle(circuit_devices, num_devices):
    """
    Detect if there is a cycle in the circuit represented by devices.

    Args:
        circuit_devices (dict): Dictionary representing the circuit.
        num_devices (int): Number of devices.

    Returns:
        bool: True if a cycle is detected, False otherwise.
    """
    graph = {}
    for device_id in range(1, num_devices + 1):
        _, inputs_ports = circuit_devices[device_id]
        for port in inputs_ports:
            if port[0] == 'O':
                neighbor = int(port[1:])
                if neighbor not in graph:
                    graph[neighbor] = []
                graph[neighbor].append(device_id)
    visited = [False] * (num_devices + 1)
    stack = [False] * (num_devices + 1)
    for device_id in range(1, num_devices + 1):
        if not visited[device_id]:
            if has_cycle(graph, device_id, visited, stack):
                return True
    return False

def main():
    """
    Main function to handle input and output.
    """
    # 输入提示性语句
    # print("请输入问题数量：")
    # 读取问题数量
    num_questions = int(input())
    # 处理每个问题
    for _ in range(num_questions):
        # 输入提示性语句
        # print("请输入输入和器件的数量：")
        # 读取输入和器件的数量
        _, devices_num = map(int, input().split())
        # 初始化存储器件的字典
        devices = {}
        # 读取器件信息
        for i in range(1, devices_num + 1):
            # 输入提示性语句
            # print(f"请输入第{i}个器件的信息：")
            # 读取器件信息
            device_info = input().split()
            func = device_info[0]
            # k = int(device_info[1])
            inputs = device_info[2:]
            devices[i] = (func, inputs)
        # 检测是否存在环路
        if detect_cycle(devices, devices_num):
            print("LOOP")
            continue
        # 输入提示性语句
        # print("请输入运行次数：")
        # 读取运行次数
        num_runs = int(input())
        # 存储所有运行次数的输入值
        all_inputs = []
        for i in range(1, num_runs + 1):
            # 输入提示性语句
            # print( f"请输入第{i}次运行电路的输入值：")
            # 读取电路的输入值
            inputs = list(map(int, input().split()))
            all_inputs.append(inputs)
        # 读取所有运行次数的需要输出的器件编号及其对应的信号数量
        all_output_devices = []
        for i in range(1, num_runs + 1):
            # 输入提示性语句
            # print(f"请输入第{i}次运行需要输出的器件编号及其对应的信号数量：")
            output_devices = list(map(int, input().split()))
            all_output_devices.append(output_devices)
        # 计算所有运行次数的结果
        all_results = []
        for inputs, output_devices in zip(all_inputs, all_output_devices):
            # 存储每个器件的输出结果
            outputs = {}
            for i in range(1, devices_num + 1):
                func, in_ports = devices[i]
                in_values = []
                for input_port in in_ports:
                    if input_port[0] == 'I':  # 输入端连接的信号
                        in_values.append(inputs[int(input_port[1:]) - 1])
                    elif input_port[0] == 'O':  # 输入端连接的器件的输出信号
                        in_values.append(outputs[int(input_port[1:])])
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

if __name__ == "__main__":
    main()
            