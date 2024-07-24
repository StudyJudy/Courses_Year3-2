# summary

故障监测模块，主要完成了：
1. 分布式的 RAFT 算法实现
2. 完整的 Logger 日志记录、Timer 定时器类实现
3. 基于 UDP 实现网络通信
4. 基于 Linux Filesystem 实现机器状态监测
5. 作为模块提供给外部，关于分布式节点的元信息（node_name+ip+port）可直接通过 `nodes.json` 进行配置

提供给平台的输出内容参考 main.py 中的调用。