#!/bin/bash

# 检查是否提供了文件名参数
if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

# 获取传入的文件名参数
FILENAME=$1

# 执行 SonarScanner 命令
/opt/sonar-scanner/bin/sonar-scanner -X \
  -Dsonar.projectKey=test001_ \
  -Dsonar.sources=./"$FILENAME" \
  -Dsonar.host.url=http://127.0.0.1:9000 \
  -Dsonar.login=140b663a6653643593dbba1ea64291eb3f60adfd

