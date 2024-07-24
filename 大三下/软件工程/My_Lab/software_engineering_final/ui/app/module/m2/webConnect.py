import paramiko
import time
import os

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

host = "192.168.123.129"
port = 22
username = "sonar"
password = "123456"

def upload_file(sftp, filename):
    # print(os.getcwd())
    # 这里调用的路径是项目根文件夹下
    local_file_path = f"uploads/{filename}"
    remote_file_path = f"/opt/sonarqube/bin/linux-x86-64/{filename}"
    sftp.put(local_file_path, remote_file_path)


def CodeEvaluation(filename):
    ssh.connect(host, port, username, password)
    shell = ssh.invoke_shell()
    sftp = ssh.open_sftp()
    upload_file(sftp, filename)
    sftp.close()

    exec_commands = [
        "cd /opt/sonarqube/bin/linux-x86-64/",
        "./sonar.sh start",
        'export SONAR_SCANNER_OPTS="--add-opens java.base/java.lang=ALL-UNNAMED"',
        # "/opt/sonar-scanner/bin/sonar-scanner -X \
        # -Dsonar.projectKey=test001_ \
        # -Dsonar.sources=./main.py \
        # -Dsonar.host.url=http://127.0.0.1:9000 \
        # -Dsonar.login=140b663a6653643593dbba1ea64291eb3f60adfd",
        "bash run_sonar.sh " + filename,
        "rm " + filename,
    ]

    output = ""
    for exec_command in exec_commands:
        shell.send(exec_command + "\n")
        while True:
            if shell.recv_ready():
                output += shell.recv(1024).decode()
                if output.endswith("$ "):
                    break
            else:
                time.sleep(0.1)

    shell.close()
    ssh.close()
    return output

if __name__ == '__main__':
    print(CodeEvaluation("4_run.py"))