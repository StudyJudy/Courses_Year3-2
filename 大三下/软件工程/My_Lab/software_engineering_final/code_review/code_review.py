import os
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-e0UY1pZjUkUQN6iR2MvYxFisSFBps2dh3KQ6uTaoyolXw5lc",
    base_url="https://api.chatanywhere.tech/v1"
)


OUTPUT_TEMPLATE = "| **输出信息** | **描述** |\n" \
                  "| :----: | :---------------------: |\n" \
                  "| 文件名称 | 发生问题的文件名称 |\n" \
                  "| 报告类型 | ERROR/WARNING/INFO |\n" \
                  "| 问题描述 | 描述问题的具体细节 |\n" \
                  "| 错误位置 | Line xx:`发生错误的代码` |\n" \
                  "| 问题类型 | 与问题相关的规则编号或问题类型 |\n" \
                  "| 问题严重性 | 问题的严重程度 |\n" \
                  "| 可能的后果 | 问题可能导致的潜在影响 |\n" \
                  "| 修复建议 | 提供解决问题的建议或方法和修改后的代码 |\n" \
                  "\n\n**修改示例** \n```针对出现问题的代码，给出具体的修改后的代码段```\n"
TEST_FILE_PATH = r"test_files/example.py"
TEST_DIR_PATH = r"test_files"
SUPPORT_FILE_TYPE = ["py", "java", "c", "cpp", "js", "ts", "go", "php", "rb", "swift", "kt", "rs", "scala", "html",
                     "css", "scss", "less", "sass", "json", "yaml", "xml", "md", "txt"]
SUPPORT_MODEL = ["gpt-3.5-turbo"]


class CodeReviewer:
    """
    代码审查器
    1. 支持从输入框中加载代码
    2. 支持从文件加载代码
    3. 支持从文件夹中加载代码
    """
    def __init__(self, model="gpt-3.5-turbo", method="file"):
        self.code = {}
        self.model = model
        self.load_method = method
        self.report = ''

    def set_model(self, model):
        self.model = model

    def set_load_method(self, method):
        """
        设置加载代码的方式
        :param method: 加载代码的方式
        method 选取范围：file, input, dir
        """
        self.load_method = method

    def load_code_from_input(self, code):
        """
        从输入框中加载代码
        :param code(str): 输入框中的代码字符串
        """
        # 从输入框中加载代码
        self.code["input code"] = code

    def load_code_from_file(self, path):
        """
        从文件加载代码
        :param path: 文件路径
        """
        with open(path, 'r', encoding='utf-8') as file:
            file_name = os.path.basename(path)
            self.code[file_name] = ''
            lines = file.readlines()
            for line in lines:
                self.code[file_name] += line

    def load_code_from_directory(self, path):
        """
        从文件夹中加载代码
        :param path: 文件夹路径
        """
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORT_FILE_TYPE):
                    file_path = os.path.join(root, file)
                    self.load_code_from_file(file_path)

    def get_query_message(self, name, code):
        """
        组装输入进模型的查询消息
        :param name: 文件名称
        :param code: 代码字符串
        :return: 查询消息
        """
        messages = [
            {"role": "system", "content": "你是一个有亲和力的代码审查专家，你需要根据提供的代码审查并给出针对代码中出现的每一个问题的尽可能详细的问题报告。"},
            {"role": "user", "content": f"请审查以下代码，给出详细的问题报告。代码中的每一个问题都要严格遵循固定格式进行输出，并在开头输出对应文件的文件名和问题总数，"
                                        f"并在最后输出对该文件的总体评价。一定注意不要生成给定格式以外的内容！\n"
                                        f"当前文件名称：{name}代码：\n{code}\n"
                                        f"输出格式：\n**文件名称**\t...\n**问题总数**\t...\n"
                                        f"**详细报告**\n\n**NO.x**\n{OUTPUT_TEMPLATE}（与问题总数数量相同的表格）\n\n**文件总体评价**\t...\n\n"},
            {"role": "assistant", "content": f"\n**文件名称**\t...\n**问题总数**\t...\n"
                                             f"**详细报告**\n\n**NO.x**\n{OUTPUT_TEMPLATE}（与问题总数数量相同的表格）"
                                             f"\n\n**文件总体评价**\t...\n\n"},
        ]
        return messages

    def query(self):
        """
        调用模型进行代码审查
        :return: 模型返回的答案
        """
        answers = ''
        for name, code in self.code.items():
            messages = self.get_query_message(name, code)
            if self.model == "gpt-3.5-turbo":
                answer = gpt_35_api(messages)
            else:
                answer = "模型调用失败！"
            answers = answers + answer + "\n\n"
        return answers

    def code_review_task(self, method: str, target: str):
        """
        代码审查任务
        :param method: 加载代码的方式
        :param target: 加载代码的目标参数
        :return: str类型的审查结果，使用markdown格式

        三种加载方式：
        method = 'file' 则 target 为 文件路径
        method = 'dir' 则 target 为 文件夹路径
        method = 'input' 则 target 为 代码字符串
        """
        # 加载代码
        self.code = {}
        self.report = ''
        if method == "file":
            self.load_code_from_file(target)
        elif method == "dir":
            self.load_code_from_directory(target)
        elif method == "input":
            self.load_code_from_input(target)
        else:
            return "代码加载失败！"
        answer = self.query()
        result = f"以下是AI生成的代码审查结果：\n" + answer + "\n"
        self.report = result
        return result

    def report_classification(self, report_type):
        """
        根据报告类型进行分类
        :param report_type: 报告类型 选取范围：error、warning、info
        :return: 分类后的相应类型报告list
        """
        if report_type not in ["error", "warning", "info"]:
            print("报告类型错误！")
            return []
        if self.report == '':
            return []
        split_res = self.report.split("\n\n")
        error = []
        warning = []
        info = []
        for i in split_res:
            if "NO." in i:
                if "ERROR" in i:
                    error.append(i)
                elif "WARNING" in i:
                    warning.append(i)
                elif "INFO" in i:
                    info.append(i)
                else:
                    continue

        if report_type == "error":
            return error
        elif report_type == "warning":
            return warning
        else:
            return info


# 非流式响应
def gpt_35_api(messages: list):
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return completion.choices[0].message.content


# 流式响应
def gpt_35_api_stream(messages: list):
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == '__main__':
    model = CodeReviewer()
    # res = model.code_review_task("dir", TEST_DIR_PATH)
    # print(res)
