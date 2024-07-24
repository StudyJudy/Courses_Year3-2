# test_code_reviewer.py
import pytest
from code_review import CodeReviewer
import os

TEST_FILE_PATH = r"test_files/example.py"
TEST_DIR_PATH = r"test_files"
TEST_CODE = """
print(hello)
"""
TEST_RES = r"test_result"


class TestCodeReviewer:
    @pytest.fixture
    def code_reviewer(self):
        return CodeReviewer()

    def test_code_review_task_file(self, tmp_path):
        if not os.path.exists(TEST_RES):
            os.makedirs(TEST_RES)

        # 执行code_review_task方法，模拟从文件加载代码
        model = CodeReviewer()
        res = model.code_review_task("file", TEST_FILE_PATH)
        # 将res写入到TEST_RES目录下的文件中
        output_file_path = os.path.join(TEST_RES, "test_code_review_file_result.txt")
        with open(output_file_path, 'w', encoding="utf-8") as file:
            file.write(res)

        # 验证返回结果是否包含代码加载成功的提示
        assert "以下是AI生成的代码审查结果" in res

    def test_code_review_task_dir(self, tmp_path):
        if not os.path.exists(TEST_RES):
            os.makedirs(TEST_RES)
        # 调用模型获取代码审查结果
        model = CodeReviewer()
        res = model.code_review_task("dir", TEST_DIR_PATH)
        # 将res写入到TEST_RES目录下的文件中
        output_file_path = os.path.join(TEST_RES, "test_code_review_dir_result.txt")
        with open(output_file_path, 'w', encoding="utf-8") as file:
            file.write(res)

        # 验证返回结果是否包含"代码加载成功"的提示
        assert "以下是AI生成的代码审查结果" in res

    def test_code_review_task_input(self, tmp_path):
        if not os.path.exists(TEST_RES):
            os.makedirs(TEST_RES)
        # 调用模型获取代码审查结果
        model = CodeReviewer()
        res = model.code_review_task("input", TEST_CODE)
        # 将res写入到TEST_RES目录下的文件中
        output_file_path = os.path.join(TEST_RES, "test_code_review_input_result.txt")
        with open(output_file_path, 'w', encoding="utf-8") as file:
            file.write(res)

        # 验证返回结果是否包含"代码加载成功"的提示
        assert "以下是AI生成的代码审查结果" in res

    def test_code_review_task_invalid_method(self):
        if not os.path.exists(TEST_RES):
            os.makedirs(TEST_RES)
        # 使用无效的加载方法
        model = CodeReviewer()
        res = model.code_review_task("invalid", '')

        # 验证返回结果是否包含错误提示
        assert "代码加载失败！" in res

    def test_report_classification(self):
        model = CodeReviewer()
        res = model.code_review_task("file", TEST_FILE_PATH)
        errors = model.report_classification("error")
        warnings = model.report_classification("warning")
        infos = model.report_classification("info")
        # 验证返回结果是否包含错误、警告和信息提示
        for item in errors:
            assert "ERROR" in item
        for item in warnings:
            assert "WARNING" in item
        for item in infos:
            assert "INFO" in item


# 运行pytest时收集的类和方法
if __name__ == "__main__":
    pytest.main()
