from code_review import CodeReviewer

# 调用模型获取代码审查结果
model = CodeReviewer()
res = model.code_review_task("file", r"test_files/example.py")
print(res)

# 对结果报告进行分类
print("error类型报告：")
errors = model.report_classification("error")
for error in errors:
    print(error)

# 对目录进行审查
dir_review = model.code_review_task("dir", r"test_files")
print(dir_review)
print("warning类型报告：")
dir_warnings = model.report_classification("warning")
for warning in dir_warnings:
    print(f"\n\n{warning}")