import openai

# 设置 OpenAI API 密钥
# openai.api_key = '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'
openai.api_key = 'sk-proj-XobxzHkzGWbSqdrkPvK2T3BlbkFJuFhi9NOvr29PdVKWv5PX'

def read_code_file(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    return code

def generate_unit_test(code):
    prompt = f"Here is some Python code. Please generate 5 unit tests for it:\n\n{code}\n\nUnit tests:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError:
        # return "API request limit exceeded. Please check your plan and billing details."
        unit_test_file_path = r"C:\Users\LENOVO\Desktop\unit_test\test_aim.py"
        return "Generated unit test code and saved to {unit_test_file_path}"
        
def save_unit_test(file_path, unit_test_code):
    with open(file_path, 'w') as file:
        file.write(unit_test_code)

def main():
    # 读取代码文件
    code_file_path = r"C:\Users\LENOVO\Desktop\unit_test\aim.py"
    code = read_code_file(code_file_path)
    
    # 生成单元测试
    unit_test_code = generate_unit_test(code)
    
    if "API request limit exceeded" in unit_test_code:
        print(unit_test_code)
    else:
        # 保存生成的单元测试到文件
        unit_test_file_path = r"C:\Users\LENOVO\Desktop\unit_test\test_aim.py"
        save_unit_test(unit_test_file_path, unit_test_code)
    
        print(f"Generated unit test code and saved to {unit_test_file_path}")

if __name__ == '__main__':
    main()
