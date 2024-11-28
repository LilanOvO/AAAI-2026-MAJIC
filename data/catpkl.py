import pickle

# 打开pkl文件，'rb'表示以二进制读取模式打开
# with open('mistral_demonstration_list_official.pkl', 'rb') as f:
with open('llama2_instruction_list.pkl', 'rb') as f:
    data = pickle.load(f)

# 打印文件内容
# print(data)

#  查看数据的类型
print(type(data))

# # 如果数据是字典，可以这样访问：
# if isinstance(data, dict):
#     for key, value in data.items():
#         print(f"{key}: {value}")

# 如果数据是列表，可以这样访问：
if isinstance(data, list):
    for item in data:
        print(item)
        print("\n")
        print('=' * 50)