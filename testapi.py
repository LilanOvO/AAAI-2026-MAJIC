from openai import OpenAI
attacktype = "gemini-1.5-pro"
victim_messages = [
    {"role": "user", "content": "which type are you from ? claude or gpt?"},
]

API_SECRET_KEY= "sk-wvdsTSCg4Edw5do22c734f65145349A99b48Ed06202cD59a" # 填写我们给您的apikey
BASE_URL = "https://api.ai-gaochao.cn/v1"
print("attacking.....")
gpt_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
victim_response = gpt_client.chat.completions.create(model=attacktype, messages=victim_messages, max_tokens=512).choices[0].message.content
print("attacking over.....")
print("answer is ",victim_response)

# import anthropic

# # 模拟的参数设置
# API_SECRET_KEY = "sk-wvdsTSCg4Edw5do22c734f65145349A99b48Ed06202cD59a"  # 填写我们给您的apikey
# BASE_URL = "https://api.ai-gaochao.cn/v1"
# attacktype = "claude-3-5-sonnet-20240620"  # 模拟模型的名字，可以根据需要更改
# disguised_prompt = "What is the capital of France?"

# # 初始化 Claude 客户端
# claude_client = anthropic.Anthropic(api_key=API_SECRET_KEY, base_url=BASE_URL)

# # 构造 victim_messages
# victim_messages = [
#     {"role": "user", "content": [{"type": "text", "text": disguised_prompt}]}
# ]

# # 测试代码
# try:
#     victim_response = claude_client.messages.create(
#         model=attacktype,
#         messages=victim_messages,
#         max_tokens=512,
#         temperature=1.0
#     ).content[0].text
#     print(f"Response from Claude: {victim_response}")  # 输出 Claude 返回的内容
# except Exception as e:
#     print(f"Error: {e}")
#     victim_response = "api error"

# # 测试结果打印
# if victim_response != "api error":
#     print("API call succeeded.")
# else:
#     print("API call failed.")
