import json

# 假设你的 JSON 文件名为 data.json
# file_name = '/data/home/Weiwei/code/jailbreak/results/llama3/markov_test.json' # 402/50 = 8.04 ; ASR = 0.88
# file_name = '/data/home/Weiwei/code/jailbreak/results/gemma-2-9b/markov_test.json' # 602/50 = 12.04 ; ASR = 0.84
# file_name = '/data/home/Weiwei/code/jailbreak/results/llama2/markov_test.json' # 752/50 = 15.00 ; ASR = 0.70
# file_name = '/data/home/Weiwei/code/jailbreak/results/gpt-3.5-turbo/markov_8113_suffix_test.json' # 547/50 = 10.94 ; ASR = 0.98
# file_name = '/data/home/Weiwei/code/jailbreak/results/llama3/markov_8113_suffix_test.json'# 554/50 = 11.08 ; ASR = 0.92
# file_name = '/data/home/Weiwei/code/jailbreak/results/gpt-4o/markov_8113_suffix_test.json' # 637/50 = 12.74 ; ASR = 0.96
# file_name = '/data/home/Weiwei/code/jailbreak/results/gemini-1.5-pro/markov_8113_suffix_test.json' # 653/50 = 13.06 ; ASR = 0.94
file_name = '/data/home/Weiwei/code/jailbreak/results/qwen2.5-7B-it/markov_8113_suffix_test.json' # 563/50 = 11.26 ; ASR = 0.92


try:
    # 读取 JSON 文件
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化总和和计数
    total_query_num = 0
    best_score_count = 0

    # 遍历 JSON 对象，计算 query_num 总和和 best_score 等于 1.0 的项数
    for item in data:
        if 'query_num' in item and isinstance(item['query_num'], (int, float)):
            total_query_num += item['query_num']
        if 'best_score' in item and item['best_score'] == 1.0:
            best_score_count += 1

    print(f"所有单元格的 query_num 总和是: {total_query_num}")
    print(f"best_score 等于 1.0 的项数是: {best_score_count}")

except FileNotFoundError:
    print("文件未找到，请确认文件路径和文件名是否正确！")
except json.JSONDecodeError:
    print("JSON 文件格式有误，请检查文件内容！")
except Exception as e:
    print(f"发生错误: {e}")