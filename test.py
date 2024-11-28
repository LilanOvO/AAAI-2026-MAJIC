import json
import os
import re

def analyze_json_files(directory, keyword1="results", keyword2="gpt4o"):
    """
    分析指定目录下的 JSON 文件，匹配包含特定关键词的文件名，
    提取 "best_score" 为 1.0 的条目中的 ID，并计算覆盖率。

    Args:
        directory: 包含 JSON 文件的目录路径。
        keyword1: 文件名匹配的第一个关键词。
        keyword2: 文件名匹配的第二个关键词。

    Returns:
        一个字典，包含以下信息：
        - "all_covered_ids": 所有匹配文件中覆盖的唯一 ID 的集合。
        - "file_covered_ids": 一个字典，将文件名映射到每个文件中覆盖的 ID 集合。
        - "coverage_percentage": 所有匹配文件中覆盖的 ID 的百分比（0-49）。
    """

    all_covered_ids = set()  # 创建一个空集合，用于存储所有文件中 best_score 为 1.0 的 ID
    file_covered_ids = {}  # 创建一个空字典，用于存储每个文件覆盖的 ID
    all_possible_ids = set(range(50))  # 创建一个包含 0 到 49 所有可能 ID 的集合

    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件名是否包含 keyword1 和 keyword2
        if keyword1 in filename and keyword2 in filename:
            try:
                filepath = os.path.join(directory, filename)  # 构造完整的文件路径
                # 打开文件，使用 utf-8 编码读取
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)  # 将 JSON 文件内容加载为 Python 对象（通常是列表或字典）

                covered_ids = set()  # 创建一个空集合，用于存储当前文件覆盖的 ID
                # 遍历 JSON 数据中的每个条目
                for item in data:
                    # 检查条目中是否存在 "best_score" 键，并且其值为 1.0
                    if "best_score" in item and item["best_score"] == 1.0:
                        covered_ids.add(item["id"])  # 将符合条件的 ID 添加到 covered_ids 集合

                all_covered_ids.update(covered_ids)  # 将当前文件覆盖的 ID 添加到 all_covered_ids 集合
                file_covered_ids[filename] = covered_ids  # 将当前文件覆盖的 ID 存储到 file_covered_ids 字典中

            # 捕获 JSON 解码错误
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}")  # 打印 JSON 解码错误信息
            # 捕获其他异常
            except Exception as e:
                print(f"Error processing {filename}: {e}")  # 打印其他错误信息


    # 计算覆盖率百分比
    coverage_percentage = (len(all_covered_ids) / len(all_possible_ids)) * 100
    
    sorted_file_covered_ids = dict(sorted(file_covered_ids.items(), key=lambda item: int(re.search(r"_(\d+)_", item[0]).group(1))))

    # 返回结果字典
    return {
        "all_covered_ids": all_covered_ids,
        "file_covered_ids": sorted_file_covered_ids,
        "coverage_percentage": coverage_percentage
    }


# 示例用法（将 "." 替换为你的目录）
directory = "./results"  # 当前目录
results = analyze_json_files(directory)


print("Covered IDs in each file:")
for filename, ids in results["file_covered_ids"].items():
    print(f"{filename}: {ids}")

print("\nAll covered IDs:", results["all_covered_ids"])
print(f"\nCoverage percentage: {results['coverage_percentage']:.2f}%")