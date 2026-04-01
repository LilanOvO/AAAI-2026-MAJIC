import os
import json
import numpy as np

def calculate_best_score_ratios(folder_path):
    ratios = []
    for file_name in sorted([f for f in os.listdir(folder_path) if f.startswith('results_f') and f.endswith('.json')]):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data:
                ratios.append(0.0)  
                continue
            total_count = len(data)
            best_score_count = sum(1 for item in data if item.get("best_score") == 1.0)
            ratio = best_score_count / total_count if total_count > 0 else 0.0
            ratios.append(ratio)
    if ratios:
        ratios = ratios[1:] + ratios[:1]
    
    return ratios

def softmax_normalize(ratios):
    exp_ratios = np.exp(ratios)  
    return exp_ratios / np.sum(exp_ratios)  

def softmax_normalize_with_temperature(ratios, temperature=1.0):
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    ratios = np.array(ratios)
    exp_ratios = np.exp((ratios) / temperature) 
    return exp_ratios / np.sum(exp_ratios)

def sum_normalize(ratios):
    total = sum(ratios)
    if total == 0:
        return [0.0 for _ in ratios]
    return [x / total for x in ratios] 

def power_normalize(ratios, gamma=2):
    amplified = [x ** gamma for x in ratios]  
    total = sum(amplified) 
    if total == 0:
        return [0.0 for _ in ratios]  
    return [x / total for x in amplified] 

def main():
    base_path = './'
    folder_names = [f"f{i+1}_{suffix}" for i, suffix in enumerate(
        ["hypo", "history", "space", "reverse", "security", "word", "char", "literary", "language", "emoji"]
    )]
    
    all_normalized_ratios = [] 

    for folder_name in folder_names:
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.exists(folder_path):
            all_normalized_ratios.append([0.0] * 10) 
            continue
        ratios = calculate_best_score_ratios(folder_path)
        normalized_ratios = softmax_normalize_with_temperature(ratios,temperature=0.5)
        all_normalized_ratios.append(normalized_ratios)

    matrix = np.array(all_normalized_ratios)
    matrix = np.round(matrix, decimals=3)
    np.save("xxx", matrix)

    for row in matrix:
        formatted_row = ["{:.3f}".format(x) for x in row]  
        print(" ".join(formatted_row))

if __name__ == "__main__":
    main()