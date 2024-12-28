import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

model_id = "/hub/huggingface/models/jbmodels/llama-guard-3-8B"
device = "cuda:0"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map='auto')

def get_prompts(file_name):
    data_list = []
    with open(file_name, 'r') as file:
        data = json.load(file)
        for item in data:
            data_list.append(item)
    return data_list

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

results_path = "/data/home/Weiwei/code/jailbreak/results/gpt-3.5-turbo/markov_8113_suffix_test.json"

demo_item_list = get_prompts(results_path)
index = 0
for item in tqdm(demo_item_list):
    ori_prompt = item["goal"]
    victim_response = item["best_victim_response"]
    res = moderate([
    {"role": "user", "content": ori_prompt},
    {"role": "assistant", "content": victim_response},
    ])
    if "unsafe" in res:
        index += 1
    print("Moderated response: ", res)
    print("="*50)
    
print("Number of unsafe responses: ", index)
