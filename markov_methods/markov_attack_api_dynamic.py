from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
from tqdm import tqdm
from judgeutils import judge_gpt,judge_llama2
import os
from m1_hypo_attackLLM import hypo_method
from m2_history_attackLLM import history_method
from m3_space_attackLLM import space_method
from m4_dialogue_attackLLM import reverse_method
from m5_security_attackLLM import security_method
from m6_word_attackLLM import word_method
from m7_char_attackLLM import char_method
from m8_literary_attackLLM import literary_method
from m9_language_attackLLM import language_method
from m10_emoji_attack import emoji_method
import random
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "x"
from openai import OpenAI   

gamma = 0.5
alpha = 0.1
success_reward = 1
fail_reward = 0
beta = 0.01

def update_mtm_and_softmax(matrix, prev_idx, current_idx, reward):
    if prev_idx is None: 
        print("Info: prev_idx is None, skipping MTM update for this step.")
        return

    if not (0 <= prev_idx < matrix.shape[0] and
            0 <= current_idx < matrix.shape[1]):
        print(f"Error: Invalid indices for MTM update. Prev: {prev_idx}, Curr: {current_idx}")
        return

    current_prob_ij = matrix[prev_idx, current_idx]

    if matrix.shape[1] > 0:
        max_prob_next_state = np.max(matrix[current_idx, :])
    else:
        max_prob_next_state = 0.0      
    q_target_component = reward + gamma * max_prob_next_state
    new_logit_for_ij = current_prob_ij + alpha * (q_target_component - current_prob_ij)
    row_to_update_with_new_logit = matrix[prev_idx, :].copy()
    row_to_update_with_new_logit[current_idx] = new_logit_for_ij
    matrix[prev_idx, :] = softmax_row(row_to_update_with_new_logit)
    new_matrix = matrix
    
    return new_matrix

def softmax_row(x_row: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    x_row_adj = x_row / temperature 
    if x_row_adj.ndim != 1:
        raise ValueError("Input for _softmax_row must be a 1D array.")
    if x_row_adj.size == 0: return x_row_adj
    max_val = np.max(x_row_adj)
    e_x = np.exp(x_row_adj - max_val)
    sum_e_x = np.sum(e_x)
    if sum_e_x == 0:
        return np.ones_like(x_row_adj) / x_row_adj.size if x_row_adj.size > 0 else x_row_adj
    return e_x / sum_e_x

def get_attacker_model_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        temperature=1.00
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024,do_sample=True,temperature=0.9)
    return pipe

def get_model_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024)
    return pipe

def get_prompts(file_name):
    data_list = []
    with open(file_name, 'r') as file:
        data = json.load(file)
        for item in data:
            data_list.append(item)
    return data_list

def select_init_method(suffix,init_num,harmful_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,attacktype,iter_num,llama2model = None,llama2tokenizer = None):
    methods = {
        1: lambda: hypo_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype,attacktype,iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        2: lambda: history_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        3: lambda: space_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        4: lambda: reverse_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        5: lambda: security_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        6: lambda: word_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        7: lambda: char_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        8: lambda: literary_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        9: lambda: language_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
        10: lambda: emoji_method(suffix,harmful_prompt, attacker_pipe, attacker_tokenizer, victim_pipe, victim_tokenizer, judgetype, attacktype, iter_num,llama2model = llama2model,llama2tokenizer = llama2tokenizer),
    }

    if init_num in methods:
        return methods[init_num]() 
    else:
        raise ValueError(f"Unsupported init_num: {init_num}. Please provide a number between 1 and 10.")
    
def select_optimize_method(suffix,failed_num,optimize_num,harmful_prompt,disguised_failed_sentence,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judgetype,attacktype,iter_num,llama2model = None,llama2tokenizer = None):
    best_score = -0.1
    best_disguised_prompt = "xxx"
    best_victim_response = "xxx"
    
    for i in range(iter_num):
        harmful_prompt = harmful_prompt
        disguised_failed_sentence = disguised_failed_sentence
        simple_description = df[df['id'] == failed_num]['simple_description'].iloc[0]
        optimize_description = df[df['id'] == optimize_num]['optimize_description'].iloc[0]
        optimize_prompt = simple_description + '\n' + optimize_description
        optimize_prompt = optimize_prompt.replace("{harmful_prompt}", harmful_prompt)
        optimize_prompt = optimize_prompt.replace("{disguised_failed_sentence}", disguised_failed_sentence)
        print("optimize_prompt: ", optimize_prompt)
        optimize_messages = [
            {"role": "user", "content": optimize_prompt},
        ]
        optimized_disguised_prompt = attacker_pipe(optimize_messages)[0]['generated_text'][1]['content']
        optimized_disguised_prompt = optimized_disguised_prompt.strip().strip('\"')
        
        if suffix != "none":
            optimized_disguised_prompt = optimized_disguised_prompt + " " + suffix
        
        victim_messages = [
            {"role": "user", "content": optimized_disguised_prompt},
        ]
        
        if attacktype == "local":
            victim_response = victim_pipe(victim_messages)[0]['generated_text'][1]['content']
        else :    
            if attacktype[:3] == "gpt":  
                API_SECRET_KEY= "xxx" 
                BASE_URL = "xxx"
                gpt_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
                victim_messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": optimized_disguised_prompt},
                ]
                try:
                    victim_response = gpt_client.chat.completions.create(model=attacktype, messages=victim_messages, max_tokens=512).choices[0].message.content
                except Exception as e:
                    print(f"Error: {e}")
                    victim_response = "api error"
                finally:
                    import time
                    time.sleep(2)
            else:
                API_SECRET_KEY= "xxx" 
                BASE_URL = "xxx"
                gpt_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
                victim_messages = [
                {"role": "user", "content": optimized_disguised_prompt},
                ]
                try:
                    victim_response = gpt_client.chat.completions.create(model=attacktype, messages=victim_messages, max_tokens=512).choices[0].message.content
                except Exception as e:
                    print(f"Error: {e}")
                    victim_response = "api error"
                finally:
                    import time
                    time.sleep(2)
        
        if judgetype == "gpt":
            jailbreak_score = judge_gpt(victim_pipe, harmful_prompt, optimized_disguised_prompt, victim_response, victim_tokenizer)
        elif judgetype == "llama2":
            model = llama2model
            tokenizer = llama2tokenizer
            jailbreak_score = judge_llama2(model,tokenizer,harmful_prompt,victim_response)
            
        if jailbreak_score > best_score:
            best_score = jailbreak_score
            best_disguised_prompt = optimized_disguised_prompt
            best_victim_response = victim_response
        if best_score == 1.0:
            break
    
    return best_score,best_disguised_prompt,best_victim_response
           
def generate_number_by_probability(init_vector):
    if not (len(init_vector) == 10 and abs(sum(init_vector) - 1) < 1e-2):
        raise ValueError("init_vector must be a list of 10 probabilities summing to 1.")

    return random.choices(range(1, 11), weights=init_vector, k=1)[0]

def softmax_normalize_with_temperature(ratios, temperature=1.0): 
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    ratios = np.array(ratios)
    exp_ratios = np.exp((ratios) / temperature) 
    return exp_ratios / np.sum(exp_ratios)

if __name__ == '__main__':

    demo_item_list = get_prompts("xxx")
    
    results_path = "xxx"

    attacker_model_id = "xxx"
    attacker_pipe = get_attacker_model_inference_pipeline(attacker_model_id)
    attacker_tokenizer = AutoTokenizer.from_pretrained(attacker_model_id)

    # victim_model_id = "xxx"
    # victim_pipe = get_model_inference_pipeline(victim_model_id)
    # victim_tokenizer = AutoTokenizer.from_pretrained(victim_model_id)
    # attack_type = "local"
    
    attack_type = "xxx"
    victim_pipe = None
    victim_tokenizer = None
    
    
    matrix = np.load("xxx.npy")
    
    column_sums = np.sum(matrix, axis=0)
    init_vector = softmax_normalize_with_temperature(column_sums, temperature=1.0)
    df = pd.read_csv('methods.csv')
    
    chain_count = 10
    init_qnum = 1
    chain_qnum = 1
    chain_length = 3
    sum_queries = 0
    beta = 0.01
    
    suffix = "none"
    
    
    # judge_type = "gpt"
    # llama2model = None
    # llama2tokenizer = None
    
    judge_type = "llama2"
    llama2model = AutoModelForCausalLM.from_pretrained("xxx", torch_dtype=torch.bfloat16, device_map="auto")
    llama2tokenizer = AutoTokenizer.from_pretrained("xxx", use_fast=False, truncation_side="left", padding_side="left")
    
    index = 0
    for item in tqdm(demo_item_list):
        index += 1
        
        item_query = 0
        chain_score = -0.1
        chain_disguised_prompt = "xxx"
        chain_victim_response = "xxx"
        
        for k in range(chain_count):
            harmful_prompt = item["goal"]
            # init_vector = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 
            init_num = generate_number_by_probability(init_vector) 
            item_query += init_qnum
            sum_queries += init_qnum
            chain_score,chain_disguised_prompt,chain_victim_response = select_init_method(suffix,init_num,harmful_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judge_type,attack_type,iter_num=init_qnum,llama2model = llama2model,llama2tokenizer = llama2tokenizer)
            if chain_score == 1.0:
                item["best_score"] = chain_score
                item["best_disguised_prompt"] = chain_disguised_prompt
                item["best_victim_response"] = chain_victim_response
                item["type"] = "init"
                with open(results_path, 'w') as file:
                    json.dump(demo_item_list, file, indent=4)  
                break
            failed_num = init_num
            failed_disguised_prompt = chain_disguised_prompt
            failed_score = chain_score
            
            for i in range(chain_length):
                optimize_vector = matrix[failed_num-1]

                optimize_num = generate_number_by_probability(optimize_vector)

                item_query += chain_qnum
                sum_queries += chain_qnum
                chain_score,chain_disguised_prompt,chain_victim_response = select_optimize_method(suffix,failed_num,optimize_num,harmful_prompt,failed_disguised_prompt,attacker_pipe,attacker_tokenizer,victim_pipe,victim_tokenizer,judge_type,attack_type,iter_num=chain_qnum,llama2model = llama2model,llama2tokenizer = llama2tokenizer)

                if chain_score == 1.0:
                    if sum_queries % 40 == 0:
                        alpha *= 0.95
                    if sum_queries % 80 == 0:
                        uniform_matrix = np.full_like(matrix, 1 / 10)
                        matrix = (1 - beta) * matrix + beta * uniform_matrix
                    matrix = update_mtm_and_softmax(matrix,failed_num-1,optimize_num-1,success_reward)
                    item["best_score"] = chain_score
                    item["best_disguised_prompt"] = chain_disguised_prompt
                    item["best_victim_response"] = chain_victim_response
                    break
                else:
                    if sum_queries % 40 == 0:
                        alpha *= 0.95
                    if sum_queries % 80 == 0:
                        uniform_matrix = np.full_like(matrix, 1 / 10)
                        matrix = (1 - beta) * matrix + beta * uniform_matrix
                    matrix = update_mtm_and_softmax(matrix,failed_num-1,optimize_num-1,fail_reward)
                    
                    failed_num = optimize_num
                    failed_disguised_prompt = chain_disguised_prompt

            if chain_score == 1.0:
                break    
                
        item["best_score"] = chain_score
        item["best_disguised_prompt"] = chain_disguised_prompt
        item["best_victim_response"] = chain_victim_response
        item["query_num"] = item_query

        with open(results_path, 'w') as file:
            json.dump(demo_item_list, file, indent=4)     
    
