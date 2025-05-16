import numpy as np
import torch
import transformers
import re
import sys
from model.poem_gpt import PoemGPT
from transformers import BertTokenizer
sys.path.pop(0)  # 移除当前目录
import evaluate
import jieba
import os

print("---------")

def process_data():

    data = np.load("data/clear_data.npy")

    POEM_NUMBER = 400
    prompts1 = []
    prompts2 = []
    prompts3 = []
    candidate_texts = [] # 实际的诗歌


    for item in data[:POEM_NUMBER]:
        sens = re.split(r'[,.!，。！？]', item)
        if len(sens) <3:
            continue
        prompts1.append(item[:len(sens[0])+1])
        prompts2.append(item[:len(''.join(sens[:2]))+2])
        prompts3.append(item[:len(''.join(sens[:3]))+3])
        candidate_texts.append(item)

    greedy_poems = []
    beam_poems = []
    top_k_poems =[]
    top_p_poems = []

    device = 'cuda'
    saved = torch.load(f'checkpoints/20_20-0.001-sonnet.pt', weights_only=False)
    tokenizer = BertTokenizer.from_pretrained("cache/bert-tokenizer", local_files_only=True)
    model = PoemGPT(saved['args'], tokenizer)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    for i in range(POEM_NUMBER):
        print(i)
        encoding1 = tokenizer(prompts1[i], return_tensors='pt', padding=False, truncation=True).to(device) #(bs,sl) 词索引阶段
        encoding2 = tokenizer(prompts2[i], return_tensors='pt', padding=False, truncation=True).to(device) #(bs,sl) 词索引阶段
        encoding3 = tokenizer(prompts3[i], return_tensors='pt', padding=False, truncation=True).to(device) #(bs,sl) 词索引阶段
        # 需要把toenizer后面添加的特殊Token [SEP] 去掉
        # tok-k
        t = []
        token_ids1, generated_output1 = model.generate_top_k(encoding1['input_ids'][:, :-1],temperature=1.2,k_size=10)
        token_ids2, generated_output2 = model.generate_top_k(encoding2['input_ids'][:, :-1],temperature=1.2,k_size=10)
        token_ids3, generated_output3 = model.generate_top_k(encoding3['input_ids'][:, :-1],temperature=1.2,k_size=10)
        top_k_poems.append([generated_output1, generated_output2, generated_output3])
        # tok-q
        token_ids, generated_output1 = model.generate_top_q(encoding1['input_ids'][:, :-1],temperature=1.2, top_p=0.9)
        token_ids, generated_output2 = model.generate_top_q(encoding2['input_ids'][:, :-1],temperature=1.2, top_p=0.9)
        token_ids, generated_output3 = model.generate_top_q(encoding3['input_ids'][:, :-1],temperature=1.2, top_p=0.9)
        top_p_poems.append([generated_output1, generated_output2, generated_output3])
        # greedy search
        token_ids, generated_output1 = model.generate_greedy_search(encoding1['input_ids'][:, :-1])
        token_ids, generated_output2 = model.generate_greedy_search(encoding2['input_ids'][:, :-1])
        token_ids, generated_output3 = model.generate_greedy_search(encoding3['input_ids'][:, :-1])
        greedy_poems.append([generated_output1, generated_output2, generated_output3])
        # beam search
        token_ids, generated_output1 = model.generate_beam_search(encoding1['input_ids'][:, :-1])
        token_ids, generated_output2 = model.generate_beam_search(encoding2['input_ids'][:, :-1])
        token_ids, generated_output3 = model.generate_beam_search(encoding3['input_ids'][:, :-1])
        beam_poems.append([generated_output1, generated_output2, generated_output3])

    np.savez('data/rouge.npz',
             number=POEM_NUMBER,
             candidate_texts=candidate_texts,
             greedy_poems=greedy_poems,
             beam_poems=beam_poems,
             top_k_poems=top_k_poems,
             top_p_poems=top_p_poems)

candidate_texts = []

def compute(reference_texts):
    # 加载 ROUGE 评估器
    rouge_metric = evaluate.load('rouge')

    def char_tokenizer(text):
        return list(text.replace(" ", ""))

    print("--- 字级别 ROUGE ---")
    results_char = rouge_metric.compute(
        predictions=candidate_texts,
        references=reference_texts,
        tokenizer=char_tokenizer, 
    )
    print(f"字级别 ROUGE-1: {results_char['rouge1']:.4f}")
    print(f"字级别 ROUGE-2: {results_char['rouge2']:.4f}")

    def word_tokenizer_jieba(text):
        return jieba.lcut(text)

    print("\n--- 词级别 ROUGE (jieba) ---")
    results_word = rouge_metric.compute(
        predictions=candidate_texts,
        references=reference_texts,
        tokenizer=word_tokenizer_jieba,
        rouge_types=['rouge1', 'rouge2'],
    )
    print(f"词级别 ROUGE-1 (jieba): {results_word['rouge1']:.4f}")
    print(f"词级别 ROUGE-2 (jieba): {results_word['rouge2']:.4f}")


if __name__ == "__main__":
    data = np.load(r"data/rouge.npz")
    candidate_texts = data.get("candidate_texts")
    greedy_poems = data.get("greedy_poems")
    beam_poems = data.get("beam_poems")
    top_k_poems = data.get("top_k_poems")
    top_p_poems = data.get("top_p_poems")
    print("贪心搜索")
    compute(greedy_poems)
    print("beam")
    compute(beam_poems)
    print("top-k")
    compute(top_k_poems)
    print("top-p")
    compute(top_p_poems)