import torch
from model.poem_gpt import PoemGPT
import argparse
from transformers import BertTokenizer, TextGenerationPipeline
from train import get_args, get_tokenizer

GENERATE_METHOD = "top-q"

ses = ["春花十里","大好河山","自然语言处理"]

@torch.no_grad()
def generate_poem(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'checkpoints/20_20-0.001-sonnet.pt', weights_only=False)
    tokenizer = get_tokenizer(args)

    model = PoemGPT(saved['args'], tokenizer)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    i = 0
    with open("result/poems.txt", mode='a') as f:
        while i< 30:
            encoding = tokenizer(ses[i%3], return_tensors='pt', padding=False, truncation=True).to(device) #(bs,sl) 词索引阶段
            # 需要把toenizer后面添加的特殊Token [SEP] 去掉
            # tok-k
            # token_ids, generated_output = model.generate_top_k(encoding['input_ids'][:, :-1],temperature=args.temperature,k_size=10)
            # f.write(f'top-k\n{generated_output}\n\n')
            # # tok-q
            # token_ids, generated_output = model.generate_top_q(encoding['input_ids'][:, :-1],temperature=args.temperature, top_p=args.top_p)
            # f.write(f'top-p\n{generated_output}\n\n')
            # # greedy search
            # token_ids, generated_output = model.generate_greedy_search(encoding['input_ids'][:, :-1])
            # f.write(f'greedy search\n{generated_output}\n\n')
            # # beam search
            # token_ids, generated_output = model.generate_beam_search(encoding['input_ids'][:, :-1])
            # f.write(f'beam search\n{generated_output}\n\n')
            i += 1
            # 首字引导生成
            token_ids, generated_output = model.generate_hidden_poem(encoding['input_ids'][:,:-1], temperature=args.temperature, top_p=args.top_p)
            f.write(f'{generated_output}\n\n')
            # print(f'{generated_output}\n\n')
            # ses = input("请输入要预测的诗句，输入0退出\n")

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
    generate_poem(args)