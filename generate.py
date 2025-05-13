import torch
from model.poem_gpt import PoemGPT
import argparse
from transformers import BertTokenizer, TextGenerationPipeline
from train import get_args, get_tokenizer

@torch.no_grad()
def generate_poem(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'checkpoints/{args.epochs}_{args.filepath}', weights_only=False)
    tokenizer = get_tokenizer(args)

    model = PoemGPT(saved['args'], tokenizer)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    ses = "窗前明月光，"
    while ses != '0':
        encoding = tokenizer(ses, return_tensors='pt', padding=False, truncation=True).to(device) #(bs,sl) 词索引阶段
        # 需要把toenizer后面添加的特殊Token [SEP] 去掉
        token_ids, generated_output = model.generate_top_k(encoding['input_ids'][:, :-1],temperature=args.temperature,k_size=10)
        print(f'{generated_output}\n\n')
        ses = input("请输入要预测的诗句，输入0退出\n")

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
    generate_poem(args)