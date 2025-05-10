import torch
from model.poem_gpt import PoemGPT
import argparse
from transformers import BertTokenizer, TextGenerationPipeline
from train import get_args, get_tokenizer

@torch.no_grad()
def generate_poem(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs}_{args.filepath}', weights_only=False)
    tokenizer = get_tokenizer(args)

    model = PoemGPT(saved['args'], tokenizer)
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()

    ses = "窗前明月光，"
    while ses != '0':
        encoding = tokenizer(f'{ses}', return_tensors='pt', padding=False, truncation=True).to(device)
        output = model.generate(encoding['input_ids'][:, :-1],temperature=args.temperature,top_p=args.top_p)[0][0]
        decoded_output: str = tokenizer.decode(output)
        print(f'{decoded_output[:decoded_output.index("[SEP]")+5]}\n\n')
        ses = input("请输入要预测的诗句，输入0退出\n")

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
    generate_poem(args)


