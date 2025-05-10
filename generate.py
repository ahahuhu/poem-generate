import torch
from model.poem_gpt import PoemGPT
import argparse
from transformers import BertTokenizer, TextGenerationPipeline

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--poem_path", type=str, default="data/clear_data.npy")
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')
    
    # Generation parameters.
    parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
    parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.", default=0.9)
    
    parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("--model_name", type=str, help="The model size as specified on hugging face.", default='uer/gpt2-chinese-cluecorpussmall')
    
    args = parser.parse_args()
    return args

@torch.no_grad()
def generate_poem(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(f'{args.epochs}_{args.filepath}', weights_only=False)

    model = PoemGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model = model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

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


