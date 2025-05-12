import torch
from torch import nn
from model.gpt2 import GPT2Model
from transformers import BertTokenizer

class PoemGPT(nn.Module):

  def __init__(self, args, tokenizer):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model_name=args.model_name, model_dir=args.model_dir, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = tokenizer

    # 将最终的输出last_hidden_state转化为词汇表的概率分布
    self.vob_proj = nn.Linear(args.d, self.tokenizer.vocab_size)
    self.config = args
    self.device = "cuda" if args.use_gpu else "cpu"

    for name, param in self.gpt.named_parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    sequence_output, last_token = self.gpt(input_ids, attention_mask).values()
    return self.vob_proj(sequence_output)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    token_ids = encoding.to(self.get_device())
    # 1表示正常的输入 0表示pad的值
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    # 去除前3个token，一般前3个token为特殊的token，比如说[CLS]、[BOS]
    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())
    return token_ids, generated_output